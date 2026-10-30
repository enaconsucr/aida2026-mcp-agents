"""
run_experiment.py — Runs the benchmark question set against a provider
(OpenAI or Gemini), executes the real tools the model requests, and saves a
structured per-question log with everything needed to compute the four
metrics from Section 4.3 of the paper:
    tool-selection accuracy, tool-execution success, answer accuracy, latency.
(chaining success rate is computed separately, from "chained"-type questions
— it requires comparing against the (a)/(b)/(c) ground truth, see Section
4.1.1 of the paper.)

USAGE:
    Load OPENAI_API_KEY or GOOGLE_API_KEY securely through a hidden
    interactive prompt; do not hard-code or expose API keys in commands.
    python run_experiment.py --questions questions.json --provider openai --model gpt-4o

BEFORE RUNNING THE REAL EXPERIMENT:
    1. Freeze RESTRICT_TO_WHITELIST=True in worldbank_tool.py and oecd_tool.py.
    2. Confirm that query_local_reviews is implemented (see local_reviews_tool.py).
    3. Confirm the Costa Rica OECD validation (see the project's README.md).
    4. Do not modify the tools once the benchmark has started, to avoid
       contaminating the evaluation across runs.

Corre el set de preguntas del benchmark contra un proveedor (OpenAI o
Gemini), ejecuta las herramientas reales que el modelo solicite, y guarda un
log estructurado por pregunta con todo lo necesario para calcular las 4
métricas de la Sección 4.3 del paper:
    tool-selection accuracy, tool-execution success, answer accuracy, latency.
(chaining success rate se calcula aparte, a partir de las preguntas tipo
"chained" — requiere comparar contra el ground truth de (a)/(b)/(c), ver
Sección 4.1.1 del paper).

USO:
    Cargar OPENAI_API_KEY o GOOGLE_API_KEY de forma segura mediante un
    prompt interactivo oculto; no escribir ni exponer las claves en comandos.
    python run_experiment.py --questions questions.json --provider openai --model gpt-4o

ANTES DE CORRER EL EXPERIMENTO REAL:
    1. Congelar RESTRICT_TO_WHITELIST=True en worldbank_tool.py y oecd_tool.py.
    2. Confirmar que query_local_reviews está implementada (ver local_reviews_tool.py).
    3. Confirmar la validación de Costa Rica en OCDE (ver README.md del proyecto).
    4. No modificar las herramientas una vez iniciado el benchmark, para
       evitar contaminación de la evaluación entre ejecuciones.
"""

import argparse
import asyncio
import json
import os
import random
import time
import sys
from datetime import datetime, timezone

from providers import call_provider, ProviderError
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from mcp_client_bridge import MCPClientBridge

MAX_TOOL_ITERATIONS = 5  # cap on tool-calling rounds per question, to avoid infinite loops
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "experiment_logs")

MAX_PROVIDER_ATTEMPTS = 3
RETRY_BASE_DELAYS_SECONDS = (2.0, 4.0)
RETRY_JITTER_FRACTION = 0.20


async def call_provider_with_retry(
    provider: str,
    messages: list[dict],
    *,
    model: str | None,
    tool_definitions: list[dict] | None,
    retry_log: list[dict],
    call_phase: str,
):
    """Retry only the current LLM call after transient infrastructure errors."""
    for attempt in range(1, MAX_PROVIDER_ATTEMPTS + 1):
        try:
            return call_provider(
                provider,
                messages,
                model=model,
                tool_definitions=tool_definitions,
            )
        except ProviderError as exc:
            exhausted = attempt >= MAX_PROVIDER_ATTEMPTS

            if not exc.retryable or exhausted:
                raise

            base_delay = RETRY_BASE_DELAYS_SECONDS[attempt - 1]
            jitter_factor = random.uniform(
                1.0 - RETRY_JITTER_FRACTION,
                1.0 + RETRY_JITTER_FRACTION,
            )
            delay = base_delay * jitter_factor

            retry_log.append({
                "call_phase": call_phase,
                "attempt_number": attempt,
                "next_attempt_number": attempt + 1,
                "provider": provider,
                "model": model,
                "error_type": exc.error_type,
                "http_status_code": exc.status_code,
                "exception_type": (
                    exc.__cause__.__class__.__name__
                    if exc.__cause__ is not None
                    else exc.__class__.__name__
                ),
                "error_message": str(exc),
                "backoff_seconds": delay,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            })

            await asyncio.sleep(delay)


def load_questions(path: str) -> list[dict]:
    """
    Expects a JSON file shaped as:
    [
      {
        "id": "local_01",
        "category": "local" | "worldbank" | "oecd" | "wb_vs_oecd_A" | "wb_vs_oecd_B" | "wb_vs_oecd_C" | "chained",
        "question_en": "...",
        "expected_tool": "get_worldbank_indicator",   # null for 'chained' questions (may use several)
        "ground_truth": { ... }   # free-form structure, defined when the set is built (Section 4.2 of the paper)
      },
      ...
    ]

    Espera un JSON con la forma:
    [
      {
        "id": "local_01",
        "category": "local" | "worldbank" | "oecd" | "wb_vs_oecd_A" | "wb_vs_oecd_B" | "wb_vs_oecd_C" | "chained",
        "question_en": "...",
        "expected_tool": "get_worldbank_indicator",   # null para preguntas 'chained' (pueden usar varias)
        "ground_truth": { ... }   # estructura libre, definida al construir el set (Sección 4.2 del paper)
      },
      ...
    ]
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


async def run_single_question(question: dict, provider: str, model: str | None, bridge: MCPClientBridge, tool_definitions: list[dict], schema_context: str) -> dict:
    """
    Runs a question until tool-calling is exhausted or a final text answer
    is produced. Every tool call travels over the real MCP bridge (standard
    protocol, server subprocess already open and shared across all questions
    of the run, for efficiency — one subprocess for the whole set of
    questions, not one per question).

    The harness applies two safeguards during tool orchestration:
      - Duplicate-call detection: if the model requests the same tool with
        the same arguments more than once within a question, the repeated
        call is blocked rather than executed again. This prevents redundant
        loops and avoids artificially inflating tool-execution success through
        repeated identical attempts.
      - Fallback synthesis: if the maximum number of tool-calling iterations
        is reached before a final textual answer is produced, the model is
        asked to synthesize an answer using only the observations already
        obtained. This condition is recorded explicitly through
        ``used_fallback_synthesis`` for subsequent analysis.

    Returns a complete record for later analysis — it deliberately does NOT
    determine here whether the answer is correct (that requires the ground
    truth and is done in scoring.py).

    Corre una pregunta hasta agotar tool-calling o llegar a una respuesta de
    texto final. Cada llamada a herramienta viaja por el bridge MCP real
    (protocolo estándar, subproceso servidor ya abierto y compartido entre
    todas las preguntas de la corrida, por eficiencia — un solo subproceso
    para las 30 preguntas, no uno por pregunta).

    El harness aplica dos salvaguardas durante la orquestación:
      - Detección de llamadas duplicadas: una solicitud repetida de la misma
        herramienta con los mismos argumentos se bloquea en lugar de
        ejecutarse nuevamente. Esto evita ciclos redundantes y que intentos
        idénticos inflen artificialmente la métrica de éxito de ejecución.
      - Síntesis de respaldo: si se alcanza el máximo de iteraciones sin una
        respuesta textual final, se solicita una síntesis basada únicamente
        en las observaciones ya obtenidas. Esta condición se registra mediante
        ``used_fallback_synthesis`` para su análisis posterior.

    Devuelve un registro completo para análisis posterior — deliberadamente
    NO calcula aquí si la respuesta es correcta (eso requiere el ground
    truth y se hace en scoring.py).
    """
    messages = [
        {
            "role": "system",
            "content": (
                "Use the following semantic instructions and source descriptions "
                "published by the MCP server when selecting tools and interpreting "
                "their results. Do not invent evidence outside the available sources.\n\n"
                + schema_context
            ),
        },
        {"role": "user", "content": question["question_en"]},
    ]
    tool_call_log = []
    provider_response_log = []
    provider_retry_log = []
    executed_signatures = set()
    start_time = time.monotonic()

    final_text = None
    used_fallback_synthesis = False

    for iteration in range(MAX_TOOL_ITERATIONS):
        try:
            response = await call_provider_with_retry(
                provider,
                messages,
                model=model,
                tool_definitions=tool_definitions,
                retry_log=provider_retry_log,
                call_phase=f"agent_loop_iteration_{iteration + 1}",
            )
        except ProviderError as exc:
            return {
                "question_id": question["id"],
                "category": question["category"],
                "provider": provider,
                "model": model,
                "status": "provider_error",
                "error": str(exc),
                "tool_calls": tool_call_log,
                "provider_responses": provider_response_log,

                "provider_retries": provider_retry_log,

                "infrastructure_retry_count": len(provider_retry_log),

                "retry_backoff_seconds": sum(

                    item["backoff_seconds"] for item in provider_retry_log

                ),
                "final_text": None,
                "used_fallback_synthesis": False,
                "latency_seconds": time.monotonic() - start_time,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            }

        provider_response_log.append(response.metadata)

        if not response.tool_calls:
            final_text = response.text
            break

        messages.append({
            "role": "assistant",
            "content": response.text or "",
            "tool_calls": [
                {
                    "name": tc.name,
                    "arguments": tc.arguments,
                    "call_id": tc.call_id,
                    "thought_signature": tc.thought_signature,
                }
                for tc in response.tool_calls
            ],
        })
        for tc in response.tool_calls:
            signature = json.dumps({"tool": tc.name, "arguments": tc.arguments}, sort_keys=True, default=str)

            if signature in executed_signatures:
                # Identical call already executed — block it and inform the
                # model, instead of hitting the tool (or the external network) again.
                blocked_result = {
                    "success": False,
                    "result": None,
                    "error": "Llamada idéntica ya ejecutada en esta pregunta. Use la observación anterior o finalice.",
                }
                tool_call_log.append({
                    "call_id": tc.call_id,
                    "tool_name": tc.name,
                    "arguments": tc.arguments,
                    "success": False,
                    "error": blocked_result["error"],
                    "result_summary": None,
                    "blocked_as_duplicate": True,
                })
                messages.append({
                    "role": "tool",
                    "name": tc.name,
                    "tool_call_id": tc.call_id,
                    "content": json.dumps(blocked_result, ensure_ascii=False, default=str),
                })
                continue

            executed_signatures.add(signature)
            result = await bridge.call_tool(tc.name, tc.arguments)
            tool_call_log.append({
                "call_id": tc.call_id,
                "tool_name": tc.name,
                "arguments": tc.arguments,
                "success": result["success"],
                "error": result["error"],
                "result_summary": result["result"] if result["success"] else None,
                "blocked_as_duplicate": False,
            })
            messages.append({
                "role": "tool",
                "name": tc.name,
                "tool_call_id": tc.call_id,
                "content": json.dumps(result, ensure_ascii=False, default=str),
            })
    else:
        # Iterations exhausted without a final answer — fallback synthesis
        # using what was observed so far, instead of giving up.
        try:
            synthesis_prompt = (
                f"Original question: {question['question_en']}\n\n"
                "You have run out of tool-calling turns. Based ONLY on the "
                "tool observations already gathered in this conversation, "
                "provide the best final answer you can. If the evidence is "
                "insufficient, say so explicitly rather than guessing."
            )
            messages.append({"role": "user", "content": synthesis_prompt})
            fallback_response = await call_provider_with_retry(
                provider,
                messages,
                model=model,
                tool_definitions=tool_definitions,
                retry_log=provider_retry_log,
                call_phase="fallback_synthesis",
            )
            provider_response_log.append(fallback_response.metadata)
            final_text = fallback_response.text
            used_fallback_synthesis = True
        except ProviderError:
            final_text = None

    return {
        "question_id": question["id"],
        "category": question["category"],
        "provider": provider,
        "model": model,
        "status": "completed" if final_text is not None else "max_iterations_reached",
        "error": None,
        "tool_calls": tool_call_log,
        "provider_responses": provider_response_log,

        "provider_retries": provider_retry_log,

        "infrastructure_retry_count": len(provider_retry_log),

        "retry_backoff_seconds": sum(

            item["backoff_seconds"] for item in provider_retry_log

        ),
        "final_text": final_text,
        "used_fallback_synthesis": used_fallback_synthesis,
        "latency_seconds": time.monotonic() - start_time,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


async def main_async():
    parser = argparse.ArgumentParser(description="Correr el benchmark de 30 preguntas contra un proveedor LLM, vía protocolo MCP real.")
    parser.add_argument("--questions", required=True, help="Ruta al JSON de preguntas (ver load_questions).")
    parser.add_argument("--provider", required=True, choices=["openai", "gemini"])
    parser.add_argument("--model", default=None, help="Nombre de modelo específico (opcional).")
    parser.add_argument("--limit", type=int, default=None, help="Correr solo las primeras N preguntas (para pruebas).")
    args = parser.parse_args()

    questions = load_questions(args.questions)
    if args.limit:
        questions = questions[: args.limit]

    os.makedirs(LOG_DIR, exist_ok=True)
    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = os.path.join(LOG_DIR, f"run_{args.provider}_{run_timestamp}.jsonl")

    print(f"Corriendo {len(questions)} preguntas con provider={args.provider} model={args.model or '(default)'} vía MCP real")
    print(f"Log: {log_path}")

    async with MCPClientBridge() as bridge:
        tools_available = await bridge.list_tools()
        schema_context = await bridge.read_resource("data://schema")
        print(f"Servidor MCP conectado. Herramientas disponibles: {[t['name'] for t in tools_available]}")

        with open(log_path, "w", encoding="utf-8") as log_file:
            for i, q in enumerate(questions, 1):
                print(f"  [{i}/{len(questions)}] {q['id']} ({q['category']})...", end=" ", flush=True)
                record = await run_single_question(
                    q,
                    args.provider,
                    args.model,
                    bridge,
                    tools_available,
                    schema_context,
                )
                log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                log_file.flush()
                print(record["status"])

    print(f"\nCompletado. Log completo en: {log_path}")
    print("Siguiente paso: scoring.py para calcular métricas contra ground_truth (por construir).")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
