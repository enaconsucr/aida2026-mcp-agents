"""Provider adapters for LLM-based tool-use experiments.

English
-------
This module provides a provider-neutral interface for invoking language models
with MCP tool definitions. It normalizes model responses into ``ToolCall`` and
``ProviderResponse`` objects so that the experimental harness can process tool
requests independently of provider-specific message formats.

The adapters currently support OpenAI and Google Gemini. Provider-specific
message conversion, tool schemas, response metadata, and tool-call identifiers
are handled internally while preserving a common contract for the experimental
pipeline.

OpenAI uses the Chat Completions interface. Gemini uses the ``google-genai``
SDK with automatic function calling disabled so that tool execution remains
under explicit control of the experimental harness.

Español
-------
Este módulo proporciona una interfaz neutral respecto del proveedor para
invocar modelos de lenguaje con definiciones de herramientas MCP. Las
respuestas se normalizan mediante objetos ``ToolCall`` y ``ProviderResponse``
para que el harness experimental pueda procesar las solicitudes de
herramientas independientemente del formato específico de cada proveedor.

Los adaptadores actualmente admiten OpenAI y Google Gemini. La conversión de
mensajes, los esquemas de herramientas, los metadatos de respuesta y los
identificadores de llamadas se gestionan internamente, preservando un contrato
común para el flujo experimental.

OpenAI utiliza la interfaz Chat Completions. Gemini utiliza el SDK
``google-genai`` con la llamada automática de funciones deshabilitada, de modo
que la ejecución de herramientas permanezca bajo control explícito del harness
experimental.
"""

import os
import json
from dataclasses import dataclass, field
from typing import Any

from tool_schemas import to_openai_format


@dataclass
class ToolCall:
    name: str
    arguments: dict
    call_id: str  # provider-assigned identifier used to associate the tool result with this call
    thought_signature: object | None = None


@dataclass
class ProviderResponse:
    text: str | None            # final text answer, if the model made no further tool requests
    tool_calls: list[ToolCall] = field(default_factory=list)  # empty if no tool was requested
    raw_response: Any = None    # raw SDK object, kept for debugging/archival
    metadata: dict = field(default_factory=dict)  # normalized metadata for reproducibility


class ProviderError(RuntimeError):
    """Controlled provider API failure with retry classification metadata.

    Excepción controlada de la API del proveedor con metadatos estructurados
    para distinguir fallos transitorios de infraestructura de errores no
    reintentables.
    """

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
        error_type: str = "provider",
    ):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable
        self.error_type = error_type


_RETRYABLE_HTTP_STATUS = {429, 500, 502, 503, 504}


def _classify_provider_exception(provider: str, exc: Exception) -> ProviderError:
    """Normalize SDK-specific exceptions into a structured ProviderError."""
    status_code = getattr(exc, "status_code", None)

    if status_code is None:
        response = getattr(exc, "response", None)
        status_code = getattr(response, "status_code", None)

    if status_code is None:
        code = getattr(exc, "code", None)
        if isinstance(code, int):
            status_code = code

    try:
        status_code = int(status_code) if status_code is not None else None
    except (TypeError, ValueError):
        status_code = None

    class_name = exc.__class__.__name__.lower()
    message_lower = str(exc).lower()

    transport_failure = any(
        token in class_name or token in message_lower
        for token in (
            "timeout",
            "connectionerror",
            "connection error",
            "connection reset",
            "connection aborted",
            "connection refused",
            "temporarily unavailable",
            "service unavailable",
        )
    )

    retryable = (
        status_code in _RETRYABLE_HTTP_STATUS
        or transport_failure
    )

    error_type = "infrastructure" if retryable else "provider"

    return ProviderError(
        f"Fallo llamando a {provider}: {exc}",
        status_code=status_code,
        retryable=retryable,
        error_type=error_type,
    )


# ---------------------------------------------------------------------------
# OpenAI
# ---------------------------------------------------------------------------

def _to_openai_messages(messages: list[dict]) -> list[dict]:
    """
    Convert the provider-neutral conversation history used by the harness
    into the OpenAI Chat Completions message format.

    Convierte el historial de conversación neutral del harness al formato
    de mensajes requerido por OpenAI Chat Completions.
    """
    converted: list[dict] = []

    for message in messages:
        role = message.get("role")

        if role in {"system", "user"}:
            converted.append({
                "role": role,
                "content": message.get("content", ""),
            })
            continue

        if role == "assistant":
            native_message = {
                "role": "assistant",
                "content": message.get("content") or "",
            }

            tool_calls = message.get("tool_calls") or []
            if tool_calls:
                native_message["tool_calls"] = [
                    {
                        "id": tc["call_id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(
                                tc.get("arguments", {}),
                                ensure_ascii=False,
                                default=str,
                            ),
                        },
                    }
                    for tc in tool_calls
                ]

            converted.append(native_message)
            continue

        if role == "tool":
            converted.append({
                "role": "tool",
                "tool_call_id": message["tool_call_id"],
                "content": message.get("content", ""),
            })
            continue

        raise ProviderError(f"Unsupported neutral message role for OpenAI: {role!r}")

    return converted


def call_openai(
    messages: list[dict],
    model: str = "gpt-4o",
    tool_definitions: list[dict] | None = None,
) -> ProviderResponse:
    """
    Invoke a model through the OpenAI Chat Completions interface using a
    provider-neutral message list (role/content, and role='tool' with
    tool_call_id for already-executed tool results).

    Invoca un modelo mediante la interfaz Chat Completions de OpenAI a partir
    de una lista de mensajes en formato neutral (role/content, y role='tool'
    con tool_call_id para resultados de herramientas ya ejecutadas).
    """
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ProviderError(
            "Falta el paquete 'openai'. Instalar con: pip install openai"
        ) from exc

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ProviderError("Variable de entorno OPENAI_API_KEY no configurada.")

    client = OpenAI(api_key=api_key)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=_to_openai_messages(messages),
            tools=to_openai_format(tool_definitions or []),
            temperature=0.0,
        )
    except Exception as exc:  # SDK-specific errors are normalized for the experimental harness
        raise _classify_provider_exception("OpenAI", exc) from exc

    choice = resp.choices[0]
    message = choice.message

    tool_calls = []
    if message.tool_calls:
        for tc in message.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError as exc:
                raise ProviderError(
                    f"OpenAI devolvió argumentos de herramienta no parseables como JSON: "
                    f"{tc.function.arguments}"
                ) from exc
            tool_calls.append(ToolCall(name=tc.function.name, arguments=args, call_id=tc.id))

    usage = getattr(resp, "usage", None)
    usage_data = (
        usage.model_dump()
        if usage is not None and hasattr(usage, "model_dump")
        else None
    )

    return ProviderResponse(
        text=message.content,
        tool_calls=tool_calls,
        raw_response=resp.model_dump() if hasattr(resp, "model_dump") else str(resp),
        metadata={
            "requested_model": model,
            "effective_model": getattr(resp, "model", None),
            "temperature": 0.0,
            "response_id": getattr(resp, "id", None),
            "finish_reason": getattr(choice, "finish_reason", None),
            "usage": usage_data,
        },
    )


# ---------------------------------------------------------------------------
# Gemini
# ---------------------------------------------------------------------------

def _to_gemini_contents(messages: list[dict], types):
    """
    Convert the provider-neutral conversation history into Gemini contents
    while preserving function-call identifiers and grouping consecutive tool
    responses into a single user turn.

    Convierte el historial neutral al formato de Gemini, preservando los
    identificadores de llamadas a funciones y agrupando respuestas consecutivas
    de herramientas en un único turno de usuario.

    Returns
    -------
    tuple
        (system_instruction, contents)
    """
    system_parts = []
    contents = []
    pending_function_responses = []

    def flush_function_responses():
        nonlocal pending_function_responses
        if pending_function_responses:
            contents.append(
                types.Content(
                    role="user",
                    parts=pending_function_responses,
                )
            )
            pending_function_responses = []

    for message in messages:
        role = message.get("role")

        if role == "system":
            flush_function_responses()
            content = message.get("content", "")
            if content:
                system_parts.append(content)
            continue

        if role == "user":
            flush_function_responses()
            contents.append(
                types.Content(
                    role="user",
                    parts=[
                        types.Part.from_text(
                            text=message.get("content", "")
                        )
                    ],
                )
            )
            continue

        if role == "assistant":
            flush_function_responses()

            parts = []

            content = message.get("content") or ""
            if content:
                parts.append(types.Part.from_text(text=content))

            for tc in message.get("tool_calls") or []:
                part_kwargs = {
                    "function_call": types.FunctionCall(
                        id=tc["call_id"],
                        name=tc["name"],
                        args=tc.get("arguments", {}),
                    )
                }

                thought_signature = tc.get("thought_signature")
                if thought_signature:
                    part_kwargs["thought_signature"] = thought_signature

                parts.append(types.Part(**part_kwargs))

            if parts:
                contents.append(
                    types.Content(
                        role="model",
                        parts=parts,
                    )
                )
            continue

        if role == "tool":
            raw_content = message.get("content", "")

            try:
                parsed_content = json.loads(raw_content)
            except (json.JSONDecodeError, TypeError):
                parsed_content = raw_content

            if isinstance(parsed_content, dict):
                response_payload = {"output": parsed_content}
            else:
                response_payload = {"output": parsed_content}

            pending_function_responses.append(
                types.Part(
                    function_response=types.FunctionResponse(
                        id=message["tool_call_id"],
                        name=message["name"],
                        response=response_payload,
                    )
                )
            )
            continue

        raise ProviderError(
            f"Unsupported neutral message role for Gemini: {role!r}"
        )

    flush_function_responses()

    system_instruction = "\n\n".join(system_parts) if system_parts else None

    return system_instruction, contents


def call_gemini(
    messages: list[dict],
    model: str = "gemini-1.5-pro",
    tool_definitions: list[dict] | None = None,
) -> ProviderResponse:
    """
    Invoke a model through the Google Gemini API (``google-genai`` SDK) using
    the same provider-neutral message list accepted by ``call_openai``
    (roles: system, user, assistant, tool). This function internally
    translates that neutral history into Gemini's content format via
    ``_to_gemini_contents``.

    Validation status: role mapping and function-response handling have been
    validated end-to-end against the live google-genai API, covering a single
    tool call, parallel tool calls within one turn, function-response
    continuation, execution through the real MCP bridge, and error
    propagation and recovery.

    Invoca un modelo mediante la API de Google Gemini (SDK ``google-genai``)
    utilizando la misma lista de mensajes en formato neutral que acepta
    ``call_openai`` (roles: system, user, assistant, tool). Esta función
    traduce internamente ese historial neutral al formato de contenidos de
    Gemini mediante ``_to_gemini_contents``.

    Estado de validación: el mapeo de roles y el manejo de function-response
    fueron validados de extremo a extremo contra la API real de google-genai,
    cubriendo una llamada simple a herramienta, llamadas paralelas dentro de
    un mismo turno, continuación mediante FunctionResponse, ejecución a
    través del bridge MCP real, y propagación y recuperación ante errores.
    """
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise ProviderError(
            "Missing package 'google-genai'. Install with: "
            "pip install google-genai"
        ) from exc

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ProviderError("Variable de entorno GOOGLE_API_KEY no configurada.")

    client = genai.Client(api_key=api_key)

    function_declarations = [
        types.FunctionDeclaration(
            name=t["name"],
            description=t.get("description", ""),
            parameters_json_schema=t.get(
                "input_schema",
                t.get("parameters", {"type": "object", "properties": {}}),
            ),
        )
        for t in (tool_definitions or [])
    ]

    gemini_tools = (
        [types.Tool(function_declarations=function_declarations)]
        if function_declarations
        else []
    )

    system_instruction, contents = _to_gemini_contents(messages, types)

    try:
        resp = client.models.generate_content(
            model=model,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=gemini_tools or None,
                temperature=0.0,
                automatic_function_calling=types.AutomaticFunctionCallingConfig(
                    disable=True
                ),
            ),
        )
    except Exception as exc:
        raise _classify_provider_exception("Gemini", exc) from exc

    tool_calls = []
    text_parts = []
    for part in resp.candidates[0].content.parts:
        if hasattr(part, "function_call") and part.function_call and part.function_call.name:
            tool_calls.append(
                ToolCall(
                    name=part.function_call.name,
                    arguments=dict(part.function_call.args),
                    call_id=(
                        part.function_call.id
                        or f"{part.function_call.name}_{len(tool_calls) + 1}"
                    ),
                    thought_signature=getattr(part, "thought_signature", None),
                )
            )
        elif hasattr(part, "text") and part.text:
            text_parts.append(part.text)

    usage = getattr(resp, "usage_metadata", None)
    usage_data = (
        usage.model_dump(mode="json")
        if usage is not None and hasattr(usage, "model_dump")
        else None
    )

    candidate = resp.candidates[0] if getattr(resp, "candidates", None) else None
    finish_reason = getattr(candidate, "finish_reason", None)
    if hasattr(finish_reason, "value"):
        finish_reason = finish_reason.value
    elif finish_reason is not None:
        finish_reason = str(finish_reason)

    return ProviderResponse(
        text="".join(text_parts) if text_parts else None,
        tool_calls=tool_calls,
        raw_response=str(resp),
        metadata={
            "requested_model": model,
            "effective_model": getattr(resp, "model_version", None),
            "temperature": 0.0,
            "response_id": getattr(resp, "response_id", None),
            "finish_reason": finish_reason,
            "usage": usage_data,
        },
    )


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

PROVIDERS = {
    "openai": call_openai,
    "gemini": call_gemini,
}


def call_provider(
    provider_name: str,
    messages: list[dict],
    model: str | None = None,
    tool_definitions: list[dict] | None = None,
) -> ProviderResponse:
    if provider_name not in PROVIDERS:
        raise ProviderError(f"Proveedor desconocido: '{provider_name}'. Opciones: {list(PROVIDERS.keys())}")
    fn = PROVIDERS[provider_name]
    kwargs = {"tool_definitions": tool_definitions}
    if model:
        kwargs["model"] = model
    return fn(messages, **kwargs)
