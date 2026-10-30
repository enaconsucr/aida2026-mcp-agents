"""
scoring.py — Calcula las métricas de la Sección 4.3 del paper a partir de:
  (1) el log de ejecución producido por run_experiment.py (un .jsonl con un
      registro por pregunta: tool_calls, final_text, latency, etc.)
  (2) el archivo de preguntas con su ground_truth (mismo que se le pasó a
      run_experiment.py — ver questions_template.json)

La ejecución del benchmark y su evaluación se mantienen separadas para
permitir la revisión o reanotación del ground truth sin necesidad de repetir
las llamadas a los proveedores ni alterar los registros experimentales.

MÉTRICAS CALCULADAS (ver Sección 4.3 del paper):
  - tool_selection_accuracy: solo sobre categorías con herramienta única
    esperada (local, worldbank, oecd, wb_vs_oecd_A, wb_vs_oecd_B)
  - tool_execution_success: sobre todas las llamadas a herramienta registradas
  - answer_accuracy: requiere comparación con ground_truth — automatizada
    solo para valores numéricos con tolerancia; el resto queda marcado
    "needs_manual_review" (ver nota más abajo)
  - source_preference_under_ambiguity: solo sobre categoría wb_vs_oecd_C
  - chaining_success_rate: solo sobre categoría 'chained'
  - latency: agregada por categoría

LIMITACIÓN IMPORTANTE Y DELIBERADA: "answer accuracy" para preguntas de
texto libre (especialmente 'chained', que requiere juicio sobre si la
respuesta integra evidencia sin caer en causalidad injustificada — ver
Sección 4.1.1 del paper) NO se puede calcular automáticamente de forma
confiable. Este script calcula lo que es mecánicamente verificable (match
numérico con tolerancia, selección de herramienta) y deja explícitamente
marcado qué preguntas requieren revisión manual — no intenta simular un
juicio cualitativo con heurísticas frágiles que inflarían falsamente el
rigor del paper.

USO:
    python scoring.py --log experiment_logs/run_openai_20260825T....jsonl --questions questions.json --output scores.json
"""

import argparse
import json
import os
import re
from collections import defaultdict


NUMERIC_TOLERANCE_PCT = 0.02  # 2% de tolerancia relativa para comparar valores numéricos del ground truth

# Categorías donde tool-selection accuracy es una métrica válida (hay una
# única herramienta "correcta" esperada). wb_vs_oecd_C queda fuera a
# propósito; esta condición mide preferencia de fuente bajo ambigüedad.
TOOL_SELECTION_CATEGORIES = {"local", "worldbank", "oecd", "wb_vs_oecd_A", "wb_vs_oecd_B"}
AMBIGUITY_CATEGORY = "wb_vs_oecd_C"
CHAINED_CATEGORY = "chained"

EXPECTED_TOOL_BY_CATEGORY_PREFIX = {
    "local": "query_local_reviews",
    "worldbank": "get_worldbank_indicator",
    "oecd": "get_oecd_indicator",
    "wb_vs_oecd_A": "get_worldbank_indicator",
    "wb_vs_oecd_B": "get_oecd_indicator",
}


# --- v2: banderas ---
NEGATION_FIX = True          # corrige 'do not match' que pasaba como positivo (ver main: se calculan ambas versiones)
STUDY_COUNTRY = "CRI"        # país del estudio (validez contextual)
GROUNDING_PATH = "grounding_annotations.json"


def load_log(path: str) -> list[dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_questions_by_id(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        questions = json.load(f)
    return {q["id"]: q for q in questions}


def numeric_match(actual: float | None, expected: float | None, tolerance_pct: float = NUMERIC_TOLERANCE_PCT) -> bool | None:
    """
    Compara dos valores numéricos con tolerancia relativa. Devuelve None
    (no aplicable) si cualquiera de los dos es None — un ground truth
    incompleto (placeholder sin llenar, ej. "value": null en la plantilla)
    no debe contarse como fallo silencioso, debe marcarse aparte.
    """
    if actual is None or expected is None:
        return None
    if expected == 0:
        return actual == 0
    return abs(actual - expected) / abs(expected) <= tolerance_pct


def score_tool_selection(record: dict, question: dict) -> dict | None:
    """
    Para categorías con herramienta única esperada: ¿la primera (o única)
    llamada a herramienta coincide con la esperada?

    Nota de diseño: se evalúa la PRIMERA llamada a herramienta, no "en algún
    momento llamó la correcta" — si el agente prueba varias herramientas
    hasta acertar, eso es información relevante (posible indecisión) que
    debe verse en tool_calls, no enmascararse contando como éxito.
    """
    category = question["category"]
    if category not in TOOL_SELECTION_CATEGORIES:
        return None

    # category puede venir con o sin sufijo numerado del schema original;
    # usar el prefijo reconocible.
    prefix_key = next((k for k in EXPECTED_TOOL_BY_CATEGORY_PREFIX if category.startswith(k)), None)
    expected_tool = question.get("expected_tool") or (
        EXPECTED_TOOL_BY_CATEGORY_PREFIX.get(prefix_key) if prefix_key else None
    )
    if expected_tool is None:
        return {"applicable": False, "reason": "sin expected_tool definido en la pregunta"}

    tool_calls = record.get("tool_calls", [])
    if not tool_calls:
        return {"applicable": True, "correct": False, "reason": "no invocó ninguna herramienta"}

    first_call = tool_calls[0]
    correct = first_call["tool_name"] == expected_tool
    return {
        "applicable": True,
        "correct": correct,
        "expected": expected_tool,
        "actual": first_call["tool_name"],
    }


def score_tool_execution(record: dict) -> dict:
    """
    Fracción de llamadas a herramienta que ejecutaron sin error, sobre el
    total de llamadas REALES de la pregunta — excluye las bloqueadas por
    duplicado, que no son intentos de
    ejecución fallidos sino bloqueos deliberados del harness.
    """
    tool_calls = [tc for tc in record.get("tool_calls", []) if not tc.get("blocked_as_duplicate", False)]
    if not tool_calls:
        return {"applicable": False, "reason": "no se invocó ninguna herramienta (excluyendo duplicados bloqueados)"}
    successes = sum(1 for tc in tool_calls if tc["success"])
    return {
        "applicable": True,
        "total_calls": len(tool_calls),
        "successful_calls": successes,
        "success_rate": successes / len(tool_calls),
    }


def score_ambiguity_condition(record: dict, question: dict) -> dict:
    """
    Para wb_vs_oecd_C: registra qué fuente(s) consultó el agente y si los
    valores de WB/OCDE del ground truth coinciden — NO asigna "correcto/
    incorrecto" (no existe una única respuesta correcta por diseño, ver
    Sección 4.1.2 del paper) sino que deja los datos para análisis
    cualitativo en Discussion.
    """
    tool_calls = record.get("tool_calls", [])
    tools_used = [tc["tool_name"] for tc in tool_calls]
    consulted_both = "get_worldbank_indicator" in tools_used and "get_oecd_indicator" in tools_used

    gt = question.get("ground_truth", {})
    values_match = gt.get("values_match")

    return {
        "tools_consulted": tools_used,
        "consulted_both_sources": consulted_both,
        "ground_truth_values_match": values_match,
        "note": "Requiere revisión manual: ¿la respuesta final reconoce la ambigüedad si consultó una sola fuente y los valores GT difieren?",
    }


def score_chained(record: dict, question: dict) -> dict:
    """
    Para preguntas encadenadas: solo verifica mecánicamente si se invocaron
    AMBAS categorías de herramienta necesarias (local + institucional). La
    evaluación de "evidence-grounded synthesis" (¿integra sin caer en
    causalidad injustificada? ver Sección 4.1.1) queda marcada para revisión
    manual — no es automatizable de forma confiable con heurísticas simples.
    """
    tool_calls = record.get("tool_calls", [])
    tools_used = {tc["tool_name"] for tc in tool_calls}
    used_local = "query_local_reviews" in tools_used
    used_institutional = bool(tools_used & {"get_worldbank_indicator", "get_oecd_indicator"})

    return {
        "chaining_attempted": used_local and used_institutional,
        "used_local_source": used_local,
        "used_institutional_source": used_institutional,
        "needs_manual_review": True,
        "manual_review_checklist": [
            "¿Identificó correctamente el problema dominante (a)?",
            "¿Reportó correctamente la tendencia del indicador (b)?",
            "¿La relación entre (a) y (b) es defendible, sin inferencia causal no sustentada (c)?",
        ],
    }



def score_answer_auto(question: dict, record: dict) -> dict:
    """
    Evalúa automáticamente únicamente los componentes explícitamente
    solicitados por cada pregunta. No convierte todos los campos auxiliares
    del ground_truth en criterios de puntuación.
    """
    import re

    answer = (record.get("final_text") or "").strip()
    answer_lower = answer.lower()
    gt = question.get("ground_truth", {})
    qid = question.get("id", "")
    checks = []

    def norm_text(x):
        return str(x).strip().lower()

    def contains_text(value):
        if value is None:
            return False
        return norm_text(value) in answer_lower

    def contains_number(value):
        if value is None or isinstance(value, bool):
            return False

        try:
            target = float(value)
        except (TypeError, ValueError):
            return False

        nums = re.findall(r'[-+]?\d[\d,]*(?:\.\d+)?', answer)
        for raw in nums:
            # Normalización numérica:
            # - 1,253   -> 1253   (separador de miles)
            # - 53,21   -> 53.21  (coma decimal)
            # - 1,253.4 -> 1253.4
            if "." not in raw and raw.count(",") == 1:
                left, right = raw.split(",")
                if 1 <= len(right) <= 2:
                    cleaned = left + "." + right
                else:
                    cleaned = raw.replace(",", "")
            else:
                cleaned = raw.replace(",", "")

            try:
                observed = float(cleaned)
            except ValueError:
                continue

            # Coincidencia exacta práctica
            if abs(observed - target) <= 1e-9:
                return True

            # Permitir redondeo según los decimales efectivamente reportados
            if "." in cleaned:
                decimals = len(cleaned.split(".")[-1])
            else:
                decimals = 0

            tolerance = 0.5 * (10 ** (-decimals))
            if abs(observed - target) <= tolerance + 1e-12:
                return True

            # En deltas negativos, permitir magnitud positiva si la dirección
            # se expresa lingüísticamente y se evalúa en otro check.
            if target < 0 and abs(observed - abs(target)) <= tolerance + 1e-12:
                return True

        return False

    def add_check(name, expected, correct):
        checks.append({
            "name": name,
            "expected": expected,
            "correct": bool(correct),
        })

    def direction_ok(value):
        v = norm_text(value)
        if v in {"increase", "increased", "increasing", "up"}:
            return any(x in answer_lower for x in [
                "increase", "increased", "increasing",
                "rose", "rising", "grew", "growth",
                "net increase", "increased by",
                "aument", "increment"
            ])
        if v in {"decrease", "decreased", "decreasing", "down"}:
            return any(x in answer_lower for x in [
                "decrease", "decreased", "decreasing",
                "fell", "falling", "decline", "declined",
                "net decrease", "decreased by",
                "reduction", "redujo", "dismin"
            ])
        return contains_text(value)

    def relationship_ok(value):
        v = norm_text(value)
        if "opposite" in v:
            return any(x in answer_lower for x in [
                "opposite", "opposite directions",
                "directions were opposite", "direcciones opuestas"
            ])
        if "same" in v:
            return any(x in answer_lower for x in [
                "same direction", "same directions",
                "both increased", "both decreased",
                "both rose", "both fell",
                "misma dirección", "ambos aument", "ambos dismin"
            ])
        if "lower" in v:
            return any(x in answer_lower for x in ["lower", "less than", "menor"])
        if "higher" in v:
            return any(x in answer_lower for x in ["higher", "greater than", "mayor"])
        return contains_text(value)

    def boolean_ok(name, value):
        if name == "values_match_at_3_decimals":
            if NEGATION_FIX:
                neg = re.search(r"\b(do not|does not|don't|doesn't|not|no)\s+(match|equal|coincid\w*|agree)|\b(differ|differs|are different|not the same|difier\w*)\b", answer_lower)
                pos = re.search(r"\b(match|matches|equal|are the same|same at three decimals|coincid\w*|iguales)\b", answer_lower)
                return bool(neg) if not value else (bool(pos) and not neg)
            if value:
                return any(x in answer_lower for x in ["match", "equal", "same at three decimals", "coincide", "iguales"])
            return any(x in answer_lower for x in ["do not match", "not equal", "differ", "no coincide", "difier"])

        if name == "diverged_in_direction":
            if value:
                return relationship_ok("opposite")
            return any(x in answer_lower for x in [
                "same direction", "both increased", "both decreased",
                "no directional divergence", "did not diverge",
                "misma dirección", "no diverg"
            ])

        return contains_text(str(value))

    def walk(obj, allowed_keys=None, prefix=""):
        if not isinstance(obj, dict):
            return

        for k, v in obj.items():
            path_name = f"{prefix}.{k}" if prefix else k

            if isinstance(v, dict):
                walk(v, allowed_keys, path_name)
                continue

            if allowed_keys is not None and k not in allowed_keys:
                continue

            if k in {"direction", "trend_direction"}:
                add_check(path_name, v, direction_ok(v))

            elif k in {"relationship", "relationship_of_directions"}:
                add_check(path_name, v, relationship_ok(v))

            elif k in {"values_match_at_3_decimals", "diverged_in_direction"}:
                add_check(path_name, v, boolean_ok(k, v))

            elif k in {"category", "main_problem"}:
                if isinstance(v, str):
                    add_check(path_name, v, contains_text(v))

            elif k in {"top_negative_keywords", "highest_years"}:
                vals = v if isinstance(v, list) else [v]
                for item in vals:
                    if isinstance(item, dict):
                        word = (
                            item.get("word")
                            or item.get("keyword")
                            or item.get("term")
                            or item.get("year")
                        )
                        count = item.get("count")
                        if word is not None:
                            add_check(f"{path_name}.item", word, contains_text(word))
                        if count is not None:
                            add_check(f"{path_name}.count", count, contains_number(count))
                    else:
                        add_check(f"{path_name}.item", item,
                                  contains_number(item) if isinstance(item, (int, float))
                                  else contains_text(item))

            elif isinstance(v, bool):
                add_check(path_name, v, boolean_ok(k, v))

            elif isinstance(v, (int, float)):
                add_check(path_name, v, contains_number(v))

            elif isinstance(v, str):
                add_check(path_name, v, contains_text(v))

    answer_rubric_by_id = {
        "wb_01": {"year", "value"},
        "wb_02": {"value"},
        "wb_03": {"direction"},
        "wb_04": {"relationship"},
        "wb_05": {"delta", "delta_percentage_points", "direction"},

        "oecd_01": {"year", "value"},
        "oecd_02": {"value"},
        "oecd_03": {"direction"},
        "oecd_04": {"highest_years"},
        "oecd_05": {"delta", "delta_percentage_points", "direction"},

        "local_01": {"negative_review_count"},
        "local_02": {"category"},
        "local_03": {"percentage_negative"},
        "local_04": {"direction"},
        "local_05": {"top_negative_keywords", "category", "count"},

        "wb_oecd_01": {"value", "worldbank_value", "oecd_value"},
        "wb_vs_oecd_neutral_01": {
            "worldbank_value",
            "oecd_value",
            "values_match_at_3_decimals",
        },
        "wb_oecd_02": {"value", "worldbank_value", "oecd_value"},
        "wb_oecd_03": {"value", "worldbank_value", "oecd_value"},
        "wb_oecd_04": {"value", "worldbank_value", "oecd_value"},
        "wb_oecd_05": {
            "worldbank_value",
            "oecd_value",
            "native_difference_percentage_points",
        },

        "chained_01": {"category", "direction"},
        "chained_02": {"category", "direction"},
        "chained_03": {"category", "direction"},
        "chained_04": {
            "worldbank_value",
            "oecd_value",
            "worldbank_rounded_3",
            "oecd_rounded_3",
            "values_match_at_3_decimals",
            "native_difference_percentage_points",
        },
        "chained_05": {
            "top_negative_keywords",
            "top_keyword_percentage_of_negative_reviews",
            "direction",
        },
        "chained_06": {
            "category",
            "relationship_of_directions",
        },
        "chained_07": {
            "category",
            "pct_of_negative_reviews",
            "direction",
            "relationship_of_directions",
        },
        "chained_08": {
            "category",
            "direction",
            "relationship_of_directions",
        },
        "chained_09": {
            "top_negative_keywords",
            "top_keyword_percentage_of_negative_reviews",
            "direction",
            "relationship_of_directions",
        },
        "chained_10": {
            "category",
            "diverged_in_direction",
        },
    }

    allowed = answer_rubric_by_id.get(qid)

    if allowed is None:
        add_check("rubric_missing", qid, False)
    else:
        walk(gt, allowed_keys=allowed)

    evaluated = len(checks)
    correct = sum(1 for x in checks if x["correct"])

    return {
        "answer_accuracy_auto": (correct / evaluated) if evaluated else None,
        "answer_checks_correct": correct,
        "answer_checks_total": evaluated,
        "answer_checks": checks,
        "needs_manual_answer_review": True,
    }

def score_run(log_path: str, questions_path: str) -> dict:
    records = load_log(log_path)
    questions_by_id = load_questions_by_id(questions_path)

    # Manual grounding annotations are maintained as a separate frozen
    # evaluation artifact and are not part of factual answer accuracy.
    grounding_path = GROUNDING_PATH
    grounding_by_key = {}

    try:
        with open(grounding_path, "r", encoding="utf-8") as f:
            grounding_data = json.load(f)

        grounding_by_key = {
            (row["question_id"], row["model"]): row
            for row in grounding_data.get("annotations", [])
        }
    except FileNotFoundError:
        print(f"AVISO: no se encontró {grounding_path}; grounding manual = vacío (no se promedia).")
        grounding_by_key = {}

    run_provider = None
    run_model = None
    grounding_model_label = None

    if records:
        run_provider = records[0].get("provider")
        run_model = records[0].get("model")
        if run_provider:
            grounding_model_label = run_provider.upper()

    per_question = []
    tool_selection_results = []
    tool_execution_results = []
    latencies_by_category = defaultdict(list)

    for record in records:
        qid = record["question_id"]
        question = questions_by_id.get(qid)
        if question is None:
            per_question.append({
                "question_id": qid,
                "error": f"Pregunta '{qid}' del log no encontrada en el archivo de preguntas — "
                         f"¿se usó un questions.json distinto al de esta corrida?",
            })
            continue

        category = question["category"]
        latencies_by_category[category].append(record.get("latency_seconds"))

        entry = {
            "question_id": qid,
            "category": category,
            "status": record.get("status"),
        }

        ts_result = score_tool_selection(record, question)
        if ts_result is not None:
            entry["tool_selection"] = ts_result
            if ts_result.get("applicable"):
                tool_selection_results.append(ts_result["correct"])

        te_result = score_tool_execution(record)
        entry["tool_execution"] = te_result
        if te_result.get("applicable"):
            tool_execution_results.append(te_result["success_rate"])

        if category == AMBIGUITY_CATEGORY:
            entry["ambiguity_analysis"] = score_ambiguity_condition(record, question)

        if category == CHAINED_CATEGORY:
            entry["chained_analysis"] = score_chained(record, question)

        aa_result = score_answer_auto(question, record)
        entry.update(aa_result)

        grounding = grounding_by_key.get(
            (qid, grounding_model_label)
        )
        if grounding is not None:
            entry["grounding"] = {
                "entity_grounding": grounding["entity_grounding"],
                "evidence_grounding": grounding["evidence_grounding"],
                "causal_grounding": grounding["causal_grounding"],
                "overall_grounding": grounding["overall_grounding"],
                "rationale": grounding.get("rationale"),
            }

        per_question.append(entry)

    answer_accuracy_values = [
        r.get("answer_accuracy_auto")
        for r in per_question
        if r.get("answer_accuracy_auto") is not None
    ]

    answer_checks_correct_total = sum(
        r.get("answer_checks_correct", 0) for r in per_question
    )
    answer_checks_total = sum(
        r.get("answer_checks_total", 0) for r in per_question
    )

    exact_answers = sum(
        1
        for r in per_question
        if r.get("answer_checks_total", 0) > 0
        and r.get("answer_checks_correct", 0) == r.get("answer_checks_total", 0)
    )

    grounding_results = [
        r["grounding"]
        for r in per_question
        if "grounding" in r
    ]

    def grounding_mean(key: str):
        if not grounding_results:
            return None
        return (
            sum(r[key] for r in grounding_results)
            / len(grounding_results)
        )

    summary = {
        "total_questions_scored": len(per_question),
        "tool_selection_accuracy": (
            sum(tool_selection_results) / len(tool_selection_results)
            if tool_selection_results else None
        ),
        "tool_selection_n": len(tool_selection_results),
        "avg_tool_execution_success_rate": (
            sum(tool_execution_results) / len(tool_execution_results)
            if tool_execution_results else None
        ),
        "answer_accuracy_macro": (
            sum(answer_accuracy_values) / len(answer_accuracy_values)
            if answer_accuracy_values else None
        ),
        "answer_accuracy_n": len(answer_accuracy_values),
        "answer_accuracy_micro_components": (
            answer_checks_correct_total / answer_checks_total
            if answer_checks_total else None
        ),
        "answer_checks_correct_total": answer_checks_correct_total,
        "answer_checks_total": answer_checks_total,
        "exact_answer_rate": (
            exact_answers / len(per_question)
            if per_question else None
        ),
        "exact_answers": exact_answers,
        "grounding_n": len(grounding_results),
        "entity_grounding": grounding_mean("entity_grounding"),
        "evidence_grounding": grounding_mean("evidence_grounding"),
        "causal_grounding": grounding_mean("causal_grounding"),
        "overall_grounding": grounding_mean("overall_grounding"),
        "grounding_annotation_model": grounding_model_label,
        "run_provider": run_provider,
        "run_model": run_model,
        "latency_by_category_seconds": {
            cat: {
                "mean": sum(v for v in vals if v is not None) / len(vals) if vals else None,
                "n": len(vals),
            }
            for cat, vals in latencies_by_category.items()
        },
        "note": (
            "La exactitud factual de las respuestas se calcula automáticamente "
            "mediante checks explícitos contra el ground_truth. La revisión manual "
            "El grounding se incorpora desde anotaciones manuales externas y "
            "congeladas, separadas de la exactitud factual automática. Aspectos "
            "cualitativos adicionales de la síntesis pueden requerir revisión manual."
        ),
    }

    return {"summary": summary, "per_question": per_question}


def contextual_validity(log_path: str, questions_path: str) -> dict:
    """Métrica SEPARADA de exactitud factual: ¿los argumentos country_iso3 de las herramientas son el país del estudio?
    Una respuesta puede ser factualmente exacta respecto a los datos devueltos y aun así contextualmente inválida."""
    recs, qs = load_log(log_path), load_questions_by_id(questions_path)
    rows = []
    for r in recs:
        calls = []
        for t in r.get("tool_calls", []):
            a = t.get("arguments") or {}
            if "country_iso3" in a:
                calls.append({"tool": t["tool_name"], "country": a["country_iso3"], "success": bool(t.get("success"))})
        other = [c for c in calls if c["country"] != STUDY_COUNTRY]            # cualquier intento fuera del país del estudio
        other_ok = [c for c in other if c["success"]]                           # datos de otro país entraron a la respuesta
        study_ok = [c for c in calls if c["country"] == STUDY_COUNTRY and c["success"]]
        study_fail = [c for c in calls if c["country"] == STUDY_COUNTRY and not c["success"]]
        q = qs.get(r["question_id"], {})
        rows.append({
            "question_id": r["question_id"], "category": r.get("category"),
            "question_names_costa_rica": "costa rica" in q.get("question_en", "").lower(),
            "n_country_calls": len(calls),
            "countries_used": sorted({c["country"] for c in calls}),
            "non_study_country_attempts": len(other),
            "non_study_country_data_entered": len(other_ok),
            "study_country_calls_failed": len(study_fail),   # fallo de infraestructura (timeout), NO es país equivocado
            # 1 = ningún intento fuera de Costa Rica; 0 = se pidió otro país (aunque la llamada fallara)
            "country_valid_auto": int(len(other) == 0),
            "needs_manual_review": bool(other) and bool(study_ok),  # p. ej. consulta adicional a otro país junto con CRI
        })
    def agg(sub):
        sub = [x for x in sub if x["n_country_calls"] > 0]
        return {"n": len(sub), "country_valid": sum(x["country_valid_auto"] for x in sub)}
    ch = [x for x in rows if x["category"] == "chained"]
    return {
        "all_with_country_args": agg(rows),
        "chained": agg(ch),
        "chained_question_names_country": agg([x for x in ch if x["question_names_costa_rica"]]),
        "chained_question_omits_country": agg([x for x in ch if not x["question_names_costa_rica"]]),
        "per_question": rows,
        "note": "country_valid_auto = 0 si hubo CUALQUIER intento con un país distinto de Costa Rica (aunque fallara); las fallas de la llamada a CRI se reportan aparte (study_country_calls_failed). Los casos needs_manual_review (consulta adicional a otro país junto con CRI) requieren juicio manual.",
    }


def main():
    parser = argparse.ArgumentParser(description="Calcular métricas del benchmark contra ground truth.")
    parser.add_argument("--log", required=True, help="Ruta al .jsonl producido por run_experiment.py")
    parser.add_argument("--questions", required=True, help="Ruta al JSON de preguntas con ground_truth")
    parser.add_argument("--grounding", default="grounding_annotations.json", help="Ruta a las anotaciones manuales (original o auditada)")
    parser.add_argument("--output", default=None, help="Ruta donde guardar el resultado (default: <log>_scored.json)")
    args = parser.parse_args()

    global NEGATION_FIX, GROUNDING_PATH
    GROUNDING_PATH = args.grounding
    NEGATION_FIX = True
    corrected = score_run(args.log, args.questions)
    NEGATION_FIX = False
    original_logic = score_run(args.log, args.questions)
    result = {
        "summary": corrected["summary"], "per_question": corrected["per_question"],
        "original_logic_summary": original_logic["summary"], "original_logic_per_question": original_logic["per_question"],
        "contextual_validity": contextual_validity(args.log, args.questions),
        "meta": {"scoring_version": "v2", "negation_fix_in_main_fields": True, "grounding_file": args.grounding,
                 "study_country": STUDY_COUNTRY},
    }

    output_path = args.output or (os.path.splitext(args.log)[0] + "_scored_v2.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(f"Scoring completado. Resumen:")
    print(json.dumps(result["summary"], indent=2, ensure_ascii=False))
    cv = result["contextual_validity"]
    print("\nValidez contextual (country_iso3):", json.dumps({k: cv[k] for k in ("all_with_country_args","chained","chained_question_names_country","chained_question_omits_country")}, ensure_ascii=False))
    print(f"\nDetalle completo por pregunta guardado en: {output_path}")
    print("\nRECORDATORIO: revisar manualmente las preguntas marcadas 'needs_manual_answer_review' "
          "antes de reportar Results en el paper — este script no reemplaza esa revisión.")


if __name__ == "__main__":
    main()
