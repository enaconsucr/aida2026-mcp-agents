"""Local review analysis tool for the EDUS Google Play corpus.

This module provides the local textual evidence component of the experimental
agent. It queries a frozen corpus of 60,278 Spanish-language Google Play
reviews of EDUS and produces complementary descriptive and semantic outputs
for a user-specified date range.

Methodological design
---------------------
Sentiment is operationalized deterministically from the original star rating:
ratings 1-2 are classified as negative (NEG), rating 3 as neutral (NEU), and
ratings 4-5 as positive (POS). This mapping is a rule-based analytical
operationalization and must not be interpreted as the output or performance
of a trained sentiment classifier.

Negative-review analysis has two distinct components. First, token frequencies
provide a descriptive lexical summary through ``top_negative_keywords``.
Second, substantive problem categories are obtained from frozen offline
BERTopic assignments linked by ``review_id`` to a separately frozen,
human-validated topic taxonomy. BERTopic is therefore not fitted or inferred
at query time.

Cryptographic SHA-256 checks bind runtime analysis to the validated topic
assignment and taxonomy artifacts. Reviews assigned to BERTopic noise
(``topic_id == -1``), reviews without a frozen topic assignment, and topics
explicitly excluded by the taxonomy are reported separately rather than
imputed into substantive categories. ``main_problem`` denotes the most
frequent substantive problem category among the eligible negative reviews.

The original Spanish review text is preserved for NLP analysis. Translation
is deliberately avoided because it would alter the linguistic evidence on
which lexical and topic-based analyses depend.


Herramienta de análisis de reseñas locales del corpus de Google Play de EDUS.

Este módulo proporciona el componente de evidencia textual local del agente
experimental. Consulta un corpus congelado de 60.278 reseñas en español de
EDUS publicadas en Google Play y produce resultados descriptivos y semánticos
complementarios para un intervalo de fechas especificado por el usuario.

Diseño metodológico
-------------------
El sentimiento se operacionaliza de manera determinista a partir de la
calificación original por estrellas: las calificaciones 1-2 se clasifican como
negativas (NEG), 3 como neutral (NEU) y 4-5 como positivas (POS). Este mapeo
es una operacionalización analítica basada en reglas y no debe interpretarse
como la salida ni como el desempeño de un clasificador de sentimiento
entrenado.

El análisis de las reseñas negativas tiene dos componentes diferenciados. En
primer lugar, las frecuencias de términos proporcionan un resumen léxico
descriptivo mediante ``top_negative_keywords``. En segundo lugar, las
categorías sustantivas de problemas se obtienen de asignaciones BERTopic
congeladas fuera de línea, vinculadas mediante ``review_id`` con una taxonomía
de tópicos también congelada y validada por criterio humano. Por tanto,
BERTopic no se entrena ni realiza inferencia durante cada consulta.

Las verificaciones criptográficas SHA-256 vinculan el análisis ejecutado con
las versiones validadas de los artefactos de asignación de tópicos y
taxonomía. Las reseñas clasificadas como ruido por BERTopic
(``topic_id == -1``), las reseñas sin asignación congelada y los tópicos
excluidos explícitamente por la taxonomía se contabilizan por separado y no
se imputan a categorías sustantivas. ``main_problem`` representa la categoría
sustantiva de problema más frecuente entre las reseñas negativas elegibles.

El texto original en español se conserva para el análisis NLP. Se evita
deliberadamente su traducción porque modificaría la evidencia lingüística
sobre la que se sustentan los análisis léxico y temático.
"""

import os
import re
import sqlite3
import csv
import hashlib
import sys
from collections import Counter
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DEFAULT_REVIEWS_DB_PATH = os.path.join(
    BASE_DIR,
    "data",
    "resenas_edus_google_play.csv",
)

REVIEWS_DB_PATH = os.environ.get(
    "REVIEWS_DB_PATH",
    DEFAULT_REVIEWS_DB_PATH,
)

TOPIC_ASSIGNMENTS_PATH = os.environ.get(
    "TOPIC_ASSIGNMENTS_PATH",
    os.path.join(
        BASE_DIR,
        "models",
        "bertopic_negative_reviews_v1",
        "data",
        "negative_reviews_with_topics.csv",
    ),
)

TOPIC_TAXONOMY_PATH = os.environ.get(
    "TOPIC_TAXONOMY_PATH",
    os.path.join(
        BASE_DIR,
        "topic_audit_38_coded_v1.csv",
    ),
)


EXPECTED_TOPIC_ASSIGNMENTS_SHA256 = (
    "3be6c465aa0ebc8203785217686fe34c3650283032bce73a335bfc33c093077c"  # public release: personal data redacted in 8 review texts and Google Play review IDs replaced by pseudonyms (EDUS-000001...); topic assignments unchanged. Hash used in the experiment: 98b2b5b96665b418cd06b17711a576d39ef64f218995956d6e6e20379a4c9c5d
)

EXPECTED_TOPIC_TAXONOMY_SHA256 = (
    "09437966f1b7f57b25b63131dd41a82c70ee07b5668524f78979dc02295e5e22"
)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)

    return h.hexdigest()


def _validate_frozen_artifact(path: str, expected_sha256: str, label: str) -> None:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Frozen artifact not found ({label}): {path}"
        )

    actual_sha256 = _sha256_file(path)

    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"The frozen {label} artifact does not match the validated version. "
            f"Expected SHA-256={expected_sha256}; obtained={actual_sha256}; "
            f"path={path}"
        )

DATE_COLUMN_CANDIDATES = ["date", "at", "review_date", "fecha"]
SENTIMENT_COLUMN_CANDIDATES = ["y_pred", "sentiment_label", "sentiment", "prediccion"]
RATING_COLUMN_CANDIDATES = ["calificacion", "rating", "score", "stars", "puntuacion"]
CONTENT_COLUMN_CANDIDATES = ["content", "text", "texto", "review_text", "comentario"]
ID_COLUMN_CANDIDATES = ["reviewId", "review_id", "id"]


NEGATIVE_LABELS = {"NEG", "Negativo", "negative", "0"}

_TOPIC_ASSIGNMENTS_CACHE = None
_TOPIC_TAXONOMY_CACHE = None


def _load_topic_assignments() -> dict[str, int]:
    """Load and validate the frozen BERTopic review-to-topic assignments.
    
    The artifact maps each negative review, identified by ``review_id``, to the
    ``topic_id`` assigned during the offline BERTopic analysis. Its SHA-256 digest
    is verified before parsing so that runtime queries cannot silently use a
    different version of the topic assignments.
    
    The loader validates the required schema, rejects empty or duplicated
    ``review_id`` values, and requires every ``topic_id`` to be integer-compatible.
    The validated mapping is cached in memory after the first successful load.
    
    No BERTopic model is fitted or used for inference in this function.
    
    
    Carga y valida las asignaciones congeladas de reseñas a tópicos de BERTopic.
    
    El artefacto vincula cada reseña negativa, identificada mediante ``review_id``,
    con el ``topic_id`` asignado durante el análisis BERTopic realizado fuera de
    línea. Su huella SHA-256 se verifica antes de procesarlo para impedir que las
    consultas en tiempo de ejecución utilicen silenciosamente una versión distinta
    de las asignaciones de tópicos.
    
    La función valida el esquema requerido, rechaza valores ``review_id`` vacíos o
    duplicados y exige que cada ``topic_id`` sea compatible con un número entero.
    El mapeo validado se conserva en memoria después de la primera carga exitosa.
    
    Esta función no entrena un modelo BERTopic ni realiza inferencia con él.
    """
    global _TOPIC_ASSIGNMENTS_CACHE

    if _TOPIC_ASSIGNMENTS_CACHE is not None:
        return _TOPIC_ASSIGNMENTS_CACHE

    _validate_frozen_artifact(
        TOPIC_ASSIGNMENTS_PATH,
        EXPECTED_TOPIC_ASSIGNMENTS_SHA256,
        "BERTopic assignments",
    )

    assignments = {}

    with open(TOPIC_ASSIGNMENTS_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required = {"review_id", "topic_id"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "The BERTopic artifact is missing required columns: "
                + ", ".join(sorted(missing))
            )

        for row in reader:
            review_id = str(row["review_id"]).strip()

            if not review_id:
                raise ValueError("Empty review_id in BERTopic artifact.")

            if review_id in assignments:
                raise ValueError(
                    f"Duplicate review_id in BERTopic artifact: {review_id}"
                )

            try:
                topic_id = int(row["topic_id"])
            except (TypeError, ValueError):
                raise ValueError(
                    f"Invalid topic_id for review_id={review_id}: "
                    f"{row['topic_id']!r}"
                )

            assignments[review_id] = topic_id

    _TOPIC_ASSIGNMENTS_CACHE = assignments
    return assignments


def _load_topic_taxonomy() -> dict[int, dict]:
    """Load and validate the frozen human-reviewed topic taxonomy.
    
    The taxonomy maps each BERTopic ``topic_id`` to a conceptual
    ``problem_category`` and an ``include_as_problem`` decision. This separation
    preserves the distinction between unsupervised topic discovery and the
    subsequent human conceptual interpretation used to construct substantive
    problem categories.
    
    The taxonomy artifact is SHA-256 verified before parsing and cached after a
    successful load. Runtime analysis therefore consumes the validated taxonomy
    rather than reconstructing or modifying it dynamically.
    
    
    Carga y valida la taxonomía congelada de tópicos revisada por criterio humano.
    
    La taxonomía vincula cada ``topic_id`` de BERTopic con una
    ``problem_category`` conceptual y con una decisión ``include_as_problem``.
    Esta separación conserva la distinción entre el descubrimiento no supervisado
    de tópicos y la interpretación conceptual humana posterior utilizada para
    construir las categorías sustantivas de problemas.
    
    La huella SHA-256 del artefacto de taxonomía se verifica antes de procesarlo y
    el resultado se conserva en memoria después de una carga exitosa. Por tanto,
    el análisis en tiempo de ejecución consume la taxonomía validada en lugar de
    reconstruirla o modificarla dinámicamente.
    """
    global _TOPIC_TAXONOMY_CACHE

    if _TOPIC_TAXONOMY_CACHE is not None:
        return _TOPIC_TAXONOMY_CACHE

    _validate_frozen_artifact(
        TOPIC_TAXONOMY_PATH,
        EXPECTED_TOPIC_TAXONOMY_SHA256,
        "topic taxonomy",
    )

    taxonomy = {}

    with open(TOPIC_TAXONOMY_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        required = {
            "topic_id",
            "topic_type",
            "problem_category",
            "human_label",
            "include_as_problem",
        }

        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                "The taxonomy is missing required columns: "
                + ", ".join(sorted(missing))
            )

        for row in reader:
            try:
                topic_id = int(row["topic_id"])
            except (TypeError, ValueError):
                raise ValueError(
                    f"Invalid topic_id in taxonomy: {row['topic_id']!r}"
                )

            if topic_id in taxonomy:
                raise ValueError(
                    f"Duplicate topic_id in taxonomy: {topic_id}"
                )

            include_raw = str(row["include_as_problem"]).strip().upper()

            if include_raw not in {"YES", "NO"}:
                raise ValueError(
                    "include_as_problem debe ser YES o NO "
                    f"para topic_id={topic_id}"
                )

            taxonomy[topic_id] = {
                "topic_type": str(row["topic_type"]).strip(),
                "problem_category": str(row["problem_category"]).strip(),
                "human_label": str(row["human_label"]).strip(),
                "include_as_problem": include_raw == "YES",
            }

    _TOPIC_TAXONOMY_CACHE = taxonomy
    return taxonomy

SPANISH_STOPWORDS = {
    "el", "la", "los", "las", "de", "del", "en", "y", "a", "que", "es",
    "un", "una", "no", "se", "por", "con", "para", "lo", "más", "muy",
    "esta", "este", "pero", "como", "su", "sus", "me", "mi", "ya", "al",
    "le", "les", "o", "si", "porque", "hay", "he", "ha", "app",
    "aplicacion", "aplicación",
}


class LocalReviewsToolError(RuntimeError):
    """Controlled exception for local-review tool failures.

    Excepción controlada para fallos de la herramienta de reseñas locales.
    """


def _resolve_column(available_columns: list[str], candidates: list[str]) -> str:
    for candidate in candidates:
        if candidate in available_columns:
            return candidate
    raise LocalReviewsToolError(
        f"No candidate column found. Searched: {candidates}. "
        f"Available columns: {available_columns}"
    )


def derive_sentiment_from_rating(rating) -> str:
    """
    Convert a valid 1–5 rating into the deterministic sentiment classes used
    by the local-review analysis: 1–2 -> NEG, 3 -> NEU, and 4–5 -> POS.
    Invalid, non-numeric, NaN, non-integral, or out-of-range values raise
    ValueError.

    Convierte una calificación válida de 1 a 5 en las clases determinísticas
    de sentimiento utilizadas en el análisis de reseñas locales:
    1–2 -> NEG, 3 -> NEU y 4–5 -> POS. Los valores inválidos, no numéricos,
    NaN, no enteros o fuera de rango generan ValueError.
    """
    try:
        numeric_rating = float(rating)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Non-numeric rating: {rating!r}") from exc

    if numeric_rating != numeric_rating:
        raise ValueError(f"Invalid rating (NaN): {rating!r}")

    if not numeric_rating.is_integer():
        raise ValueError(
            f"Rating must represent an exact integer between 1 and 5: {rating!r}"
        )

    rating_int = int(numeric_rating)

    if rating_int not in (1, 2, 3, 4, 5):
        raise ValueError(f"Rating outside the valid range 1-5: {rating!r}")

    if rating_int <= 2:
        return "NEG"
    if rating_int == 3:
        return "NEU"
    return "POS"


def resolve_sentiment_or_derive(
    rows: list[dict], columns: list[str]
) -> tuple[list[dict], str, int]:
    """
    Resolve the sentiment field used for analysis. If an existing sentiment
    column is available, it is used as-is. Otherwise, sentiment is derived
    deterministically from a rating column and rows with invalid ratings are
    excluded. Returns the retained rows, the sentiment-column name, and the
    number of excluded invalid-rating rows.

    Resuelve el campo de sentimiento utilizado para el análisis. Si existe
    una columna de sentimiento, se utiliza directamente. En caso contrario,
    el sentimiento se deriva de forma determinística a partir de una columna
    de calificación y se excluyen las filas con calificaciones inválidas.
    Retorna las filas conservadas, el nombre de la columna de sentimiento y
    la cantidad de filas excluidas por calificaciones inválidas.
    """
    try:
        sentiment_col = _resolve_column(columns, SENTIMENT_COLUMN_CANDIDATES)
        return rows, sentiment_col, 0
    except LocalReviewsToolError:
        pass

    try:
        rating_col = _resolve_column(columns, RATING_COLUMN_CANDIDATES)
    except LocalReviewsToolError as exc:
        raise LocalReviewsToolError(
            "Neither a usable sentiment column nor a usable rating column is available. "
            f"Sentiment candidates: {SENTIMENT_COLUMN_CANDIDATES}. "
            f"Rating candidates: {RATING_COLUMN_CANDIDATES}. "
            f"Available columns: {columns}"
        ) from exc

    valid_rows = []
    invalid_count = 0

    for original_row in rows:
        row = dict(original_row)
        try:
            row["_derived_sentiment"] = derive_sentiment_from_rating(row.get(rating_col))
            valid_rows.append(row)
        except ValueError:
            invalid_count += 1

    return valid_rows, "_derived_sentiment", invalid_count


def _load_reviews(start_date: str, end_date: str) -> list[dict]:
    if not REVIEWS_DB_PATH:
        raise LocalReviewsToolError("REVIEWS_DB_PATH is not configured.")

    if not os.path.exists(REVIEWS_DB_PATH):
        raise LocalReviewsToolError(f"Corpus file not found: {REVIEWS_DB_PATH}")

    try:
        start = datetime.fromisoformat(start_date)
        end_inclusive = datetime.fromisoformat(end_date)
    except ValueError as exc:
        raise LocalReviewsToolError(
            "Invalid dates. Expected ISO format YYYY-MM-DD."
        ) from exc

    if start > end_inclusive:
        raise LocalReviewsToolError("start_date cannot be later than end_date.")

    # Interpret the interval as complete calendar days:
    # [start of start_date, start of the day after end_date).
    from datetime import timedelta
    end = end_inclusive + timedelta(days=1)

    ext = os.path.splitext(REVIEWS_DB_PATH)[1].lower()

    if ext in (".db", ".sqlite", ".sqlite3"):
        return _load_from_sqlite(start, end)
    if ext == ".csv":
        return _load_from_csv(start, end)

    raise LocalReviewsToolError(f"Unsupported file extension: {ext}")


def _load_from_sqlite(start: datetime, end: datetime) -> list[dict]:
    conn = sqlite3.connect(REVIEWS_DB_PATH)
    conn.row_factory = sqlite3.Row

    try:
        cursor = conn.execute("SELECT * FROM reviews LIMIT 1")
        columns = [d[0] for d in cursor.description]
        date_col = _resolve_column(columns, DATE_COLUMN_CANDIDATES)
        all_rows = conn.execute("SELECT * FROM reviews").fetchall()
    finally:
        conn.close()

    result = []
    for row in all_rows:
        row_dict = dict(row)
        row_date = _parse_row_date(row_dict.get(date_col))
        if row_date and start <= row_date < end:
            result.append(row_dict)

    return result


def _load_from_csv(start: datetime, end: datetime) -> list[dict]:
    with open(REVIEWS_DB_PATH, "r", encoding="utf-8") as file_obj:
        reader = csv.DictReader(file_obj)
        columns = reader.fieldnames or []
        date_col = _resolve_column(columns, DATE_COLUMN_CANDIDATES)

        result = []
        for row_dict in reader:
            row_date = _parse_row_date(row_dict.get(date_col))
            if row_date and start <= row_date < end:
                result.append(row_dict)

    return result


def _parse_row_date(raw_value) -> datetime | None:
    if raw_value is None:
        return None

    value = str(raw_value).strip()
    if not value:
        return None

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        pass

    for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y/%m/%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue

    return None



def _analyze_problem_categories(
    negative_rows: list[dict],
) -> dict:
    """Aggregate substantive problem categories for negative reviews.

    The analysis follows the frozen semantic linkage
    ``review_id -> topic_id -> problem_category``. BERTopic is not fitted
    or used for inference at query time; this function only consumes the
    validated offline topic assignments and the frozen human-reviewed
    taxonomy.

    Negative reviews are distinguished into four analytically relevant
    states: reviews without a frozen topic assignment, BERTopic noise
    (``topic_id == -1``), assigned topics excluded by the taxonomy, and
    substantive problem categories. Missing assignments and excluded cases
    are reported explicitly and are not imputed.

    Category percentages are reported using two denominators:
    ``pct_of_substantive_problems`` uses only reviews assigned to substantive
    categories, whereas ``pct_of_negative_reviews`` uses all negative reviews
    in the queried subset. ``substantive_coverage_pct`` therefore quantifies
    the proportion of negative reviews represented by substantive categories.

    ``main_problem`` is the most frequent substantive problem category. It is
    a semantic summary derived from the frozen BERTopic-plus-taxonomy layer,
    not from lexical keyword frequency and not from new topic-model inference.

    Agrega categorías sustantivas de problemas para las reseñas negativas.

    El análisis sigue el vínculo semántico congelado
    ``review_id -> topic_id -> problem_category``. BERTopic no se entrena ni
    realiza inferencia durante la consulta; esta función únicamente utiliza
    las asignaciones de tópicos validadas fuera de línea y la taxonomía
    congelada revisada por criterio humano.

    Las reseñas negativas se distinguen en cuatro estados analíticamente
    relevantes: reseñas sin asignación congelada de tópico, ruido de BERTopic
    (``topic_id == -1``), tópicos asignados pero excluidos por la taxonomía y
    categorías sustantivas de problemas. Las asignaciones ausentes y los casos
    excluidos se reportan explícitamente y no se imputan.

    Los porcentajes por categoría utilizan dos denominadores:
    ``pct_of_substantive_problems`` considera únicamente las reseñas asignadas
    a categorías sustantivas, mientras que ``pct_of_negative_reviews`` utiliza
    todas las reseñas negativas del subconjunto consultado.
    ``substantive_coverage_pct`` cuantifica, por tanto, la proporción de
    reseñas negativas representada por categorías sustantivas.

    ``main_problem`` es la categoría sustantiva de problema más frecuente. Es
    un resumen semántico derivado de la capa congelada BERTopic más taxonomía,
    no de la frecuencia léxica de palabras clave ni de una nueva inferencia
    del modelo de tópicos.
    """
    if not negative_rows:
        return {
            "negative_reviews": 0,
            "with_topic_assignment": 0,
            "missing_topic_assignment_count": 0,
            "assigned_non_noise_count": 0,
            "topic_noise_count": 0,
            "excluded_assigned_review_count": 0,
            "substantive_problem_count": 0,
            "substantive_coverage_pct": 0.0,
            "problem_categories": [],
            "main_problem": None,
        }

    columns = list(negative_rows[0].keys())
    review_id_col = _resolve_column(
        columns,
        ["review_id", "id", "reviewId", "review_id_str"],
    )

    assignments = _load_topic_assignments()
    taxonomy = _load_topic_taxonomy()

    category_counts = Counter()

    with_assignment = 0
    missing_assignment_count = 0
    noise_count = 0
    excluded_count = 0
    substantive_count = 0

    for row in negative_rows:
        review_id = str(row.get(review_id_col, "")).strip()

        if not review_id:
            continue

        topic_id = assignments.get(review_id)

        # Negative reviews absent from the frozen assignment artifact are not imputed.
        if topic_id is None:
            missing_assignment_count += 1
            continue

        with_assignment += 1

        if topic_id == -1:
            noise_count += 1
            continue

        topic_meta = taxonomy.get(topic_id)

        if topic_meta is None:
            raise LocalReviewsToolError(
                f"topic_id={topic_id} is not present in the frozen taxonomy."
            )

        if not topic_meta["include_as_problem"]:
            excluded_count += 1
            continue

        category = topic_meta["problem_category"]

        category_counts[category] += 1
        substantive_count += 1

    problem_categories = []

    for category, count in sorted(
        category_counts.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        pct_substantive = (
            round(100 * count / substantive_count, 4)
            if substantive_count
            else 0.0
        )

        pct_all_negative = (
            round(100 * count / len(negative_rows), 4)
            if negative_rows
            else 0.0
        )

        problem_categories.append(
            {
                "category": category,
                "count": count,
                "pct_of_substantive_problems": pct_substantive,
                "pct_of_negative_reviews": pct_all_negative,
            }
        )

    main_problem = (
        problem_categories[0]
        if problem_categories
        else None
    )

    substantive_coverage_pct = (
        round(100 * substantive_count / len(negative_rows), 4)
        if negative_rows
        else 0.0
    )

    return {
        "negative_reviews": len(negative_rows),
        "with_topic_assignment": with_assignment,
        "missing_topic_assignment_count": missing_assignment_count,
        "assigned_non_noise_count": with_assignment - noise_count,
        "topic_noise_count": noise_count,
        "excluded_assigned_review_count": excluded_count,
        "substantive_problem_count": substantive_count,
        "substantive_coverage_pct": substantive_coverage_pct,
        "problem_categories": problem_categories,
        "main_problem": main_problem,
    }


def query_local_reviews(
    start_date: str,
    end_date: str,
    query: str | None = None,
) -> dict:
    """
    Query local reviews for a date range and return aggregate analytical
    results. An optional case-insensitive text filter may be applied before
    analysis. The function resolves or derives sentiment, reports invalid
    ratings, summarizes sentiment, extracts frequent terms and examples from
    negative reviews, and integrates frozen topic/taxonomy assignments through
    the problem-category analysis.

    Consulta reseñas locales para un rango de fechas y retorna resultados
    analíticos agregados. Puede aplicarse un filtro textual opcional sin
    distinción entre mayúsculas y minúsculas antes del análisis. La función
    resuelve o deriva el sentimiento, reporta calificaciones inválidas,
    resume la distribución de sentimiento, extrae términos frecuentes y
    ejemplos de reseñas negativas, e integra las asignaciones congeladas de
    tópicos y taxonomía mediante el análisis de categorías de problema.
    """
    rows = _load_reviews(start_date, end_date)

    if query and rows:
        content_col_for_query = _resolve_column(
            list(rows[0].keys()),
            CONTENT_COLUMN_CANDIDATES,
        )
        query_lower = query.lower()
        rows = [
            row
            for row in rows
            if query_lower in str(row.get(content_col_for_query, "")).lower()
        ]

    total_reviews_raw = len(rows)

    if not rows:
        return {
            "source": "local_reviews",
            "date_range": {"start": start_date, "end": end_date},
            "total_reviews": 0,
            "valid_reviews_for_analysis": 0,
            "sentiment_distribution": {},
            "top_negative_keywords": [],
            "sample_negative_reviews": [],
            "negative_review_count": 0,
            "topic_coverage": {
                "with_topic_assignment": 0,
                "missing_topic_assignment_count": 0,
                "assigned_non_noise_count": 0,
                "topic_noise_count": 0,
                "excluded_assigned_review_count": 0,
                "substantive_problem_count": 0,
                "substantive_coverage_pct": 0.0,
            },
            "problem_categories": [],
            "main_problem": None,
            "invalid_rating_count": 0,
            "invalid_rating_pct": 0.0,
        }

    columns = list(rows[0].keys())
    rows, sentiment_col, invalid_rating_count = resolve_sentiment_or_derive(
        rows,
        columns,
    )

    valid_reviews_for_analysis = len(rows)

    invalid_rating_pct = (
        round(100 * invalid_rating_count / total_reviews_raw, 4)
        if total_reviews_raw
        else 0.0
    )

    if invalid_rating_pct > 0.1:
        print(
            f"WARNING [local_reviews_tool]: {invalid_rating_count} rows "
            f"({invalid_rating_pct}%) were excluded due to invalid ratings.",
            file=sys.stderr,
        )

    if not rows:
        return {
            "source": "local_reviews",
            "date_range": {"start": start_date, "end": end_date},
            "total_reviews": total_reviews_raw,
            "valid_reviews_for_analysis": 0,
            "sentiment_distribution": {},
            "top_negative_keywords": [],
            "sample_negative_reviews": [],
            "negative_review_count": 0,
            "topic_coverage": {
                "with_topic_assignment": 0,
                "missing_topic_assignment_count": 0,
                "assigned_non_noise_count": 0,
                "topic_noise_count": 0,
                "excluded_assigned_review_count": 0,
                "substantive_problem_count": 0,
                "substantive_coverage_pct": 0.0,
            },
            "problem_categories": [],
            "main_problem": None,
            "invalid_rating_count": invalid_rating_count,
            "invalid_rating_pct": invalid_rating_pct,
        }

    columns = list(rows[0].keys())
    content_col = _resolve_column(columns, CONTENT_COLUMN_CANDIDATES)

    sentiment_counts = Counter(
        str(row.get(sentiment_col, "UNKNOWN"))
        for row in rows
    )

    negative_rows = [
        row
        for row in rows
        if str(row.get(sentiment_col, "")) in NEGATIVE_LABELS
    ]

    word_counter = Counter()

    for row in negative_rows:
        text = str(row.get(content_col, "")).lower()
        words = re.findall(r"[a-záéíóúñü]+", text)
        word_counter.update(
            word
            for word in words
            if word not in SPANISH_STOPWORDS and len(word) > 2
        )

    problem_analysis = _analyze_problem_categories(negative_rows)

    return {
        "source": "local_reviews",
        "date_range": {"start": start_date, "end": end_date},
        "total_reviews": total_reviews_raw,
        "valid_reviews_for_analysis": valid_reviews_for_analysis,
        "sentiment_distribution": dict(sentiment_counts),
        "top_negative_keywords": word_counter.most_common(15),
        "sample_negative_reviews": [
            str(row.get(content_col, ""))
            for row in negative_rows[:5]
        ],
        "negative_review_count": problem_analysis["negative_reviews"],
        "topic_coverage": {
            "with_topic_assignment": problem_analysis["with_topic_assignment"],
            "missing_topic_assignment_count": problem_analysis["missing_topic_assignment_count"],
            "assigned_non_noise_count": problem_analysis["assigned_non_noise_count"],
            "topic_noise_count": problem_analysis["topic_noise_count"],
            "excluded_assigned_review_count": problem_analysis["excluded_assigned_review_count"],
            "substantive_problem_count": problem_analysis["substantive_problem_count"],
            "substantive_coverage_pct": problem_analysis["substantive_coverage_pct"],
        },
        "problem_categories": problem_analysis["problem_categories"],
        "main_problem": problem_analysis["main_problem"],
        "invalid_rating_count": invalid_rating_count,
        "invalid_rating_pct": invalid_rating_pct,
    }


if __name__ == "__main__":
    import json

    if not REVIEWS_DB_PATH:
        print("Configurar REVIEWS_DB_PATH antes de ejecutar la prueba.")
    else:
        result = query_local_reviews("2023-01-01", "2023-12-31")
        print(json.dumps(result, indent=2, ensure_ascii=False))
