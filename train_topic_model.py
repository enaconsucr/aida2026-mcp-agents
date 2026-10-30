"""
train_topic_model.py — Entrena BERTopic UNA SOLA VEZ, offline, sobre todas
las reseñas negativas válidas del corpus local y congela el resultado:
modelo, asignación de topic_id por reseña y metadatos de reproducibilidad.

La herramienta MCP `query_local_reviews`, implementada en
`local_reviews_tool.py`, consumirá posteriormente las asignaciones
precomputadas. BERTopic NO se reentrena ni ejecuta inferencia dentro de la
herramienta MCP.

DECISIONES DE DISEÑO:
  - Solo reseñas NEGATIVAS entran al entrenamiento.
  - El modelo se entrena una sola vez sobre todo el histórico disponible.
  - topic_id = -1 (ruido de HDBSCAN) se conserva en los datos, pero se
    excluye del cálculo de prevalencia entre tópicos válidos y se reporta
    por separado.
  - "Problema principal" en un período se definirá posteriormente en
    local_reviews_tool.py como el tópico no-ruido con mayor conteo dentro
    de las reseñas negativas del período.
  - Se fijan las semillas controlables del pipeline y se congelan los
    artefactos y versiones del entorno. No se afirma determinismo absoluto
    de toda la pila de dependencias.

USO:
    export REVIEWS_DB_PATH=/ruta/al/corpus.csv
    python train_topic_model.py \
        --output-dir models/bertopic_negative_reviews_v1

VALIDACIÓN PREVIA SIN ENTRENAR BERTopic:
    python train_topic_model.py \
        --output-dir models/bertopic_negative_reviews_v1 \
        --filter-only

DEPENDENCIAS PARA EL ENTRENAMIENTO:
    bertopic
    sentence-transformers
    umap-learn
    hdbscan
    scikit-learn

IMPORTANTE:
  - Cada versión del modelo debe usar un output-dir distinto.
  - Por defecto se impide sobrescribir un directorio no vacío.
  - Tras el entrenamiento se genera automáticamente
    requirements_topic_model.txt dentro del mismo output-dir.
  - El modelo NO debe conectarse al benchmark hasta completar la revisión
    manual de reports/topic_summary.csv.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from local_reviews_tool import (
    REVIEWS_DB_PATH,
    DATE_COLUMN_CANDIDATES,
    SENTIMENT_COLUMN_CANDIDATES,
    RATING_COLUMN_CANDIDATES,
    CONTENT_COLUMN_CANDIDATES,
    ID_COLUMN_CANDIDATES,
    NEGATIVE_LABELS,
    _resolve_column,
    derive_sentiment_from_rating,
    LocalReviewsToolError,
)

SEED = 42
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
SENTIMENT_RULE = "1-2=NEG;3=NEU;4-5=POS"

SPANISH_STOPWORDS_EXTENDED = sorted({
    "el", "la", "los", "las", "de", "del", "en", "y", "a", "que", "es",
    "un", "una", "unos", "unas", "no", "se", "por", "con", "para", "lo",
    "más", "muy", "esta", "este", "estas", "estos", "pero", "como", "su",
    "sus", "me", "mi", "ya", "al", "le", "les", "o", "si", "porque",
    "hay", "he", "ha", "app", "aplicacion", "aplicación", "esa", "ese",
    "esas", "esos", "soy", "eres", "somos", "son", "fue", "fueron",
    "ser", "estar", "está", "están", "tiene", "tienen", "tener", "sin",
    "sobre", "entre", "cuando", "todo", "toda", "todos", "todas", "otro",
    "otra", "yo", "tu", "tú", "él", "ella", "nosotros", "ellos", "ellas",
    "nos", "cual", "cuales", "quien", "quienes", "donde", "cómo", "qué",
    "cuándo", "también", "solo", "sólo", "asi", "así", "cada",
})


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as file_obj:
        for chunk in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(distribution_name: str) -> str | None:
    try:
        return importlib_metadata.version(distribution_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def collect_package_versions() -> dict[str, str | None]:
    names = [
        "bertopic",
        "sentence-transformers",
        "umap-learn",
        "hdbscan",
        "scikit-learn",
        "numpy",
        "pandas",
        "torch",
    ]
    return {name: package_version(name) for name in names}


def load_all_reviews(reviews_db_path: str) -> pd.DataFrame:
    if not reviews_db_path:
        raise LocalReviewsToolError(
            "REVIEWS_DB_PATH no está configurado. Exportar la variable de "
            "entorno o pasar --reviews-db."
        )

    path = os.path.abspath(os.path.expanduser(reviews_db_path))
    if not os.path.exists(path):
        raise LocalReviewsToolError(f"Archivo de corpus no encontrado: {path}")

    ext = os.path.splitext(path)[1].lower()

    if ext == ".csv":
        return pd.read_csv(path)

    if ext in (".db", ".sqlite", ".sqlite3"):
        import sqlite3
        conn = sqlite3.connect(path)
        try:
            return pd.read_sql_query("SELECT * FROM reviews", conn)
        finally:
            conn.close()

    raise LocalReviewsToolError(
        f"Extensión no soportada: '{ext}' (esperado .csv/.db/.sqlite/.sqlite3)."
    )


def _resolve_or_derive_sentiment_column(
    df: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, str, int, str]:
    try:
        sentiment_col = _resolve_column(columns, SENTIMENT_COLUMN_CANDIDATES)
        return df, sentiment_col, 0, f"existing_column:{sentiment_col}"
    except LocalReviewsToolError:
        pass

    try:
        rating_col = _resolve_column(columns, RATING_COLUMN_CANDIDATES)
    except LocalReviewsToolError as exc:
        raise LocalReviewsToolError(
            "No existe columna de sentimiento ni columna de calificación "
            f"utilizable. Sentimiento buscado: {SENTIMENT_COLUMN_CANDIDATES}. "
            f"Calificación buscada: {RATING_COLUMN_CANDIDATES}. "
            f"Columnas disponibles: {columns}"
        ) from exc

    def _try_derive(rating):
        try:
            return derive_sentiment_from_rating(rating)
        except ValueError:
            return None

    result = df.copy()
    result["_derived_sentiment"] = result[rating_col].apply(_try_derive)
    invalid_count = int(result["_derived_sentiment"].isna().sum())
    result = result[result["_derived_sentiment"].notna()].copy()

    return result, "_derived_sentiment", invalid_count, f"derived_from_rating:{rating_col}"


def filter_negative_reviews(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    columns = list(df.columns)
    total_rows_considered = len(df)

    valid_df, sentiment_col, invalid_rating_count, sentiment_source = (
        _resolve_or_derive_sentiment_column(df, columns)
    )

    current_columns = list(valid_df.columns)
    content_col = _resolve_column(current_columns, CONTENT_COLUMN_CANDIDATES)
    id_col = _resolve_column(current_columns, ID_COLUMN_CANDIDATES)
    date_col = _resolve_column(current_columns, DATE_COLUMN_CANDIDATES)

    is_negative = valid_df[sentiment_col].astype(str).isin(NEGATIVE_LABELS)

    has_content = (
        valid_df[content_col].notna()
        & valid_df[content_col].astype(str).str.strip().ne("")
    )

    negative_before_content = int(is_negative.sum())
    empty_negative_count = int((is_negative & ~has_content).sum())

    negative_df = valid_df.loc[
        is_negative & has_content,
        [id_col, content_col, date_col, sentiment_col],
    ].copy()

    negative_df.columns = [
        "review_id",
        "content",
        "date",
        "sentiment_label",
    ]
    negative_df = negative_df.reset_index(drop=True)

    metrics = {
        "total_rows_considered": total_rows_considered,
        "invalid_rating_count": invalid_rating_count,
        "invalid_rating_pct": (
            round(100 * invalid_rating_count / total_rows_considered, 4)
            if total_rows_considered
            else 0.0
        ),
        "sentiment_source": sentiment_source,
        "negative_before_content_filter": negative_before_content,
        "empty_negative_reviews_excluded": empty_negative_count,
        "negative_reviews_valid": len(negative_df),
        "content_column": content_col,
        "id_column": id_col,
        "date_column": date_col,
        "sentiment_column_used": sentiment_col,
    }

    return negative_df, metrics


def validate_review_ids(negative_df: pd.DataFrame) -> None:
    ids = negative_df["review_id"]

    missing_mask = ids.isna() | ids.astype(str).str.strip().eq("")
    missing_count = int(missing_mask.sum())
    if missing_count:
        raise LocalReviewsToolError(
            f"Hay {missing_count} review_id nulos o vacíos en las reseñas "
            "negativas válidas. Corregir antes de entrenar."
        )

    duplicate_mask = ids.astype(str).duplicated(keep=False)
    duplicate_count = int(duplicate_mask.sum())
    if duplicate_count:
        examples = (
            ids[duplicate_mask]
            .astype(str)
            .drop_duplicates()
            .head(10)
            .tolist()
        )
        raise LocalReviewsToolError(
            f"Hay {duplicate_count} filas con review_id duplicado. "
            f"Ejemplos: {examples}. Corregir antes de entrenar."
        )


def train_bertopic_model(texts: list[str], min_cluster_size: int, seed: int):
    try:
        from bertopic import BERTopic
        from sentence_transformers import SentenceTransformer
        from umap import UMAP
        from hdbscan import HDBSCAN
        from sklearn.feature_extraction.text import CountVectorizer
    except ImportError as exc:
        raise RuntimeError(
            "Faltan dependencias de BERTopic. Instalar únicamente después "
            "de validar el filtrado con --filter-only."
        ) from exc

    random.seed(seed)
    np.random.seed(seed)

    embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    umap_model = UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=seed,
    )

    hdbscan_model = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=None,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True,
    )

    vectorizer_model = CountVectorizer(
        stop_words=SPANISH_STOPWORDS_EXTENDED,
        ngram_range=(1, 2),
    )

    topic_model = BERTopic(
        language="spanish",
        embedding_model=embedding_model,
        umap_model=umap_model,
        hdbscan_model=hdbscan_model,
        vectorizer_model=vectorizer_model,
        top_n_words=10,
        calculate_probabilities=False,
        nr_topics=None,
        verbose=True,
    )

    topics, _ = topic_model.fit_transform(texts)
    return topic_model, topics


def build_topic_summary(topic_model, texts: list[str], topics: list[int]) -> pd.DataFrame:
    total_negative = len(texts)
    total_non_noise = sum(1 for topic in topics if int(topic) != -1)

    topic_info = topic_model.get_topic_info()
    rows: list[dict] = []

    for _, row in topic_info.iterrows():
        topic_id = int(row["Topic"])
        if topic_id == -1:
            continue

        count = int(row["Count"])
        top_words = [
            word for word, _ in (topic_model.get_topic(topic_id) or [])
        ][:10]

        representative_docs = topic_model.get_representative_docs(topic_id) or []

        rows.append(
            {
                "topic_id": topic_id,
                "count": count,
                "prevalence_pct_of_negative": (
                    round(100 * count / total_negative, 2)
                    if total_negative else None
                ),
                "prevalence_pct_of_non_noise": (
                    round(100 * count / total_non_noise, 2)
                    if total_non_noise else None
                ),
                "top_terms": ", ".join(top_words),
                "representative_reviews": " ||| ".join(
                    str(doc) for doc in representative_docs[:5]
                ),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "topic_id",
                "count",
                "prevalence_pct_of_negative",
                "prevalence_pct_of_non_noise",
                "top_terms",
                "representative_reviews",
            ]
        )

    return pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)


def validate_topics_checklist(
    summary_df: pd.DataFrame,
    noise_count: int,
    total_negative: int,
) -> dict:
    noise_pct = (
        round(100 * noise_count / total_negative, 2)
        if total_negative else None
    )

    small_topics = summary_df[summary_df["count"] < 10]

    top5 = []
    for record in (
        summary_df.head(5)[["topic_id", "count", "top_terms"]].to_dict("records")
    ):
        top5.append(
            {
                "topic_id": int(record["topic_id"]),
                "count": int(record["count"]),
                "top_terms": str(record["top_terms"]),
            }
        )

    return {
        "noise_pct": noise_pct,
        "noise_pct_within_threshold_35pct": (
            noise_pct is not None and noise_pct <= 35.0
        ),
        "n_topics_found": int(len(summary_df)),
        "n_small_topics_below_10": int(len(small_topics)),
        "top5_topics_for_manual_review": top5,
        "note": (
            "Este resumen NO reemplaza la validación manual: lectura de "
            "términos y reseñas representativas, prueba del intruso y "
            "contraste con top_negative_keywords."
        ),
    }


def prepare_output_dir(
    output_dir: str,
    allow_overwrite: bool,
) -> tuple[Path, Path, Path]:
    output_path = Path(output_dir).expanduser().resolve()

    if output_path.exists() and any(output_path.iterdir()) and not allow_overwrite:
        raise LocalReviewsToolError(
            f"El directorio de salida ya existe y no está vacío: "
            f"{output_path}. Use otra versión (recomendado) o "
            "--allow-overwrite si realmente desea reemplazarlo."
        )

    output_path.mkdir(parents=True, exist_ok=True)

    data_dir = output_path / "data"
    reports_dir = output_path / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    return output_path, data_dir, reports_dir


def write_pip_freeze(output_path: Path) -> Path:
    requirements_path = output_path / "requirements_topic_model.txt"

    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        capture_output=True,
        text=True,
        check=True,
    )
    requirements_path.write_text(result.stdout, encoding="utf-8")
    return requirements_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Entrenar BERTopic offline sobre todas las reseñas negativas "
            "válidas del corpus local."
        )
    )
    parser.add_argument(
        "--reviews-db",
        default=REVIEWS_DB_PATH,
        help="Ruta al corpus (default: $REVIEWS_DB_PATH).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help=(
            "Directorio autocontenido del artefacto, por ejemplo "
            "models/bertopic_negative_reviews_v1"
        ),
    )
    parser.add_argument(
        "--min-cluster-size",
        type=int,
        default=55,
        help="HDBSCAN min_cluster_size (default: 55).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Semilla controlable principal (default: 42).",
    )
    parser.add_argument(
        "--filter-only",
        action="store_true",
        help=(
            "Valida carga, derivación de sentimiento, contenido y review_id "
            "sin importar ni entrenar BERTopic."
        ),
    )
    parser.add_argument(
        "--allow-overwrite",
        action="store_true",
        help="Permite escribir sobre un output-dir no vacío.",
    )
    args = parser.parse_args()

    reviews_path = os.path.abspath(os.path.expanduser(args.reviews_db))

    print(f"Cargando corpus completo desde: {reviews_path}")
    df = load_all_reviews(reviews_path)
    print(f"  {len(df)} filas totales en el corpus.")

    negative_df, filter_metrics = filter_negative_reviews(df)
    validate_review_ids(negative_df)

    print(
        f"  {filter_metrics['negative_before_content_filter']} reseñas "
        "negativas antes del filtro de contenido."
    )
    print(
        f"  {filter_metrics['empty_negative_reviews_excluded']} reseñas "
        "negativas excluidas por contenido vacío/NaN."
    )
    print(f"  {len(negative_df)} reseñas negativas válidas para BERTopic.")
    print(
        "  invalid_rating_count = "
        f"{filter_metrics['invalid_rating_count']} "
        "(invalid_rating_pct = "
        f"{filter_metrics['invalid_rating_pct']}%)"
    )
    print(f"  sentiment_source = {filter_metrics['sentiment_source']}")
    print("  review_id: validación OK (sin nulos/vacíos ni duplicados).")

    if filter_metrics["invalid_rating_pct"] > 0.1:
        print(
            "ADVERTENCIA: invalid_rating_pct supera 0.1%. No se aborta "
            "automáticamente, pero debe revisarse antes de aceptar el modelo."
        )

    if args.filter_only:
        print("\nFILTER-ONLY OK: no se importó ni entrenó BERTopic.")
        print(
            json.dumps(
                {
                    "corpus_rows": len(df),
                    **filter_metrics,
                    "corpus_sha256": sha256_file(reviews_path),
                },
                indent=2,
                ensure_ascii=False,
            )
        )
        return

    if len(negative_df) < args.min_cluster_size * 2:
        raise LocalReviewsToolError(
            f"Solo hay {len(negative_df)} reseñas negativas válidas para "
            f"min_cluster_size={args.min_cluster_size}. Revisar parámetros "
            "antes de entrenar."
        )

    output_path, data_dir, reports_dir = prepare_output_dir(
        args.output_dir,
        args.allow_overwrite,
    )

    texts = negative_df["content"].astype(str).tolist()

    print(
        "\nEntrenando BERTopic "
        f"(embedding={EMBEDDING_MODEL_NAME}, "
        f"min_cluster_size={args.min_cluster_size}, seed={args.seed})..."
    )

    topic_model, topics = train_bertopic_model(
        texts,
        args.min_cluster_size,
        args.seed,
    )

    negative_df["topic_id"] = [int(topic) for topic in topics]
    noise_count = int((negative_df["topic_id"] == -1).sum())

    summary_df = build_topic_summary(topic_model, texts, topics)
    checklist = validate_topics_checklist(
        summary_df,
        noise_count,
        len(negative_df),
    )

    model_path = output_path / "model"
    topic_model.save(
        str(model_path),
        serialization="safetensors",
        save_ctfidf=True,
        save_embedding_model=EMBEDDING_MODEL_NAME,
    )
    print(f"\nModelo guardado en: {model_path}")

    negative_reviews_path = data_dir / "negative_reviews_with_topics.csv"
    negative_df.to_csv(
        negative_reviews_path,
        index=False,
        encoding="utf-8",
    )
    print(f"CSV de reseñas negativas con topic_id: {negative_reviews_path}")

    topic_summary_path = reports_dir / "topic_summary.csv"
    summary_df.to_csv(
        topic_summary_path,
        index=False,
        encoding="utf-8",
    )
    print(f"Resumen de tópicos: {topic_summary_path}")

    requirements_path = write_pip_freeze(output_path)
    print(f"Entorno congelado en: {requirements_path}")

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "reviews_db_path": reviews_path,
        "corpus_sha256": sha256_file(reviews_path),
        "n_corpus_rows": int(len(df)),
        "sentiment_source": filter_metrics["sentiment_source"],
        "sentiment_rule_if_derived": SENTIMENT_RULE,
        "negative_labels_used": sorted(str(x) for x in NEGATIVE_LABELS),
        "n_negative_before_content_filter": int(
            filter_metrics["negative_before_content_filter"]
        ),
        "n_empty_negative_reviews_excluded": int(
            filter_metrics["empty_negative_reviews_excluded"]
        ),
        "n_negative_reviews_total": int(len(negative_df)),
        "invalid_rating_count": int(filter_metrics["invalid_rating_count"]),
        "invalid_rating_pct": filter_metrics["invalid_rating_pct"],
        "id_column_source": filter_metrics["id_column"],
        "content_column_source": filter_metrics["content_column"],
        "date_column_source": filter_metrics["date_column"],
        "embedding_model": EMBEDDING_MODEL_NAME,
        "embedding_model_revision": "e8f8c211226b894fcb81acc59f3b34ba3efd5f42",
        "embedding_model_revision_note": (
            "Revisión exacta del snapshot Hugging Face utilizado durante "
            "el entrenamiento definitivo."
        ),
        "bertopic_language": "spanish",
        "seed": int(args.seed),
        "min_cluster_size": int(args.min_cluster_size),
        "hdbscan_params": {
            "min_samples": None,
            "metric": "euclidean",
            "cluster_selection_method": "eom",
            "prediction_data": True,
        },
        "umap_params": {
            "n_neighbors": 15,
            "n_components": 5,
            "min_dist": 0.0,
            "metric": "cosine",
            "random_state": int(args.seed),
        },
        "vectorizer_ngram_range": [1, 2],
        "top_n_words": 10,
        "calculate_probabilities": False,
        "nr_topics": None,
        "n_topics_found": checklist["n_topics_found"],
        "noise_count": noise_count,
        "noise_pct": checklist["noise_pct"],
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "package_versions": collect_package_versions(),
        "requirements_freeze_file": requirements_path.name,
        "artifact_policy": (
            "Modelo entrenado offline una sola vez. No se reentrena ni "
            "ejecuta BERTopic dentro de la herramienta MCP."
        ),
    }

    metadata_path = output_path / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Metadata de reproducibilidad: {metadata_path}")

    print("\n" + "=" * 60)
    print("CHECKLIST CUANTITATIVO (no reemplaza validación manual)")
    print("=" * 60)
    print(json.dumps(checklist, indent=2, ensure_ascii=False))

    print(
        "\nSIGUIENTE PASO OBLIGATORIO: revisar manualmente "
        f"{topic_summary_path} mediante lectura de términos, reseñas "
        "representativas, prueba del intruso y contraste con "
        "top_negative_keywords. No conectar aún al benchmark."
    )


if __name__ == "__main__":
    main()
