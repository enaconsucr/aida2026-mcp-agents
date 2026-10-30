"""MCP server for the public-health evidence agent.

This module exposes the project's three validated data tools through the
Model Context Protocol (MCP): local EDUS review analysis, World Bank
indicators, and OECD health statistics. The server acts as a thin
orchestration layer: methodological transformations, validation rules, and
source-specific retrieval logic remain implemented in their respective data
modules rather than being duplicated here.

The ``data://schema`` resource provides models with concise methodological
and interpretive context for the exposed outputs. Its purpose is to support
correct tool selection and evidence interpretation without changing the
underlying data-processing procedures.

Servidor MCP para el agente de evidencia en salud pública.

Este módulo expone mediante el Model Context Protocol (MCP) las tres
herramientas de datos validadas del proyecto: análisis de reseñas locales de
EDUS, indicadores del Banco Mundial y estadísticas de salud de la OCDE. El
servidor funciona como una capa ligera de orquestación: las transformaciones
metodológicas, reglas de validación y procedimientos de consulta específicos
de cada fuente permanecen implementados en sus respectivos módulos de datos,
en lugar de duplicarse aquí.

El recurso ``data://schema`` proporciona a los modelos un contexto
metodológico e interpretativo conciso sobre las salidas expuestas. Su
propósito es favorecer una selección correcta de herramientas y una
interpretación adecuada de la evidencia sin modificar los procedimientos
subyacentes de procesamiento de datos.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.mcpserver import MCPServer

from worldbank_tool import get_worldbank_indicator, WorldBankToolError
from oecd_tool import get_oecd_indicator, OECDToolError
from local_reviews_tool import query_local_reviews, LocalReviewsToolError

mcp = MCPServer(
    name="public-health-data-agent",
    description=(
        "Exposes citizen sentiment and institutional health indicators "
        "(World Bank REST, OECD SDMX) as MCP tools for an LLM agent."
    ),
    instructions=(
        "Read the 'data://schema' resource before selecting a tool. "
        "It clarifies what each of the three sources represents and how "
        "they relate to each other — in particular, that World Bank and "
        "OECD may report the same conceptual indicator with slightly "
        "different values, and that a null value for one year does not "
        "mean a country lacks the indicator entirely."
    ),
)


@mcp.resource("data://schema")
def data_schema() -> str:
    """
    Provide methodological and interpretive context for the three data
    sources exposed by the MCP server before the agent selects a tool.

    The resource documents what each source represents, the principal output
    fields, and important interpretive limitations. It is intended to reduce
    ambiguity in tool selection and evidence interpretation without
    duplicating the data-processing logic implemented in the underlying
    modules.

    Proporciona contexto metodológico e interpretativo sobre las tres fuentes
    de datos expuestas por el servidor MCP antes de que el agente seleccione
    una herramienta.

    El recurso documenta qué representa cada fuente, los principales campos
    de salida y las limitaciones relevantes para su interpretación. Su
    propósito es reducir la ambigüedad en la selección de herramientas y en
    la interpretación de la evidencia sin duplicar la lógica de procesamiento
    implementada en los módulos subyacentes.
    """
    return (
        "Fuentes de datos disponibles — TRES fuentes heterogéneas, ninguna "
        "sustituye a las otras:\n\n"
        "1. query_local_reviews — corpus LOCAL de reseñas de usuarios de la "
        "aplicación de salud digital (percepción ciudadana). El sentimiento "
        "se deriva determinísticamente de la calificación: 1-2 estrellas=NEG, "
        "3=NEU y 4-5=POS. NO representa datos institucionales ni indicadores "
        "agregados a nivel país. top_negative_keywords es evidencia léxica "
        "de frecuencia y NO debe interpretarse como una clasificación semántica "
        "de problemas. problem_categories se obtiene mediante asignaciones "
        "BERTopic precomputadas y una taxonomía conceptual humana congelada; "
        "BERTopic no se ejecuta durante la consulta MCP. main_problem es la "
        "categoría sustantiva con mayor número de reseñas negativas asignadas "
        "en el subconjunto solicitado. Se excluyen del ranking los tópicos de "
        "ruido y los codificados como insatisfacción genérica, discordancia "
        "rating-texto o mixtos/no clasificables. pct_of_substantive_problems "
        "usa como denominador únicamente las reseñas negativas incluidas en "
        "categorías sustantivas; pct_of_negative_reviews usa todas las reseñas "
        "negativas del subconjunto; substantive_coverage_pct indica qué "
        "proporción de todas las negativas quedó representada por categorías "
        "sustantivas. topic_coverage reporta además asignaciones faltantes, "
        "reseñas no-ruido, ruido, reseñas asignadas a tópicos excluidos y "
        "problemas sustantivos.\n\n"
        "2. get_worldbank_indicator — datos INSTITUCIONALES en vivo vía "
        "REST del World Bank (World Development Indicators / Global Health "
        "Expenditure Database). Cobertura amplia de países, actualización "
        "periódica (no diaria). Un valor 'value': null para un año significa "
        "que ese año específico no tiene dato reportado, NO que el país "
        "carezca del indicador por completo — revisar otros años antes de "
        "concluir ausencia total.\n\n"
        "3. get_oecd_indicator — datos INSTITUCIONALES en vivo vía SDMX de "
        "la OCDE (System of Health Accounts). Cobertura limitada a países "
        "miembros/socios de la OCDE. IMPORTANTE: World Bank y OCDE pueden "
        "reportar el MISMO indicador conceptual con valores ligeramente "
        "distintos (metodologías de recolección diferentes) — si ambas "
        "fuentes se consultan para la misma pregunta y los valores no "
        "coinciden exactamente, esto es esperado, no un error; repórtelo "
        "explícitamente en la respuesta en vez de elegir un valor y "
        "descartar el otro en silencio."
    )


@mcp.tool(name="get_worldbank_indicator")
def get_worldbank_indicator_handler(
    country_iso3: str,
    indicator: str,
    most_recent_n: int | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
) -> dict:
    """
    Query a health indicator for a country from the World Bank REST API.
    Use this when the question explicitly asks for World Bank data.

    Args:
        country_iso3: ISO3 country code, e.g. 'CRI' for Costa Rica.
        indicator: One of current_health_expenditure_pct_gdp,
            current_health_expenditure_per_capita_usd,
            domestic_govt_health_expenditure_pct_gdp,
            domestic_govt_health_expenditure_pct_govt_expenditure,
            out_of_pocket_expenditure_pct_che, external_health_expenditure_pct_che.
        most_recent_n: if set, return only the N most recent non-empty observations.
        start_year: optional start year for a range query.
        end_year: optional end year for a range query.
    """
    try:
        return get_worldbank_indicator(
            country_iso3=country_iso3,
            indicator=indicator,
            start_year=start_year,
            end_year=end_year,
            most_recent_n=most_recent_n,
        )
    except WorldBankToolError as exc:
        # Re-raise as ValueError: the MCP SDK converts handler exceptions
        # into a tool result with isError=True, visible to the LLM client —
        # this lets the model observe the failure and react to it, instead
        # of the server process crashing.
        raise ValueError(str(exc)) from exc


@mcp.tool(name="get_oecd_indicator")
def get_oecd_indicator_handler(
    country_iso3: str,
    indicator: str,
    start_period: str | None = None,
) -> dict:
    """
    Query a health indicator for a country from the OECD SDMX API.
    Use this when the question explicitly asks for OECD data.

    Args:
        country_iso3: ISO3 country code, e.g. 'CRI' for Costa Rica.
        indicator: currently only 'health_expenditure_pct_gdp'.
        start_period: optional start year, e.g. '2015'.
    """
    try:
        return get_oecd_indicator(
            country_iso3=country_iso3,
            indicator=indicator,
            start_period=start_period,
        )
    except OECDToolError as exc:
        raise ValueError(str(exc)) from exc


@mcp.tool(name="query_local_reviews")
def query_local_reviews_handler(
    start_date: str,
    end_date: str,
    query: str | None = None,
) -> dict:
    """
    Search and analyze the local corpus of user reviews of the digital
    health application for a requested period.

    Use this tool for questions about review sentiment, frequent negative
    terms, reported problems, or the principal problem reported by users.

    Sentiment is derived deterministically from the review star rating:
    1-2 stars = NEG, 3 stars = NEU, and 4-5 stars = POS.

    The field ``top_negative_keywords`` is lexical frequency evidence and
    must not be interpreted as a semantic problem classification.

    Problem categories are obtained from frozen offline BERTopic topic
    assignments followed by a frozen human-coded conceptual taxonomy.
    BERTopic is not trained or executed during MCP queries.

    ``main_problem`` is the substantive conceptual problem category with
    the largest number of assigned negative reviews in the requested
    subset. Noise topics and topics coded as generic dissatisfaction,
    rating-text discordance, or mixed/non-classifiable are excluded from
    that ranking.

    ``pct_of_substantive_problems`` uses only negative reviews assigned to
    included substantive problem categories as its denominator.
    ``pct_of_negative_reviews`` uses all negative reviews in the requested
    subset as its denominator. ``substantive_coverage_pct`` reports the
    proportion of all negative reviews represented by included substantive
    problem categories.

    Args:
        start_date: ISO date, e.g. '2023-01-01'.
        end_date: ISO date, e.g. '2023-12-31'; the complete end day is included.
        query: optional keyword filter (substring match in review content).
    """
    try:
        return query_local_reviews(start_date=start_date, end_date=end_date, query=query)
    except LocalReviewsToolError as exc:
        raise ValueError(str(exc)) from exc


if __name__ == "__main__":
    mcp.run(transport="stdio")
