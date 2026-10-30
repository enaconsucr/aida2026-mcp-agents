"""
World Bank health-indicator retrieval tool.

English
-------
Purpose
    This module retrieves country-level health indicators from the World Bank
    REST API v2 and returns a normalized representation for use by the
    MCP-based research agent.

Expected API structure
    The API is expected to return ``[metadata, observations]``. The first
    element contains pagination metadata and the second contains observation
    records. Missing values may legitimately be returned as ``null`` and are
    preserved as ``None`` rather than imputed.

Reproducibility and provenance
    The exact query URL and a local archival copy of the raw API response are
    preserved for each retrieval. These artifacts support auditability of the
    external evidence used in an experimental run, but do not imply that the
    upstream World Bank database is immutable.

Indicator control
    ``HEALTH_INDICATORS`` defines a curated set of official World Bank
    indicator codes. During exploratory development, direct official indicator
    codes may also be accepted. For a frozen benchmark, the admissible
    indicator universe should remain fixed across all trials.

Official API documentation
    https://datahelpdesk.worldbank.org/knowledgebase/articles/898581

Español
-------
Propósito
    Este módulo recupera indicadores de salud por país desde la API REST v2
    del Banco Mundial y devuelve una representación normalizada para su uso
    por el agente de investigación basado en MCP.

Estructura esperada de la API
    Se espera que la API devuelva ``[metadata, observations]``. El primer
    elemento contiene metadatos de paginación y el segundo contiene los
    registros de observaciones. Los valores faltantes pueden aparecer
    legítimamente como ``null`` y se conservan como ``None``, sin imputación.

Reproducibilidad y procedencia
    Para cada recuperación se preservan la URL exacta de consulta y una copia
    local de la respuesta cruda de la API. Estos artefactos permiten auditar
    la evidencia externa utilizada en una ejecución experimental, pero no
    implican que la base de datos del Banco Mundial sea inmutable.

Control de indicadores
    ``HEALTH_INDICATORS`` define un conjunto curado de códigos oficiales del
    Banco Mundial. Durante el desarrollo exploratorio también pueden aceptarse
    códigos oficiales directos. Para un benchmark congelado, el universo de
    indicadores admisibles debe permanecer fijo durante todos los ensayos.
"""


import requests
import json
import os
from datetime import datetime, timezone

WB_BASE_URL = "https://api.worldbank.org/v2"

# Raw API responses are archived as provenance artifacts for each retrieval.
# The archive supports auditability without assuming that the upstream
# World Bank service or its data remain unchanged.
ARCHIVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_responses", "worldbank")

# Curated health-indicator mapping using official World Bank indicator codes.
# For a frozen benchmark, this mapping should remain unchanged across trials.
HEALTH_INDICATORS = {
    "current_health_expenditure_pct_gdp": "SH.XPD.CHEX.GD.ZS",
    "current_health_expenditure_per_capita_usd": "SH.XPD.CHEX.PC.CD",
    "domestic_govt_health_expenditure_pct_gdp": "SH.XPD.GHED.GD.ZS",
    "domestic_govt_health_expenditure_pct_govt_expenditure": "SH.XPD.GHED.GE.ZS",
    "out_of_pocket_expenditure_pct_che": "SH.XPD.OOPC.CH.ZS",
    "external_health_expenditure_pct_che": "SH.XPD.EHEX.CH.ZS",
}

# Restricting indicator access to this whitelist controls the information
# space available to the agent during benchmark evaluation. Exploratory runs
# may allow direct official World Bank indicator codes, whereas a frozen
# benchmark should enable this restriction before experimental trials begin.
RESTRICT_TO_WHITELIST = False


class WorldBankToolError(RuntimeError):
    """
    Controlled exception for World Bank retrieval or parsing failures.

    English
    -------
    This exception distinguishes network, HTTP, and response-structure failures
    from legitimate missing observations. An upstream ``null`` value is valid
    data and therefore does not constitute a tool failure.

    Español
    -------
    Excepción controlada para fallos de recuperación o procesamiento de datos
    del Banco Mundial. Permite distinguir errores de red, HTTP o estructura de
    respuesta de observaciones legítimamente ausentes. Un valor ``null`` del
    servicio externo es un dato válido y no constituye un fallo de la
    herramienta.
    """


def _archive_raw_response(query_url: str, raw_text: str, country_iso3: str, indicator_code: str) -> str:
    """
    Archive the unmodified upstream response as a provenance artifact.

    English
    -------
    The raw response, exact query URL, retrieval timestamp, country code, and
    indicator code are stored locally to support auditing of the external
    evidence used in a specific experimental run. Archival failure does not
    interrupt retrieval because provenance preservation is an additional
    safeguard rather than a prerequisite for returning available data.

    Returns
    -------
    str
        Path to the archived record, or an empty string if archival fails.

    Español
    -------
    Archiva la respuesta sin modificar del servicio externo como artefacto de
    procedencia. Se almacenan la respuesta cruda, la URL exacta de consulta, el
    momento de recuperación, el código de país y el código del indicador para
    facilitar la auditoría de la evidencia utilizada en una ejecución
    experimental específica.

    Un fallo de archivado no interrumpe la recuperación de datos, porque esta
    salvaguarda de procedencia es adicional y no un requisito para devolver
    información disponible.

    Retorna
    -------
    str
        Ruta del registro archivado, o una cadena vacía si el archivado falla.
    """
    try:
        os.makedirs(ARCHIVE_DIR, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        filename = f"{timestamp}_{country_iso3}_{indicator_code}.json"
        filepath = os.path.join(ARCHIVE_DIR, filename)
        record = {
            "query_url": query_url,
            "retrieved_at_utc": timestamp,
            "country_iso3": country_iso3,
            "indicator_code": indicator_code,
            "raw_response": raw_text,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        return filepath
    except OSError:
        return ""


def get_worldbank_indicator(
    country_iso3: str,
    indicator: str,
    start_year: int | None = None,
    end_year: int | None = None,
    most_recent_n: int | None = None,
) -> dict:
    """
    Retrieve a World Bank indicator for a country.

    English
    -------
    Parameters
    ----------
    country_iso3 : str
        ISO 3166-1 alpha-3 country code, for example ``"CRI"``.
    indicator : str
        Either a curated key from ``HEALTH_INDICATORS`` or, when whitelist
        restriction is disabled, an official World Bank indicator code.
    start_year, end_year : int, optional
        Optional year bounds. If both are provided, the query requests the
        corresponding inclusive date range. If only ``start_year`` is
        provided, that single year is requested.
    most_recent_n : int, optional
        Number of most recent non-empty observations requested through the
        World Bank ``mrnev`` parameter. When supplied, this option takes
        precedence over year-range parameters.

    Returns
    -------
    dict
        Normalized result containing the data source, resolved indicator code,
        indicator name, ISO3 country code (``country_iso3``), chronologically
        ordered observations, exact query URL, and path to the archived raw
        response.

    Notes
    -----
    Upstream missing values are preserved rather than imputed. Observations are
    sorted in ascending chronological order because the World Bank API normally
    returns them in descending order.

    The exact request URL and raw upstream response are retained as provenance
    artifacts for auditing a specific experimental run.

    Raises
    ------
    WorldBankToolError
        Raised when the HTTP request fails, the response is not valid JSON, the
        payload does not conform to the expected World Bank response structure,
        or no observation block is returned for the requested country and
        indicator.

    Español
    -------
    Recupera un indicador del Banco Mundial para un país.

    Parámetros
    ----------
    country_iso3 : str
        Código ISO 3166-1 alfa-3 del país, por ejemplo ``"CRI"``.
    indicator : str
        Clave curada de ``HEALTH_INDICATORS`` o, cuando la restricción de la
        whitelist está desactivada, un código oficial de indicador del Banco
        Mundial.
    start_year, end_year : int, opcional
        Límites opcionales de años. Si se proporcionan ambos, la consulta
        solicita el rango inclusivo correspondiente. Si solo se proporciona
        ``start_year``, se solicita ese año.
    most_recent_n : int, opcional
        Número de observaciones no vacías más recientes solicitado mediante el
        parámetro ``mrnev`` del Banco Mundial. Cuando se especifica, esta opción
        tiene precedencia sobre los parámetros de rango temporal.

    Retorna
    -------
    dict
        Resultado normalizado con la fuente de datos, código del indicador
        resuelto, nombre del indicador, código ISO3 del país
        (``country_iso3``), observaciones ordenadas cronológicamente, URL
        exacta de consulta y ruta de la respuesta cruda archivada.

    Notas
    -----
    Los valores faltantes del servicio externo se conservan sin imputación. Las
    observaciones se ordenan cronológicamente de forma ascendente porque la API
    del Banco Mundial normalmente las devuelve en orden descendente.

    La URL exacta de consulta y la respuesta cruda se preservan como artefactos
    de procedencia para facilitar la auditoría de una ejecución experimental
    específica.

    Excepciones
    -----------
    WorldBankToolError
        Se genera cuando falla la solicitud HTTP, la respuesta no es JSON
        válido, el payload no cumple la estructura esperada o no se devuelve
        un bloque de observaciones para el país y el indicador solicitados.
    """

    indicator_code = HEALTH_INDICATORS.get(indicator, indicator)

    if RESTRICT_TO_WHITELIST and indicator_code not in HEALTH_INDICATORS.values():
        raise WorldBankToolError(
            f"Indicator '{indicator}' (resolved code: '{indicator_code}') is not in "
            f"the frozen benchmark whitelist. Allowed codes: "
            f"{list(HEALTH_INDICATORS.values())}. If a new indicator is required, "
            f"add it to HEALTH_INDICATORS before benchmark questions are generated, "
            f"never during evaluation."
        )

    params = {"format": "json", "per_page": "1000"}
    if most_recent_n is not None:
        params["mrnev"] = str(most_recent_n)
    elif start_year is not None and end_year is not None:
        params["date"] = f"{start_year}:{end_year}"
    elif start_year is not None:
        params["date"] = str(start_year)

    url = f"{WB_BASE_URL}/country/{country_iso3}/indicator/{indicator_code}"

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise WorldBankToolError(
            f"Network failure while querying the World Bank API: {exc}"
        ) from exc

    # Archive the raw upstream response before parsing so that the exact
    # external evidence remains available even if subsequent parsing fails.
    archive_path = _archive_raw_response(resp.url, resp.text, country_iso3, indicator_code)

    try:
        payload = resp.json()
    except ValueError as exc:
        raise WorldBankToolError(
            f"The World Bank response is not valid JSON: {resp.text[:200]}"
        ) from exc

    # The World Bank API may return an error object rather than the expected
    # two-element list when a country or indicator code is invalid. Fail fast
    # instead of propagating a structurally invalid payload downstream.
    if not isinstance(payload, list) or len(payload) != 2:
        raise WorldBankToolError(
            f"Unexpected World Bank response structure (possibly an invalid "
            f"indicator or country code: '{indicator_code}' / '{country_iso3}'). "
            f"Raw payload: {payload}"
        )

    metadata, observations = payload
    if observations is None:
        raise WorldBankToolError(
            f"No observations were returned for indicator='{indicator_code}' "
            f"and country='{country_iso3}'. Verify that the indicator code and "
            f"ISO3 country code are valid."
        )

    obs_list = [
        {"year": obs["date"], "value": obs["value"]}
        for obs in observations
    ]
    # Normalize observations to ascending chronological order.
    obs_list.sort(key=lambda o: o["year"])

    indicator_name = observations[0]["indicator"]["value"] if observations else indicator_code

    return {
        "source": "World Bank",
        "indicator_code": indicator_code,
        "indicator_name": indicator_name,
        "country_iso3": country_iso3,
        "observations": obs_list,
        "query_url": resp.url,
        "raw_response_archived_at": (
            os.path.relpath(archive_path, start=os.path.dirname(__file__))
            if archive_path else ""
        ),  # Repository-relative provenance reference; empty if archival fails.
    }


if __name__ == "__main__":
    # Minimal manual validation requiring outbound internet access.
    import json
    result = get_worldbank_indicator("CRI", "current_health_expenditure_pct_gdp", most_recent_n=5)
    print(json.dumps(result, indent=2, ensure_ascii=False))
