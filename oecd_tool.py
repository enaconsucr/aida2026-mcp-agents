"""
OECD Health Indicator Retrieval Tool
====================================

ENGLISH
-------

Purpose
~~~~~~~
This module implements the OECD health-data retrieval component of the
MCP-based research agent. It queries the OECD SDMX REST API without requiring
authentication and returns structured observations together with provenance
information designed to support experimental traceability, auditability, and
reproducibility.

SDMX data model
~~~~~~~~~~~~~~~
Unlike conventional REST APIs, in which variables are commonly specified
through named query parameters, SDMX identifies observations through two
interdependent elements:

1. a *dataflow*, which identifies the statistical dataset; and
2. a dimension-ordered *series key*, which identifies a particular statistical
   series within that dataset.

The position of each value in the series key is defined by the Data Structure
Definition (DSD) associated with the dataflow. Consequently, the dimensional
order is semantically significant: changing the position of a value may refer
to a different statistical concept or produce an invalid query.

The health-expenditure indicator implemented in this module uses the OECD
System of Health Accounts dataflow and requests health expenditure as a
percentage of gross domestic product. The series key fixes the reference area,
annual frequency, expenditure measure, unit of measure, and the aggregate
values required for the remaining dimensions.

Dataflow version resolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The dataflow version is intentionally left unspecified in the operational
query so that the OECD API resolves the currently active version. Validation
against the live endpoint showed that explicitly requesting an older version
returned no matching series, whereas version-neutral resolution returned the
expected observations.

This design introduces a potential reproducibility risk because a future OECD
dataflow release could alter the structure or contents returned by the same
logical request. The implementation mitigates this risk by archiving the raw
API response for each retrieval and by exposing response metadata in the tool
output. These artifacts provide an auditable record of the data supplied by
the OECD API at the time of each experimental run.

Reproducibility safeguards
~~~~~~~~~~~~~~~~~~~~~~~~~~
The implementation follows a defensive, reproducibility-oriented design:

* SDMX requests are constructed deterministically from an explicit whitelist
  of supported indicators.
* Raw external responses are archived before analytical interpretation.
* Response metadata and the exact series key are retained for provenance.
* Empty observation sets are treated as legitimate data-availability outcomes
  rather than automatically interpreted as software failures.
* Responses containing multiple matching series are rejected instead of
  silently selecting an arbitrary series.
* The DSD can be retrieved independently to verify the dimensional structure
  used to construct a series key.

These safeguards are intended to reduce the possibility that upstream changes,
ambiguous SDMX structures, or undocumented assumptions silently affect the
experimental results.


ESPAÑOL
-------

Propósito
~~~~~~~~~
Este módulo implementa el componente de recuperación de datos de salud de la
OCDE utilizado por el agente de investigación basado en MCP. Consulta la API
REST SDMX de la OCDE sin requerir autenticación y devuelve observaciones
estructuradas junto con información de procedencia orientada a garantizar la
trazabilidad, auditabilidad y reproducibilidad del experimento.

Modelo de datos SDMX
~~~~~~~~~~~~~~~~~~~~
A diferencia de las API REST convencionales, en las que las variables suelen
especificarse mediante parámetros de consulta con nombre, SDMX identifica las
observaciones mediante dos elementos interdependientes:

1. un *flujo de datos* (`dataflow`), que identifica el conjunto estadístico; y
2. una *clave de serie* (`series key`) con dimensiones ordenadas, que identifica
   una serie estadística específica dentro de dicho conjunto.

La posición de cada valor en la clave de serie está definida por la
Definición de Estructura de Datos (`Data Structure Definition`, DSD)
correspondiente al flujo de datos. Por ello, el orden de las dimensiones posee
significado semántico: modificar la posición de un valor puede hacer referencia
a otro concepto estadístico o producir una consulta inválida.

El indicador de gasto en salud implementado en este módulo utiliza el flujo de
datos del Sistema de Cuentas de Salud de la OCDE y recupera el gasto en salud
como porcentaje del producto interno bruto. La clave de serie fija el área de
referencia, la frecuencia anual, la medida de gasto, la unidad de medida y los
valores agregados necesarios para las dimensiones restantes.

Resolución de la versión del flujo de datos
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
La consulta operativa no fija explícitamente una versión del flujo de datos,
de modo que la API de la OCDE resuelva la versión actualmente activa. Las
pruebas realizadas contra el endpoint operativo mostraron que solicitar
explícitamente una versión anterior no devolvía series coincidentes, mientras
que la resolución sin versión explícita devolvía las observaciones esperadas.

Esta decisión introduce un posible riesgo para la reproducibilidad, ya que una
futura versión publicada por la OCDE podría modificar la estructura o el
contenido asociado con la misma consulta lógica. La implementación mitiga este
riesgo mediante el archivado de la respuesta original de la API para cada
consulta y la incorporación de metadatos de respuesta en la salida de la
herramienta. Estos artefactos permiten auditar qué información proporcionó la
API de la OCDE en cada ejecución experimental.

Salvaguardas de reproducibilidad
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
La implementación adopta un diseño defensivo orientado a la reproducibilidad:

* Las consultas SDMX se construyen de manera determinista a partir de una lista
  explícita de indicadores admitidos.
* Las respuestas externas originales se archivan antes de su interpretación
  analítica.
* Los metadatos de respuesta y la clave exacta de la serie se conservan como
  información de procedencia.
* La ausencia de observaciones se considera un resultado legítimo de
  disponibilidad de datos y no necesariamente un fallo del software.
* Las respuestas que contienen múltiples series coincidentes se rechazan en
  lugar de seleccionar silenciosamente una serie arbitraria.
* La DSD puede recuperarse de manera independiente para verificar la estructura
  dimensional utilizada al construir una clave de serie.

Estas salvaguardas buscan reducir la posibilidad de que cambios en las fuentes
externas, estructuras SDMX ambiguas o supuestos no documentados modifiquen
silenciosamente los resultados experimentales.
"""

import requests
import json
import os
from datetime import datetime, timezone
import xml.etree.ElementTree as ET

OECD_BASE_URL = "https://sdmx.oecd.org/public/rest"

# Raw OECD responses are archived as provenance artifacts for each query.
# This is particularly important for externally maintained SDMX services,
# whose dataflows and dimensional structures may evolve over time. Retaining
# the original payload makes it possible to verify exactly what the upstream
# service returned during a given experimental run.
ARCHIVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "raw_responses", "oecd")

HEALTH_DATAFLOWS = {
    "health_expenditure_pct_gdp": {
        # The operational dataflow deliberately omits a fixed version so that
        # the OECD API resolves the currently active release. This behavior was
        # validated empirically because an explicitly requested older version
        # returned no matching series, whereas version-neutral resolution
        # returned the expected observations.
        #
        # Reproducibility is preserved through two complementary mechanisms:
        # raw responses are archived for every query, and response metadata is
        # propagated to the tool output so that the external evidence used in
        # each experimental run remains auditable.
        "dataflow": "OECD.ELS.HD,DSD_SHA@DF_SHA,",
        "key_template": "{country}.A.EXP_HEALTH.PT_B1GQ._T._Z._T._T._T._Z._Z._Z",
    },
}

# Benchmark runs should use a frozen whitelist configuration so that the
# space of admissible indicators remains constant across experimental trials.
RESTRICT_TO_WHITELIST = False


class OECDToolError(RuntimeError):
    """
    Controlled exception for OECD retrieval and parsing failures.

    ENGLISH
    This exception distinguishes operational failures—such as network errors,
    malformed SDMX payloads, or ambiguous responses—from legitimate cases in
    which the requested combination of dimensions has no available
    observations.

    ESPAÑOL
    Esta excepción distingue los fallos operativos —como errores de red,
    respuestas SDMX mal formadas o resultados ambiguos— de los casos legítimos
    en los que la combinación solicitada de dimensiones no dispone de
    observaciones.
    """


def _archive_raw_response(query_url: str, raw_text: str, country_iso3: str, dataflow: str, key: str) -> str:
    """
    Archive the original OECD response as a provenance artifact.

    ENGLISH
    The unmodified payload is stored with a timestamp and query metadata before
    subsequent analytical interpretation. This provides an immutable record of
    the evidence supplied by the external API and supports retrospective
    verification if OECD dataflows, dimensional ordering, or response formats
    change.

    ESPAÑOL
    La respuesta original se almacena con una marca temporal y metadatos de la
    consulta antes de cualquier interpretación analítica posterior. Esto
    proporciona un registro inalterado de la evidencia suministrada por la API
    externa y permite verificar retrospectivamente los resultados si la OCDE
    modifica sus flujos de datos, el orden de las dimensiones o el formato de
    respuesta.

    """
    try:
        os.makedirs(ARCHIVE_DIR, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        safe_dataflow = dataflow.replace(",", "_").replace("@", "_")
        filename = f"{timestamp}_{country_iso3}_{safe_dataflow}.json"
        filepath = os.path.join(ARCHIVE_DIR, filename)
        record = {
            "query_url": query_url,
            "retrieved_at_utc": timestamp,
            "country_iso3": country_iso3,
            "dataflow": dataflow,
            "key": key,
            "raw_response": raw_text,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        return filepath
    except OSError:
        return ""


def fetch_dataflow_structure(dataflow_id: str) -> str:
    """
    Retrieve the SDMX structural definition associated with a dataflow.

    ENGLISH
    This helper retrieves the XML representation of the SDMX structure linked
    to a dataflow. Its primary purpose is methodological verification: the
    returned Data Structure Definition (DSD) can be inspected to confirm the
    official dimension names, their positional order, and the structural
    assumptions used when constructing a compound series key.

    Because SDMX series keys are positional rather than self-describing,
    validating the DSD is an important reproducibility safeguard whenever a
    new dataflow is introduced or when an upstream structural change is
    suspected. The function intentionally returns the original XML rather than
    reducing it to a locally inferred schema, thereby preserving the
    authoritative representation supplied by the OECD service.

    ESPAÑOL
    Esta función recupera la representación XML de la estructura SDMX asociada
    con un flujo de datos. Su propósito principal es la verificación
    metodológica: la Definición de Estructura de Datos (DSD) obtenida permite
    comprobar los nombres oficiales de las dimensiones, su orden posicional y
    los supuestos estructurales utilizados al construir una clave compuesta de
    serie.

    Debido a que las claves de serie SDMX son posicionales y no
    autodescriptivas, la verificación de la DSD constituye una salvaguarda
    importante para la reproducibilidad cuando se incorpora un nuevo flujo de
    datos o cuando se sospecha un cambio estructural en la fuente externa. La
    función devuelve deliberadamente el XML original, en lugar de reducirlo a
    un esquema inferido localmente, con el fin de conservar la representación
    autoritativa proporcionada por el servicio de la OCDE.
    """
    agency_id, flow_ref = dataflow_id.split(",", 1)
    url = f"{OECD_BASE_URL}/dataflow/{agency_id}/{flow_ref.split(',')[0]}"
    try:
        resp = requests.get(url, params={"references": "all"}, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise OECDToolError(
            f"Network failure while retrieving the SDMX structure: {exc}"
        ) from exc
    return resp.text


def get_oecd_indicator(
    country_iso3: str,
    indicator: str,
    start_period: str | None = None,
) -> dict:
    """
    Retrieve a whitelisted OECD health indicator for a specified country.

    ENGLISH

    Parameters
    ----------
    country_iso3:
        Three-letter ISO country code used to instantiate the reference-area
        dimension of the SDMX series key.

    indicator:
        Logical indicator identifier registered in ``HEALTH_DATAFLOWS``.
        Only explicitly configured indicators are accepted. This restriction
        keeps the admissible data space stable and auditable across
        experimental runs.

    start_period:
        Optional initial year. When omitted, the API is allowed to return the
        complete available time series for the selected indicator and country.

    Returns
    -------
    dict
        A structured result containing the source identifier, resolved
        dataflow, exact SDMX series key, country code, observations, request
        URL, archival information, and response metadata used for provenance
        and reproducibility.

        ``observations`` is represented as a list of records containing a
        period and a numeric value. An empty observation list is treated as a
        legitimate data-availability outcome rather than as an automatic
        software failure.

    Raises
    ------
    OECDToolError
        Raised when the requested indicator is not registered, when the
        external request fails, when the returned SDMX-JSON structure cannot
        be interpreted safely, or when the response is structurally ambiguous.

    ESPAÑOL

    Parámetros
    ----------
    country_iso3:
        Código ISO de tres letras utilizado para completar la dimensión de
        área de referencia de la clave de serie SDMX.

    indicator:
        Identificador lógico registrado en ``HEALTH_DATAFLOWS``. Solo se
        admiten indicadores configurados explícitamente. Esta restricción
        mantiene estable y auditable el espacio de datos permitido entre
        distintas ejecuciones experimentales.

    start_period:
        Año inicial opcional. Si se omite, la API puede devolver la serie
        temporal completa disponible para el indicador y país seleccionados.

    Retorno
    -------
    dict
        Resultado estructurado que incluye la fuente, el flujo de datos
        resuelto, la clave SDMX exacta, el código de país, las observaciones,
        la URL consultada, la información de archivo y los metadatos de
        respuesta necesarios para la procedencia y reproducibilidad.

        ``observations`` se representa como una lista de registros con periodo
        y valor numérico. Una lista vacía se interpreta como un resultado
        legítimo de disponibilidad de datos y no como un fallo automático del
        software.

    Excepciones
    -----------
    OECDToolError
        Se genera cuando el indicador solicitado no está registrado, cuando
        falla la consulta externa, cuando la estructura SDMX-JSON no puede
        interpretarse con seguridad o cuando la respuesta presenta
        ambigüedad estructural.
    """
    if indicator not in HEALTH_DATAFLOWS:
        raise OECDToolError(
            f"Indicator '{indicator}' is not registered in HEALTH_DATAFLOWS. "
            f"Available options: {list(HEALTH_DATAFLOWS.keys())}. Unlike the "
            f"World Bank tool, this module does not allow arbitrary dataflow passthrough — "
            f"the whitelist is always strict, whether frozen or not "
            f"(RESTRICT_TO_WHITELIST is retained only for documentary symmetry "
            f"with worldbank_tool.py)."
        )

    config = HEALTH_DATAFLOWS[indicator]
    dataflow = config["dataflow"]
    key = config["key_template"].format(country=country_iso3)

    params = {}
    if start_period:
        params["startPeriod"] = start_period

    url = f"{OECD_BASE_URL}/data/{dataflow}/{key}"

    # SDMX-JSON 2.0 is requested explicitly through the Accept header.
    # This configuration was validated end-to-end against the OECD API,
    # confirming an HTTP 200 response, the expected Content-Type, and a
    # response structure compatible with the parser implemented below.
    headers = {"Accept": "application/vnd.sdmx.data+json;version=2.0"}

    try:
        resp = requests.get(url, params=params, headers=headers, timeout=20)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise OECDToolError(
            f"Network failure while querying the OECD SDMX service: {exc}"
        ) from exc

    # Archive the raw payload before parsing. Preserving the original external
    # response ensures that any later parsing failure or upstream schema change
    # can be investigated against the exact evidence returned by the OECD API.
    archive_path = _archive_raw_response(resp.url, resp.text, country_iso3, dataflow, key)

    try:
        payload = resp.json()
    except ValueError as exc:
        raise OECDToolError(f"Respuesta no es JSON válido: {resp.text[:200]}") from exc

    observations = _parse_sdmx_json(payload)

    return {
        "source": "OECD",
        "dataflow": dataflow,
        "key": key,
        "country_iso3": country_iso3,
        "observations": observations,
        "query_url": resp.url,
        "raw_response_archived_at": (
            os.path.relpath(archive_path, start=os.path.dirname(__file__))
            if archive_path else ""
        ),  # Repository-relative provenance reference; empty if archival fails.
        # Preserve upstream response metadata as part of the provenance record.
        # These fields may contain preparation timestamps or response identifiers
        # useful for auditing a specific experimental run. No assumption is made
        # that this block exposes the resolved OECD dataflow version explicitly.
        "response_meta": payload.get("meta"),
    }


def _parse_sdmx_json(payload: dict) -> list[dict]:
    """
    Parse an SDMX-JSON response into period-value observations.

    ENGLISH

    SDMX-JSON 2.0 representation
    ----------------------------
    In SDMX-JSON 2.0, a series identifier is a positional compound key whose
    components are separated by colons, for example::

        31:0:0:5:0:0:0:0:0:0:0:2

    Each component is an integer index corresponding to one series-level
    dimension declared in ``data.structures[i].dimensions.series``. The
    components must therefore be interpreted in the same positional order as
    those dimensions. Each index is resolved against the ``values`` array of
    its corresponding dimension.

    The compound key must not be interpreted as a self-describing identifier,
    nor is it safe to assume that the first series returned by the API is the
    analytically relevant one.

    Observation-level indexing
    --------------------------
    Observations within a series are indexed separately against the dimensions
    declared under ``dimensions.observation``. For the OECD health data used
    by this project, the relevant observation dimension is ``TIME_PERIOD``.
    The parser explicitly locates that dimension rather than assuming a fixed
    position.

    Defensive parsing and ambiguity control
    ---------------------------------------
    The retrieval layer constructs a deliberately restricted SDMX key for a
    specific country and indicator. Under this design, a successful request is
    expected to resolve to exactly one statistical series.

    The parser therefore applies the following rules:

    * zero returned series is interpreted as legitimate absence of data;
    * one returned series is parsed normally; and
    * more than one returned series is treated as structural ambiguity and
      raises ``OECDToolError``.

    Rejecting an unexpected multi-series response is a deliberate fail-fast
    safeguard. Silently selecting the first available series could associate
    observations with an unintended country, unit, measure, or other SDMX
    dimension and would introduce a substantially more serious reproducibility
    risk than an explicit exception.

    This parsing strategy was validated end-to-end against the live OECD SDMX
    service using the configured health-expenditure query. The validation
    confirmed compatibility with the SDMX-JSON 2.0 response structure used by
    this implementation.


    ESPAÑOL

    Representación SDMX-JSON 2.0
    ----------------------------
    En SDMX-JSON 2.0, el identificador de una serie es una clave compuesta
    posicional cuyos componentes están separados por dos puntos, por ejemplo::

        31:0:0:5:0:0:0:0:0:0:0:2

    Cada componente es un índice entero correspondiente a una dimensión de
    nivel de serie declarada en
    ``data.structures[i].dimensions.series``. Por tanto, los componentes deben
    interpretarse respetando exactamente el orden posicional de dichas
    dimensiones. Cada índice se resuelve contra el arreglo ``values`` de la
    dimensión correspondiente.

    La clave compuesta no debe interpretarse como un identificador
    autodescriptivo, ni es metodológicamente seguro asumir que la primera serie
    devuelta por la API es la serie analíticamente pertinente.

    Indexación de las observaciones
    -------------------------------
    Las observaciones contenidas dentro de una serie se indexan de manera
    independiente contra las dimensiones declaradas en
    ``dimensions.observation``. Para los datos de salud de la OCDE utilizados
    en este proyecto, la dimensión de observación relevante es
    ``TIME_PERIOD``. El parser localiza explícitamente esta dimensión en lugar
    de asumir una posición fija.

    Procesamiento defensivo y control de ambigüedad
    ------------------------------------------------
    La capa de recuperación construye una clave SDMX deliberadamente
    restringida para un país y un indicador específicos. Bajo este diseño, se
    espera que una consulta exitosa corresponda exactamente a una serie
    estadística.

    Por ello, el parser aplica las siguientes reglas:

    * cero series devueltas se interpreta como ausencia legítima de datos;
    * una serie se procesa normalmente; y
    * más de una serie se considera una ambigüedad estructural y genera
      ``OECDToolError``.

    Rechazar una respuesta inesperada con múltiples series constituye una
    salvaguarda deliberada de fallo temprano (*fail-fast*). Seleccionar
    silenciosamente la primera serie disponible podría asociar las
    observaciones con un país, unidad, medida u otra dimensión SDMX distinta
    de la solicitada, lo que introduciría un riesgo para la reproducibilidad
    considerablemente mayor que una excepción explícita.

    Esta estrategia de procesamiento fue validada de extremo a extremo contra
    el servicio SDMX operativo de la OCDE mediante la consulta configurada de
    gasto en salud. La validación confirmó la compatibilidad con la estructura
    de respuesta SDMX-JSON 2.0 utilizada por esta implementación.
    """
    try:
        dataset = payload["data"]["dataSets"][0]
    except (KeyError, IndexError) as exc:
        raise OECDToolError(
            f"data.dataSets[0] was not found in the payload — unexpected SDMX-JSON "
            f"structure even at the top level. Error: {exc}"
        ) from exc

    try:
        if "structures" in payload["data"]:
            structure_index = dataset.get("structure", 0)
            structure = payload["data"]["structures"][structure_index]
        elif "structure" in payload["data"]:
            structure = payload["data"]["structure"]
        else:
            raise OECDToolError(
                "Neither 'structures' (2.0) nor 'structure' (1.0) was found in "
                "data — unrecognized SDMX-JSON version."
            )

        obs_dims = structure["dimensions"]["observation"]
        time_dim_position = next(
            i for i, dim in enumerate(obs_dims) if dim["id"] == "TIME_PERIOD"
        )
        time_values = obs_dims[time_dim_position]["values"]

        series_dims = structure["dimensions"].get("series", [])
    except (KeyError, IndexError, StopIteration) as exc:
        raise OECDToolError(
            f"The SDMX-JSON version was recognized, but "
            f"'dimensions.observation'/'dimensions.series' does not have the "
            f"expected structure. Error: {exc}"
        ) from exc

    series_dict = dataset.get("series", {})
    if not series_dict:
        return []  # No matching data is a legitimate result, not a software error.

    if len(series_dict) > 1:
        # Resolve the meaning of each returned series so that structural
        # ambiguity can be diagnosed from the actual dimension values rather
        # than reported merely as the presence of multiple series.
        resolved_labels = []
        for series_key in series_dict:
            indices = [int(x) for x in series_key.split(":")]
            label = {}
            for pos, idx in enumerate(indices):
                if pos < len(series_dims):
                    dim = series_dims[pos]
                    try:
                        label[dim["id"]] = dim["values"][idx]["id"]
                    except IndexError:
                        label[dim["id"]] = f"<index {idx} out of range>"
            resolved_labels.append(label)
        raise OECDToolError(
            f"The query returned {len(series_dict)} series instead of exactly one — "
            f"the key did not constrain the request sufficiently "
            f"(open dimensions remain). Returned series: {resolved_labels}. "
            f"Update key_template in HEALTH_DATAFLOWS to constrain additional "
            f"dimensions explicitly."
        )

    _, series_data = next(iter(series_dict.items()))
    obs = series_data.get("observations", {})

    if not obs:
        return []

    observations = []
    for time_index_str, values in obs.items():
        time_index = int(time_index_str)
        year = time_values[time_index]["id"]
        value = values[0] if values else None
        observations.append({"year": year, "value": value})

    observations.sort(key=lambda o: o["year"])
    return observations


if __name__ == "__main__":
    # Manual validation utility; requires network access.
    # Inspect the live SDMX structure before relying on the indicator query
    # whenever the upstream dataflow structure may have changed.
    import json
    structure_xml = fetch_dataflow_structure("OECD.ELS.HD,DSD_SHA@DF_SHA,")
    print(structure_xml[:2000])  # Manual inspection of the beginning of the DSD
    print("---")
    result = get_oecd_indicator("CRI", "health_expenditure_pct_gdp", start_period="2015")
    print(json.dumps(result, indent=2, ensure_ascii=False))
