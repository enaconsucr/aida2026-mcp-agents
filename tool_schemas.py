"""
tool_schemas.py — Defines the agent's tools in a provider-neutral format and
converts them into the specific schema each provider expects.

Why this layer exists: OpenAI and Gemini both declare "functions" around the
same idea (name, description, JSON Schema parameters) but with different
payload structures. Defining them once in a neutral format avoids maintaining
two copies of the same information that could drift apart.

SOURCE-SELECTION DESIGN: descriptions for World Bank and OECD tools avoid
assigning either source as the default for source-neutral questions. This
preserves genuine ambiguity in condition C and prevents tool selection from
being mechanically induced by asymmetric schema wording. Source preference
under ambiguity is therefore evaluated separately from tool-selection
accuracy, which applies only when the benchmark defines a unique expected
tool.

MCP INTEGRATION: these definitions provide a provider-neutral representation
of tool names, descriptions, and JSON Schema parameters. Provider-specific
adapters expose them through native function-calling interfaces, while actual
tool execution is delegated through the MCP client bridge to the MCP server.
Tool names must therefore remain consistent with those exposed by
mcp_server.py.

Define las herramientas del agente en un formato neutral y las convierte al
esquema específico que espera cada proveedor.

Por qué existe esta capa: OpenAI y Gemini declaran "funciones" con la misma
idea (nombre, descripción, parámetros JSON Schema) pero con estructuras de
payload distintas. Definir una sola vez en formato neutral evita mantener
dos copias de la misma información que podrían divergir.

DISEÑO DE SELECCIÓN DE FUENTE: las descripciones de las herramientas de World
Bank y OCDE evitan asignar ninguna de las dos como fuente por defecto para
preguntas neutrales respecto de la fuente. Esto preserva la ambigüedad
genuina de la condición C y evita que la selección de herramienta quede
inducida mecánicamente por una redacción asimétrica del esquema. La
preferencia de fuente bajo ambigüedad se evalúa, por tanto, por separado de
tool-selection accuracy, que solo aplica cuando el benchmark define una única
herramienta esperada.

INTEGRACIÓN CON MCP: estas definiciones ofrecen una representación neutral
respecto del proveedor de nombres de herramienta, descripciones y parámetros
en JSON Schema. Los adaptadores específicos de cada proveedor las exponen
mediante sus interfaces nativas de function-calling, mientras que la
ejecución real de las herramientas se delega, a través del bridge de cliente
MCP, al servidor MCP. Los nombres de herramienta deben, por tanto, mantenerse
consistentes con los expuestos por mcp_server.py.
"""

# Provider-neutral format: list of tools with name, description, and
# parameters in standard JSON Schema (a subset compatible with both providers).
TOOL_DEFINITIONS = [
    {
        "name": "get_worldbank_indicator",
        "description": (
            "Query a health indicator for a country from the World Bank REST "
            "API (World Development Indicators / Global Health Expenditure "
            "Database). Prefer this tool when the question explicitly asks "
            "for World Bank statistics."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "country_iso3": {
                    "type": "string",
                    "description": "ISO3 country code, e.g. 'CRI' for Costa Rica.",
                },
                "indicator": {
                    "type": "string",
                    "description": (
                        "Indicator key. One of: current_health_expenditure_pct_gdp, "
                        "current_health_expenditure_per_capita_usd, "
                        "domestic_govt_health_expenditure_pct_gdp, "
                        "domestic_govt_health_expenditure_pct_govt_expenditure, "
                        "out_of_pocket_expenditure_pct_che, external_health_expenditure_pct_che"
                    ),
                },
                "most_recent_n": {
                    "type": "integer",
                    "description": "If set, return only the N most recent non-empty observations.",
                },
                "start_year": {"type": "integer", "description": "Optional start year for a range query."},
                "end_year": {"type": "integer", "description": "Optional end year for a range query."},
            },
            "required": ["country_iso3", "indicator"],
        },
    },
    {
        "name": "get_oecd_indicator",
        "description": (
            "Query a health indicator for a country from the OECD SDMX API "
            "(System of Health Accounts). Prefer this tool when the question "
            "explicitly asks for OECD statistics."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "country_iso3": {
                    "type": "string",
                    "description": "ISO3 country code, e.g. 'CRI' for Costa Rica.",
                },
                "indicator": {
                    "type": "string",
                    "description": "Indicator key. Currently only: health_expenditure_pct_gdp",
                },
                "start_period": {"type": "string", "description": "Optional start year, e.g. '2015'."},
            },
            "required": ["country_iso3", "indicator"],
        },
    },
    {
        "name": "query_local_reviews",
        "description": (
            "Search and analyze the local corpus of user reviews of the digital "
            "health application. Use this for questions about citizen sentiment, "
            "reported problems, or complaints during a given period."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "start_date": {"type": "string", "description": "ISO date, e.g. '2023-01-01'."},
                "end_date": {"type": "string", "description": "ISO date, e.g. '2023-12-31'."},
                "query": {"type": "string", "description": "Optional keyword filter."},
            },
            "required": ["start_date", "end_date"],
        },
    },
]


def to_openai_format(tool_definitions=TOOL_DEFINITIONS) -> list[dict]:
    """Convert to OpenAI format (Chat Completions / Responses API tools=[...]).

    Convierte a formato OpenAI (Chat Completions / Responses API tools=[...]).
    """
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t["parameters"],
            },
        }
        for t in tool_definitions
    ]


