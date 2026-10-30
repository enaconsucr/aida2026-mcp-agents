"""
smoke_test.py — Verificación rápida de extremo a extremo antes de
ejecutar el benchmark experimental.

NO modifica questions_template.json ni ningún archivo de preguntas —
solo prueba que el entorno, las dependencias y las 3 herramientas MCP
respondan correctamente.

También verifica que Costa Rica devuelva observaciones reales en el
dataflow de salud de la OCDE. El resultado de ese chequeo específico se
imprime con una etiqueta [OCDE-CRI] fácil de ubicar en la salida.

USO:
    pip install -r requirements.txt
    Cargar OPENAI_API_KEY y GOOGLE_API_KEY de forma segura mediante un
    prompt interactivo oculto; no escribir ni almacenar las claves en
    este archivo ni en comandos visibles.
    export REVIEWS_DB_PATH="..."      # opcional — si falta, se reporta como esperado
    python smoke_test.py

Código de salida: 0 si los chequeos CRÍTICOS pasan (imports, servidor MCP,
listado de herramientas). Los chequeos de red externa (World Bank, OCDE)
y de claves de API se reportan pero NO hacen fallar el script — son
informativos, ya que este entorno podría no tener todo configurado aún.
"""

import asyncio
import os
import sys
import traceback

RESULTS: list[tuple[str, bool, str]] = []  # (nombre_chequeo, ok, detalle)


def record(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    status = "OK  " if ok else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))


def check_imports() -> bool:
    """
    Devuelve True solo si el import CRÍTICO (mcp) funcionó — es el único
    indispensable para levantar el servidor y probar las herramientas.
    openai/google-genai se reportan pero NO bloquean el resto de
    las pruebas: el usuario puede querer probar solo un proveedor.
    """
    print("\n=== 1. Verificando dependencias instaladas ===")
    critical_ok = True
    for module_name in ["mcp", "openai", "google.genai", "requests"]:
        try:
            __import__(module_name)
            record(f"import {module_name}", True)
        except ImportError as exc:
            is_critical = module_name in ("mcp", "requests")
            record(f"import {module_name}", False, str(exc) + (" (bloqueante)" if is_critical else " (opcional, no bloquea)"))
            if is_critical:
                critical_ok = False
    return critical_ok


def check_env_vars() -> None:
    print("\n=== 2. Verificando variables de entorno (informativo, no bloquea) ===")
    for var in ["OPENAI_API_KEY", "GOOGLE_API_KEY", "REVIEWS_DB_PATH"]:
        value = os.environ.get(var)
        if value:
            masked = value[:4] + "..." if var.endswith("KEY") else value
            record(f"env {var}", True, masked)
        else:
            record(f"env {var}", False, "no configurada — algunos chequeos posteriores se saltarán")


async def check_mcp_server() -> "MCPClientBridge | None":
    print("\n=== 3. Verificando servidor MCP (protocolo real) ===")
    try:
        from mcp_client_bridge import MCPClientBridge
    except ImportError as exc:
        record("import mcp_client_bridge", False, str(exc))
        return None

    try:
        bridge = MCPClientBridge()
        await bridge.__aenter__()
    except Exception as exc:
        record("levantar servidor MCP (subproceso)", False, f"{type(exc).__name__}: {exc}")
        return None

    record("levantar servidor MCP (subproceso)", True)

    try:
        tools = await bridge.list_tools()
        tool_names = {t["name"] for t in tools}
        expected = {"get_worldbank_indicator", "get_oecd_indicator", "query_local_reviews"}
        record("list_tools() devuelve las 3 herramientas esperadas", tool_names == expected, str(sorted(tool_names)))
    except Exception as exc:
        record("list_tools()", False, f"{type(exc).__name__}: {exc}")

    try:
        resources = await bridge.session.list_resources()
        resource_uris = {str(r.uri) for r in resources.resources}
        record("list_resources() incluye data://schema", "data://schema" in resource_uris, str(resource_uris))
    except Exception as exc:
        record("list_resources()", False, f"{type(exc).__name__}: {exc}")

    return bridge


async def check_worldbank(bridge) -> None:
    print("\n=== 4. Probando get_worldbank_indicator contra la API real (Costa Rica) ===")
    try:
        result = await bridge.call_tool(
            "get_worldbank_indicator",
            {"country_iso3": "CRI", "indicator": "current_health_expenditure_pct_gdp", "most_recent_n": 3},
        )
    except Exception as exc:
        record("get_worldbank_indicator (llamada)", False, f"{type(exc).__name__}: {exc}")
        return

    if not result["success"]:
        record("get_worldbank_indicator (CRI)", False, result["error"])
        return

    observations = result["result"].get("observations", [])
    record(
        "get_worldbank_indicator (CRI)",
        len(observations) > 0,
        f"{len(observations)} observaciones — {observations[:3]}",
    )


async def check_oecd(bridge) -> None:
    print("\n=== 5. Probando get_oecd_indicator contra la API real (Costa Rica) — [OCDE-CRI] ===")
    try:
        result = await bridge.call_tool(
            "get_oecd_indicator",
            {"country_iso3": "CRI", "indicator": "health_expenditure_pct_gdp", "start_period": "2015"},
        )
    except Exception as exc:
        record("[OCDE-CRI] get_oecd_indicator (llamada)", False, f"{type(exc).__name__}: {exc}")
        return

    if not result["success"]:
        record("[OCDE-CRI] get_oecd_indicator", False, result["error"])
        print(
            "     -> Si el error sugiere 'sin observaciones' o una key vacía, probar primero con "
            "'ESP' en vez de 'CRI' para descartar un error de construcción de la key antes de "
            "concluir que Costa Rica no tiene el dato en este dataflow."
        )
        return

    observations = result["result"].get("observations", [])
    record(
        "[OCDE-CRI] get_oecd_indicator",
        len(observations) > 0,
        f"{len(observations)} observaciones — {observations[:3]}" if observations else "CERO observaciones",
    )


async def check_local_reviews(bridge) -> None:
    print("\n=== 6. Probando query_local_reviews ===")
    if not os.environ.get("REVIEWS_DB_PATH"):
        record(
            "query_local_reviews",
            True,  # no es un fallo del smoke test, es un prerequisito no configurado aún
            "REVIEWS_DB_PATH no configurado — se salta esta prueba (esperado hasta tener el corpus real)",
        )
        return

    try:
        result = await bridge.call_tool(
            "query_local_reviews",
            {"start_date": "2023-01-01", "end_date": "2023-12-31"},
        )
    except Exception as exc:
        record("query_local_reviews (llamada)", False, f"{type(exc).__name__}: {exc}")
        return

    if not result["success"]:
        record("query_local_reviews", False, result["error"])
        return

    total = result["result"].get("total_reviews", "?")
    record("query_local_reviews", True, f"total_reviews={total}")


def check_harness_imports() -> None:
    print("\n=== 7. Verificando imports del harness (tool_schemas, providers) ===")
    harness_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "harness")
    sys.path.insert(0, harness_dir)
    try:
        import tool_schemas  # noqa: F401
        record("import harness/tool_schemas.py", True)
    except ImportError as exc:
        record("import harness/tool_schemas.py", False, str(exc))

    try:
        import providers  # noqa: F401
        record("import harness/providers.py", True)
    except ImportError as exc:
        record("import harness/providers.py", False, str(exc))


async def main_async() -> int:
    imports_ok = check_imports()
    check_env_vars()
    check_harness_imports()

    if not imports_ok:
        print("\nFaltan dependencias críticas — instalar con: pip install -r requirements.txt")
        print_summary()
        return 1

    bridge = await check_mcp_server()
    if bridge is None:
        print("\nEl servidor MCP no arrancó — revisar traceback arriba antes de continuar.")
        print_summary()
        return 1

    try:
        await check_worldbank(bridge)
        await check_oecd(bridge)
        await check_local_reviews(bridge)
    finally:
        await bridge.__aexit__(None, None, None)

    print_summary()

    critical_checks = [r for r in RESULTS if "servidor MCP" in r[0] or "list_tools" in r[0] or "import" in r[0]]
    return 0 if all(ok for _, ok, _ in critical_checks) else 1


def print_summary() -> None:
    print("\n" + "=" * 60)
    print("RESUMEN")
    print("=" * 60)
    n_ok = sum(1 for _, ok, _ in RESULTS if ok)
    print(f"{n_ok}/{len(RESULTS)} chequeos OK")
    failed = [name for name, ok, _ in RESULTS if not ok]
    if failed:
        print("Chequeos con problemas:")
        for name in failed:
            print(f"  - {name}")
    print(
        "\nRecordatorio: este script NO modifica questions_template.json ni "
        "ningún archivo de preguntas — solo verifica que el entorno y las "
        "herramientas respondan antes de diseñar/correr el benchmark."
    )


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main_async())
    except Exception:
        print("\nError inesperado no controlado:")
        traceback.print_exc()
        exit_code = 1
    sys.exit(exit_code)
