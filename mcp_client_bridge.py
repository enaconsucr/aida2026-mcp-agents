"""
mcp_client_bridge.py — MCP client that launches mcp_server.py as a subprocess
(via stdio) and exposes a simple interface for the harness to call tools by
speaking the real MCP protocol, rather than importing Python functions
directly.

This is the component that lets the paper honestly claim an "MCP-based
agent": the LLM (via OpenAI/Gemini tool-calling) decides which tool to
invoke, and this layer translates that decision into a real MCP call over
the standard protocol — in principle, the server could run on a different
machine or be replaced by any standard-conformant MCP implementation
without changing the harness.

Cliente MCP que lanza mcp_server.py como subproceso (vía stdio) y expone una
interfaz sencilla para que el harness llame herramientas hablando el
protocolo MCP real, en vez de importar funciones Python directamente.

Esta es la pieza que permite que el paper reivindique honestamente un
"agente basado en MCP": el LLM (vía tool-calling de OpenAI/Gemini) decide
qué herramienta invocar, y esta capa traduce esa decisión en una llamada MCP
real sobre el protocolo estándar — en principio, el servidor podría correr
en otra máquina o ser reemplazado por cualquier implementación MCP conforme
al estándar, sin cambiar el harness.

USAGE / USO:
    import asyncio
    from mcp_client_bridge import MCPClientBridge

    async def main():
        async with MCPClientBridge() as bridge:
            tools = await bridge.list_tools()
            result = await bridge.call_tool("query_local_reviews_tool", {"start_date": "2023-01-01", "end_date": "2023-12-31"})

    asyncio.run(main())
"""

import os
import sys
from contextlib import AsyncExitStack

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

SERVER_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server.py")


class MCPClientBridge:
    """
    Async context-manager wrapper: on entry, launches mcp_server.py as a
    subprocess and opens an MCP session; on exit, closes everything cleanly.

    Wrapper de administrador de contexto asíncrono: al entrar, lanza
    mcp_server.py como subproceso y abre una sesión MCP; al salir, cierra
    todo limpiamente.
    """

    def __init__(self, server_script: str = SERVER_SCRIPT, python_executable: str | None = None):
        self.server_script = server_script
        self.python_executable = python_executable or sys.executable
        self._exit_stack: AsyncExitStack | None = None
        self.session: ClientSession | None = None

    async def __aenter__(self) -> "MCPClientBridge":
        self._exit_stack = AsyncExitStack()
        params = StdioServerParameters(
            command=self.python_executable,
            args=[self.server_script],
            # The MCP subprocess must receive the environment variables required
            # by the server-side tools. Passing env explicitly ensures that
            # configuration such as REVIEWS_DB_PATH is available to the server
            # process. In security-sensitive deployments, this should be replaced
            # by an explicit allowlist rather than forwarding the full environment.
            env=dict(os.environ),
        )
        read_stream, write_stream = await self._exit_stack.enter_async_context(stdio_client(params))
        self.session = await self._exit_stack.enter_async_context(ClientSession(read_stream, write_stream))
        await self.session.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._exit_stack is not None:
            await self._exit_stack.aclose()

    async def list_tools(self) -> list[dict]:
        """
        Return the server's tools in a provider-neutral format (name,
        description, parameter JSON Schema) — ready to pass to
        the provider-specific adapter in providers.py.

        Devuelve las herramientas del servidor en formato neutral (nombre,
        descripción, JSON Schema de parámetros) — listas para pasar a
        el adaptador específico del proveedor en providers.py.
        """
        assert self.session is not None, "Usar dentro de 'async with MCPClientBridge() as bridge:'"
        result = await self.session.list_tools()
        return [
            {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.input_schema,
            }
            for t in result.tools
        ]

    async def read_resource(self, uri: str) -> str:
        """
        Read an MCP resource and concatenate its text blocks.

        Used, for example, to retrieve data://schema and provide it to the
        model as semantic context before tool selection.

        Lee un recurso MCP y concatena sus bloques de texto.

        Se usa, por ejemplo, para recuperar data://schema y entregarlo al
        modelo como contexto semántico antes de la selección de herramientas.
        """
        assert self.session is not None, "Usar dentro de 'async with MCPClientBridge() as bridge:'"

        result = await self.session.read_resource(uri)

        text_blocks = [
            item.text
            for item in result.contents
            if hasattr(item, "text") and item.text
        ]

        return "\n".join(text_blocks)


    async def call_tool(self, name: str, arguments: dict) -> dict:
        """
        Call a tool over the real MCP protocol and normalize the result to:
            {"success": bool, "result": dict|None, "error": str|None}

        Llama a una herramienta por protocolo MCP real y normaliza el
        resultado a:
            {"success": bool, "result": dict|None, "error": str|None}
        """
        assert self.session is not None, "Usar dentro de 'async with MCPClientBridge() as bridge:'"
        try:
            result = await self.session.call_tool(name, arguments)
        except Exception as exc:
            # Transport/protocol failure (not a failure of the tool itself) —
            # e.g. the server subprocess died.
            return {"success": False, "result": None, "error": f"Fallo de transporte MCP: {exc}"}

        if result.is_error:
            # The tool raised a controlled exception inside the server (see
            # mcp_server.py: WorldBankToolError/OECDToolError/ValueError are
            # propagated as is_error=True, not as a process crash).
            error_text = "; ".join(
                block.text for block in result.content if hasattr(block, "text")
            )
            return {"success": False, "result": None, "error": error_text}

        # The SDK may return structured_content (an already-parsed dict) or
        # only text/JSON blocks — prefer structured_content when available.
        if result.structured_content is not None:
            return {"success": True, "result": result.structured_content, "error": None}

        text_blocks = [block.text for block in result.content if hasattr(block, "text")]
        raw_text = "\n".join(text_blocks)

        # If structured_content is unavailable, the MCP SDK may expose the
        # tool result as JSON-encoded text. Attempt JSON parsing before falling
        # back to raw text so that downstream code can preserve the structured
        # result contract whenever possible.
        try:
            import json as _json
            parsed = _json.loads(raw_text)
            return {"success": True, "result": parsed, "error": None}
        except (ValueError, TypeError):
            return {"success": True, "result": {"raw_text": raw_text}, "error": None}
