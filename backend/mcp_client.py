"""
Sync bridge to the campaign-dispatch MCP server.

The mcp SDK is asyncio; run_agent is sync under Flask/gunicorn threads. Each
worker process lazily starts ONE daemon thread running an event loop that holds
a long-lived stdio ClientSession to a spawned mcp_server.py subprocess. Sync
callers submit coroutines with run_coroutine_threadsafe.

Lifecycle rules this encodes:
- Nothing starts at import time. gunicorn runs with --preload, so the module is
  imported in the master pre-fork; the thread/subprocess appear only on first
  use inside a worker, and a PID guard re-initializes after any fork.
- The stdio_client/ClientSession context managers are entered and exited inside
  a single long-lived task (_session_task) — anyio cancel scopes are bound to
  the task that created them.
- stdio_client sanitizes the child's env, so live-send credentials are forwarded
  explicitly via _PASS_ENV. ANTHROPIC_API_KEY is deliberately NOT on the list:
  the MCP server never holds the LLM key, and this process never holds the
  SMTP/Twilio creds' only consumer.
- Every failure degrades gracefully: no tools (empty list) rather than a broken
  chat, and a dead subprocess is respawned on the next call.
"""

import asyncio
import os
import sys
import threading

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client, get_default_environment

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SERVER_PATH = os.path.join(BASE_DIR, "mcp_server.py")

_PASS_ENV = ("LIVE_SEND", "LIVE_SEND_CAP", "CAMPAIGN_DB_PATH",
             "GMAIL_ADDRESS", "GMAIL_APP_PASSWORD",
             "TWILIO_ACCOUNT_SID", "TWILIO_AUTH_TOKEN", "TWILIO_FROM_NUMBER",
             "DEMO_RECIPIENT_EMAIL", "DEMO_RECIPIENT_PHONE")

STARTUP_TIMEOUT = 60    # server imports pandas/sklearn + loads model.pkl
CALL_TIMEOUT = 120

_lock = threading.Lock()
_state = None   # {"pid", "loop", "session", "tools", "ready", "failed"}


async def _session_task(state: dict) -> None:
    """Own the stdio subprocess + session for the life of the process."""
    try:
        params = StdioServerParameters(
            command=sys.executable, args=[SERVER_PATH], cwd=BASE_DIR,
            env={**get_default_environment(),
                 **{k: os.environ[k] for k in _PASS_ENV if k in os.environ}})
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                state["tools"] = (await session.list_tools()).tools
                state["session"] = session
                state["ready"].set()
                await asyncio.Event().wait()   # hold open until process exit
    except Exception as e:
        print(f"[mcp_client] campaign service failed to start: "
              f"{type(e).__name__}: {e}", file=sys.stderr)
        state["failed"] = True
        state["ready"].set()


def _ensure_started() -> dict:
    global _state
    with _lock:
        if _state is not None and _state["pid"] == os.getpid():
            return _state
        state = {"pid": os.getpid(), "loop": asyncio.new_event_loop(),
                 "session": None, "tools": [], "ready": threading.Event(),
                 "failed": False}
        threading.Thread(target=state["loop"].run_forever,
                         name="mcp-client-loop", daemon=True).start()
        asyncio.run_coroutine_threadsafe(_session_task(state), state["loop"])
        _state = state
    state["ready"].wait(timeout=STARTUP_TIMEOUT)
    if not state["ready"].is_set():
        state["failed"] = True
    return state


def _reset() -> None:
    global _state
    with _lock:
        _state = None


def get_tool_schemas() -> list:
    """MCP tools as Anthropic-native dicts; [] if the service is down."""
    state = _ensure_started()
    if state["failed"]:
        return []
    return [{"name": t.name, "description": t.description or "",
             "input_schema": t.inputSchema} for t in state["tools"]]


def tool_names() -> set:
    return {t["name"] for t in get_tool_schemas()}


def call_tool(name: str, args: dict):
    """Call an MCP tool; returns the JSON text result or an {'error': ...} dict."""
    state = _ensure_started()
    if state["failed"]:
        return {"error": "campaign dispatch service is unavailable"}
    try:
        fut = asyncio.run_coroutine_threadsafe(
            state["session"].call_tool(name, args or {}), state["loop"])
        result = fut.result(timeout=CALL_TIMEOUT)
    except Exception as e:
        _reset()   # dead subprocess / broken pipe -> respawn on next call
        return {"error": f"campaign dispatch service error: "
                         f"{type(e).__name__}: {e}"}
    text = "".join(c.text for c in result.content if c.type == "text")
    if result.isError:
        return {"error": text or f"{name} failed"}
    return text
