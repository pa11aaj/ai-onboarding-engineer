"""
agent_service.py — Framework-agnostic session logic for the AI Onboarding
Engineer's web backend.

Wraps the LangGraph graph from agents.py with plain functions
(`start_session`, `submit_answer`) that server.py's FastAPI routes call
directly. Kept separate from the web framework on purpose: the core logic
is easy to unit test and reason about without spinning up an HTTP server.

Session state (which step a learner is on, that the graph is paused
awaiting their answer, etc.) lives in whatever checkpointer CHECKPOINTER
selects — see _build_checkpointer(). This module itself holds no
per-session state in process memory, on purpose: a serverless platform
(Vercel and similar) may route two requests for the same session to two
different, unrelated processes, so anything held only in this process's
RAM would be invisible to the request that lands on a different instance.
With an external checkpointer (Postgres/Redis), every instance reads and
writes the same store, so it doesn't matter which one handles a request.

CHECKPOINTER=memory (the default, for local dev) keeps state in this
process's RAM only — fine for `uvicorn server:app` on your laptop, but NOT
suitable for a serverless deployment, since it reintroduces exactly the
cross-instance problem described above.
"""

from __future__ import annotations

import logging
import os
import uuid
from contextlib import ExitStack
from typing import List, Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agents import AgentError, build_graph, get_chat_model, get_vector_store, initial_state
from ingest import configure_logging, get_embedding_function

logger = logging.getLogger("agent_service")

load_dotenv()
configure_logging(verbose=os.environ.get("LOG_LEVEL", "").upper() == "DEBUG")


class SessionNotFoundError(Exception):
    """Raised when a thread_id doesn't correspond to a resumable, in-progress session."""


# ---------------------------------------------------------------------------
# Process-lifetime state: built once at startup, shared by every session.
# Deliberately NOT per-session — see module docstring.
# ---------------------------------------------------------------------------

STARTUP_ERROR: Optional[str] = None
_graph = None
_exit_stack: Optional[ExitStack] = None  # holds open DB/Redis connections, if any


def _open_vector_store():
    """
    Open the Pinecone-backed vector store ingest.py populated. All Pinecone
    configuration (PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_NAMESPACE,
    PINECONE_CLOUD, PINECONE_REGION) is read from os.environ inside
    agents.get_vector_store() — set these in your hosting provider's
    environment variables (e.g. Vercel Project Settings) in production, or
    in a local .env file for development.
    """
    embedding_provider = os.environ.get("EMBEDDING_PROVIDER", "local")
    embedding_model = os.environ.get("EMBEDDING_MODEL") or None
    embedding_function = get_embedding_function(embedding_provider, embedding_model)
    return get_vector_store(embedding_function, require_populated=True)


def _build_checkpointer(stack: ExitStack):
    """
    Build the checkpointer selected by CHECKPOINTER (default: "memory").

    "postgres" and "redis" checkpointers hold an open DB connection for the
    life of the process, managed as a context manager — `stack` (an
    ExitStack owned by this module, closed in shutdown()) keeps that
    connection open across every request instead of closing it the moment
    this function returns.

    Raises:
        ImportError: if the selected backend's package isn't installed.
        EnvironmentError: if its connection string env var isn't set.
        ValueError: if CHECKPOINTER is set to something unrecognized.
    """
    backend = os.environ.get("CHECKPOINTER", "memory").lower().strip()

    if backend == "memory":
        logger.warning(
            "CHECKPOINTER=memory: session state lives only in this process's RAM. Fine for "
            "local development; NOT suitable for a serverless deployment (see module docstring)."
        )
        return InMemorySaver()

    if backend == "postgres":
        try:
            from langgraph.checkpoint.postgres import PostgresSaver
        except ImportError as exc:
            raise ImportError(
                "CHECKPOINTER=postgres requires the 'langgraph-checkpoint-postgres' package "
                "(and a driver, e.g. 'psycopg[binary]'). Install with:\n"
                "    pip install langgraph-checkpoint-postgres 'psycopg[binary]'"
            ) from exc

        db_url = os.environ.get("DATABASE_URL") or os.environ.get("POSTGRES_URL")
        if not db_url:
            raise EnvironmentError(
                "CHECKPOINTER=postgres requires DATABASE_URL (or POSTGRES_URL) — a Postgres "
                "connection string, e.g. from Neon. Set it in your hosting provider's "
                "environment variables (e.g. Vercel Project Settings) or your local .env file."
            )

        checkpointer = stack.enter_context(PostgresSaver.from_conn_string(db_url))
        checkpointer.setup()  # idempotent: creates the checkpoint tables if they don't exist yet
        logger.info("Using Postgres checkpointer for session state.")
        return checkpointer

    if backend == "redis":
        try:
            from langgraph.checkpoint.redis import RedisSaver
        except ImportError as exc:
            raise ImportError(
                "CHECKPOINTER=redis requires the 'langgraph-checkpoint-redis' package. Install with:\n"
                "    pip install langgraph-checkpoint-redis\n"
                "Note: this checkpointer needs Redis with the RediSearch and RedisJSON modules "
                "(Redis Stack / Redis Cloud) — plain/serverless Redis offerings that lack those "
                "modules (e.g. Upstash) are not compatible. Postgres is the safer default choice."
            ) from exc

        redis_url = os.environ.get("REDIS_URL")
        if not redis_url:
            raise EnvironmentError(
                "CHECKPOINTER=redis requires REDIS_URL. Set it in your hosting provider's "
                "environment variables or your local .env file."
            )

        checkpointer = stack.enter_context(RedisSaver.from_conn_string(redis_url))
        checkpointer.setup()  # idempotent: creates the required indices if they don't exist yet
        logger.info("Using Redis checkpointer for session state.")
        return checkpointer

    raise ValueError(f"Unsupported CHECKPOINTER value: '{backend}' (expected 'memory', 'postgres', or 'redis')")


def initialize() -> None:
    """
    Build the shared vector store, LLM, checkpointer, and compiled graph
    once at process startup. Failures are captured in STARTUP_ERROR rather
    than raised, so the web server still comes up and can report a clear
    503 instead of crashing the whole deployment.
    """
    global _graph, STARTUP_ERROR, _exit_stack

    chat_provider = os.environ.get("CHAT_PROVIDER", "openai")
    chat_model = os.environ.get("CHAT_MODEL") or None

    stack = ExitStack()
    try:
        vector_store = _open_vector_store()
        llm = get_chat_model(chat_provider, chat_model, temperature=0.2)
        checkpointer = _build_checkpointer(stack)
        _graph = build_graph(vector_store, llm, checkpointer)
        _exit_stack = stack
        STARTUP_ERROR = None
        logger.info("Onboarding graph initialized (chat provider: %s).", chat_provider)
    except Exception as exc:
        stack.close()  # don't leak a half-open DB connection if init failed partway through
        STARTUP_ERROR = str(exc)
        logger.error("Startup failed: %s", exc)


def shutdown() -> None:
    """Close any open checkpointer connection. Call this on process shutdown
    (server.py's lifespan does, after the `yield`)."""
    global _exit_stack
    if _exit_stack is not None:
        _exit_stack.close()
        _exit_stack = None


# ---------------------------------------------------------------------------
# Session operations
# ---------------------------------------------------------------------------

def _serialize_messages(messages) -> List[dict]:
    return [{"role": "assistant" if isinstance(m, AIMessage) else "user", "content": m.content} for m in messages]


def _format_result(thread_id: str, result: dict) -> dict:
    # Always the FULL transcript, not just "messages since last call": with no
    # per-session bookkeeping in process memory, there's nothing here to track
    # "already seen" against. The transcript is small (a handful of messages
    # per step), so the client just re-renders it each time.
    done = "__interrupt__" not in result
    interrupt_payload = None if done else result["__interrupt__"][0].value

    return {
        "thread_id": thread_id,
        "messages": _serialize_messages(result.get("messages", [])),
        "interrupt": interrupt_payload,
        "done": done,
        "syllabus": result.get("syllabus"),
        "current_step_index": result.get("current_step_index", 0),
        "completed_steps": result.get("completed_steps", []),
        "quiz": result.get("quiz"),
        "work_aid": result.get("work_aid"),
    }


def start_session() -> dict:
    """Create a brand-new onboarding session and run it up to the first
    human-in-the-loop pause (or to completion, in principle)."""
    if _graph is None:
        raise AgentError(STARTUP_ERROR or "Graph is not initialized.")

    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id}}
    result = _graph.invoke(initial_state(), config=config)
    return _format_result(thread_id, result)


def submit_answer(thread_id: str, answer: str) -> dict:
    """Resume a paused session with the learner's submission and run it
    forward to the next pause or to completion.

    There's no local record of "known" thread_ids to check against (see
    module docstring), so validity is determined entirely by whether the
    checkpointer can actually resume this thread: an unknown, already-
    finished, or expired thread_id is treated uniformly as "not resumable"
    rather than relying on a specific framework exception type. An AgentError
    raised from *inside* a node (e.g. the LLM or vector store failing mid-
    session) is deliberately NOT caught here — that's a real failure with
    its own clear message, not a "session not found" situation, so it's
    left to propagate to the caller as-is.
    """
    if _graph is None:
        raise AgentError(STARTUP_ERROR or "Graph is not initialized.")

    if not thread_id or not isinstance(thread_id, str):
        raise SessionNotFoundError("A valid session thread_id is required.")

    config = {"configurable": {"thread_id": thread_id}}
    try:
        result = _graph.invoke(Command(resume=answer), config=config)
    except AgentError:
        raise
    except Exception as exc:
        raise SessionNotFoundError(
            f"Session '{thread_id}' can't be resumed — it may not exist, may have already "
            "finished, or may have expired. Start a new session."
        ) from exc

    # Defensive fallback: some checkpointer/graph combinations may not raise
    # for an invalid thread_id and instead silently hand back a near-empty
    # state. Treat that the same way rather than returning a hollow response.
    if not result.get("messages") and not result.get("syllabus"):
        raise SessionNotFoundError(
            f"Session '{thread_id}' has no history — it may not exist. Start a new session."
        )

    return _format_result(thread_id, result)