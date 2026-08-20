#!/usr/bin/env python3
"""
main.py — Execution entrypoint for the AI Onboarding Engineer.

Opens the local ChromaDB vector store built by ingest.py, wires up the
LangGraph onboarding workflow defined in agents.py, and drives an
interactive command-line session: it prints each agent's output, pauses at
the Reviewer Agent's human-in-the-loop interrupt to collect the learner's
submission, and resumes the graph until the learner finishes the syllabus
(or exhausts their attempts on a step and is pointed to a human mentor).

Usage
-----
    # First, make sure you've ingested a repo:
    python ingest.py --source-dir /path/to/repo --embedding-provider local

    # Then start an onboarding session (defaults to a local Ollama model):
    python main.py

    # Or with a cloud provider:
    python main.py --chat-provider openai --chat-model gpt-4o-mini \
                    --embedding-provider openai

    # Resume a specific prior session (its state lives in the checkpoint DB):
    python main.py --thread-id 3f9c2e5e-...
"""

from __future__ import annotations

import argparse
import logging
import sys
import uuid
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from langchain_core.messages import AIMessage
from langgraph.types import Command

from agents import AgentError, OnboardingState, build_graph, get_chat_model, get_checkpointer, initial_state
from ingest import DEFAULT_COLLECTION_NAME, DEFAULT_PERSIST_DIR, configure_logging, get_embedding_function

logger = logging.getLogger("main")

DEFAULT_CHECKPOINT_DB = "./data/checkpoints.sqlite"
DEFAULT_CHAT_PROVIDER = "ollama"


# ---------------------------------------------------------------------------
# Vector store
# ---------------------------------------------------------------------------

def open_vector_store(persist_dir: Path, collection_name: str, embedding_function):
    """
    Open an existing, populated ChromaDB collection in read/query mode.

    Raises:
        AgentError: if the persist directory, collection, or any documents
        within it can't be found — almost always meaning ingest.py hasn't
        been run yet (or was run against a different persist-dir/collection).
    """
    try:
        import chromadb
        from chromadb.config import Settings
        from langchain_chroma import Chroma
    except ImportError as exc:
        raise ImportError(
            "ChromaDB integration requires 'chromadb' and 'langchain-chroma'. Install them with:\n"
            "    pip install chromadb langchain-chroma"
        ) from exc

    if not persist_dir.exists():
        raise AgentError(
            f"No vector store found at '{persist_dir}'. Run ingest.py against your repository "
            "first to build the knowledge base."
        )

    try:
        client = chromadb.PersistentClient(path=str(persist_dir), settings=Settings(anonymized_telemetry=False))
    except Exception as exc:
        raise AgentError(f"Failed to open ChromaDB at '{persist_dir}': {exc}") from exc

    try:
        collection = client.get_collection(name=collection_name)
    except Exception as exc:
        raise AgentError(
            f"Collection '{collection_name}' not found in '{persist_dir}'. Run ingest.py first "
            f"(with --collection-name {collection_name}). Original error: {exc}"
        ) from exc

    count = collection.count()
    if count == 0:
        raise AgentError(
            f"Collection '{collection_name}' exists but is empty. Run ingest.py against your "
            "repository before starting an onboarding session."
        )

    vector_store = Chroma(client=client, collection_name=collection_name, embedding_function=embedding_function)
    logger.info("Opened vector store '%s' (%d chunks) at '%s'.", collection_name, count, persist_dir)
    return vector_store


# ---------------------------------------------------------------------------
# Interactive session
# ---------------------------------------------------------------------------

def _print_new_messages(result: dict, already_printed: int) -> int:
    messages = result.get("messages", [])
    for msg in messages[already_printed:]:
        speaker = "Onboarding Engineer" if isinstance(msg, AIMessage) else "You"
        print(f"\n[{speaker}]\n{msg.content}")
    return len(messages)


def _prompt_for_submission(payload: dict) -> str:
    print("\n" + "-" * 70)
    print(f"Step {payload['step']}/{payload['total_steps']} — attempt {payload['attempt']}/{payload['max_attempts']}")
    if payload.get("previous_feedback"):
        print(f"(Previous feedback: {payload['previous_feedback']})")
    print("\nSubmit your solution below. Type a line containing only END when you're done:\n")

    lines = []
    while True:
        try:
            line = input()
        except EOFError:
            break
        if line.strip() == "END":
            break
        lines.append(line)

    return "\n".join(lines).strip()


def _print_final_summary(result: dict) -> None:
    print("\n" + "=" * 70)
    syllabus = result.get("syllabus") or {}
    total_steps = len(syllabus.get("steps", []))
    completed = result.get("completed_steps", [])

    if total_steps and len(completed) >= total_steps:
        print(f"Onboarding complete! You finished all {total_steps} steps.")
    else:
        print(f"Session ended after completing {len(completed)}/{total_steps} step(s).")

    if completed:
        print("\nCompleted steps:")
        for record in completed:
            print(f"  {record['step']}. {record['title']} (in {record['attempts']} attempt(s))")

    remaining_idx = result.get("current_step_index", 0)
    if total_steps and remaining_idx < total_steps:
        print(f"\nTo continue, restart with the same --thread-id. Next up: {syllabus['steps'][remaining_idx]['title']}")


def run_interactive_session(graph, thread_id: str) -> None:
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\nStarting onboarding session (thread id: {thread_id})")
    print("=" * 70)

    printed = 0
    result = graph.invoke(initial_state(), config=config)
    printed = _print_new_messages(result, printed)

    while "__interrupt__" in result:
        interrupt_obj = result["__interrupt__"][0]
        answer = _prompt_for_submission(interrupt_obj.value)
        result = graph.invoke(Command(resume=answer), config=config)
        printed = _print_new_messages(result, printed)

    _print_final_summary(result)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[list] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run an interactive AI Onboarding Engineer session.")

    parser.add_argument("--persist-dir", type=str, default=DEFAULT_PERSIST_DIR,
                         help=f"Directory the ChromaDB collection was persisted to (default: {DEFAULT_PERSIST_DIR})")
    parser.add_argument("--collection-name", type=str, default=DEFAULT_COLLECTION_NAME,
                         help=f"Name of the Chroma collection to query (default: {DEFAULT_COLLECTION_NAME})")
    parser.add_argument("--embedding-provider", type=str, default="local", choices=["local", "openai"],
                         help="Must match the provider used when running ingest.py (default: local)")
    parser.add_argument("--embedding-model", type=str, default=None,
                         help="Override the default embedding model for the chosen provider.")

    parser.add_argument("--chat-provider", type=str, default=DEFAULT_CHAT_PROVIDER,
                         choices=["ollama", "openai", "anthropic"],
                         help=f"LLM used by the agents to reason (default: {DEFAULT_CHAT_PROVIDER})")
    parser.add_argument("--chat-model", type=str, default=None,
                         help="Override the default chat model for the chosen provider.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Chat model temperature (default: 0.2)")

    parser.add_argument("--checkpoint-db", type=str, default=DEFAULT_CHECKPOINT_DB,
                         help=f"Path to the SQLite checkpoint database (default: {DEFAULT_CHECKPOINT_DB})")
    parser.add_argument("--thread-id", type=str, default=None,
                         help="Reuse a prior session's thread id to resume it; omit to start a new session.")

    parser.add_argument("--verbose", action="store_true", help="Enable debug-level logging.")

    return parser.parse_args(argv)


def main(argv: Optional[list] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)
    load_dotenv()

    persist_dir = Path(args.persist_dir).expanduser().resolve()
    checkpoint_db = Path(args.checkpoint_db).expanduser().resolve()
    thread_id = args.thread_id or str(uuid.uuid4())

    try:
        embedding_function = get_embedding_function(args.embedding_provider, args.embedding_model)
        vector_store = open_vector_store(persist_dir, args.collection_name, embedding_function)
        llm = get_chat_model(args.chat_provider, args.chat_model, temperature=args.temperature)
    except (ImportError, EnvironmentError, ValueError, AgentError, RuntimeError) as exc:
        logger.error("Startup failed: %s", exc)
        return 1

    try:
        checkpointer, conn = get_checkpointer(checkpoint_db)
    except (OSError, RuntimeError) as exc:
        logger.error("Could not initialize checkpoint database: %s", exc)
        return 1

    try:
        graph = build_graph(vector_store, llm, checkpointer)
        try:
            run_interactive_session(graph, thread_id)
        except KeyboardInterrupt:
            print(f"\n\nSession paused. Resume it later with:\n    python main.py --thread-id {thread_id}")
            return 130
        except AgentError as exc:
            logger.error("Onboarding session failed: %s", exc)
            return 1
    finally:
        conn.close()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        logger.exception("Unhandled error while running the onboarding session.")
        sys.exit(1)