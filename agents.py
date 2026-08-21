#!/usr/bin/env python3
"""
agents.py — LangGraph definition for the AI Onboarding Engineer's core
multi-agent workflow.

Three nodes are wired into a single stateful graph:

    1. syllabus_designer_node — queries the Pinecone vector store for source
       context (a repository, docs site, or any ingested content) and
       produces a structured 3-step learning outline, ordered from true
       fundamentals to more advanced material. The learner is assumed to
       have NO access to any repository or file system — every step must
       be self-contained, teachable through text alone.
    2. lab_generator_node     — turns the current syllabus step into a
       concrete, hands-on exercise grounded in the retrieved context, with
       any needed code/reference snippets embedded directly in the text
       rather than referencing files the learner can't open.
    3. reviewer_node          — a human-in-the-loop node. It pauses the graph
       (via LangGraph's `interrupt`) to collect the learner's submission,
       evaluates it against the lab's success criteria and the source
       context, and routes the graph to a retry, the next step, completion,
       or escalation to a human mentor.

The graph is persisted with a SQLite checkpointer so a session's state
(including a pending human-in-the-loop interrupt) survives process restarts
as long as the same thread_id is reused.

The vector store backing retrieval is Pinecone (serverless) rather than a
local ChromaDB directory, so this module has no dependency on local disk
for document retrieval — a requirement for running behind a stateless
deployment like a Vercel serverless function. All Pinecone configuration
(API key, index name, cloud/region) is read from `os.environ`, so the same
code works with a local `.env` file and with environment variables set in
a hosting provider's dashboard (e.g. Vercel's Project Settings).
"""

from __future__ import annotations

import concurrent.futures
import functools
import json
import logging
import os
import re
import sqlite3
import time
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, List, Optional, Tuple, Type, TypedDict, TypeVar

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.types import interrupt
from pydantic import BaseModel, Field, ValidationError, field_validator

if TYPE_CHECKING:
    # Only needed for type checkers — see get_checkpointer() for why the
    # real import is deferred to runtime instead of happening here.
    from langgraph.checkpoint.sqlite import SqliteSaver

logger = logging.getLogger("agents")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class AgentError(Exception):
    """Raised for expected, user-actionable failures inside the graph
    (empty/unreachable vector store, LLM/provider misconfiguration, or an
    LLM that repeatedly fails to produce valid structured output)."""


# ---------------------------------------------------------------------------
# Structured output schemas
# ---------------------------------------------------------------------------

class LearningStep(BaseModel):
    title: str = Field(description="Short, descriptive title for this learning step.")
    objective: str = Field(
        description="What the developer should understand or be able to do after this step."
    )
    key_concepts: List[str] = Field(
        description="3-5 specific terms, concepts, or topics from the source material this step focuses on."
    )


class Syllabus(BaseModel):
    project_summary: str = Field(
        description="2-4 sentence summary of what this repository/project/documentation set covers, grounded in the retrieved context."
    )
    steps: List[LearningStep] = Field(
        description="Exactly 3 progressive learning steps, from foundational to advanced."
    )

    @field_validator("steps")
    @classmethod
    def must_have_three_steps(cls, value: List[LearningStep]) -> List[LearningStep]:
        if len(value) != 3:
            raise ValueError(f"Syllabus must contain exactly 3 steps, got {len(value)}.")
        return value


class Lab(BaseModel):
    title: str = Field(description="Short title for this hands-on lab.")
    lesson: str = Field(
        description=(
            "1-3 paragraph, self-contained explanation of the concept, written for someone "
            "completely new to it. Must fully explain everything the learner needs — they have "
            "no access to any repository, file, or external material beyond this text."
        )
    )
    challenge: str = Field(
        description=(
            "A single, focused question directly testing something explicitly covered in the "
            "lesson text above — answerable in a few sentences typed into a plain text box. "
            "Must be completable using only the lesson text and the learner's own general "
            "knowledge. NEVER ask for a diagram, drawing, image, chart, or anything visual — "
            "the learner can only submit plain text. NEVER ask for a long essay, a multi-file "
            "project, or anything requiring more than a short paragraph or a few lines of code. "
            "Never instruct the learner to open, find, or navigate to a specific file, path, or "
            "repository, since they have no access to one."
        )
    )
    starter_code: str = Field(
        default="", description="Optional starter code/skeleton for the challenge; empty string if not applicable."
    )
    success_criteria: List[str] = Field(
        description=(
            "3-5 concrete, checkable criteria a correct short answer must satisfy — scaled to "
            "a response of a few sentences or a short snippet, not a long or multi-part answer."
        )
    )


NUM_QUIZ_QUESTIONS = 3


class QuizQuestion(BaseModel):
    question: str = Field(description="A single multiple-choice question testing understanding of this step's material.")
    options: List[str] = Field(description="Exactly 4 answer options, in the order they should be displayed.")
    correct_index: int = Field(description="The 0-based index into `options` of the single correct answer.")
    explanation: str = Field(
        description="1-2 sentence explanation of why the correct answer is correct, shown to the learner after they answer."
    )

    @field_validator("options")
    @classmethod
    def must_have_four_options(cls, value: List[str]) -> List[str]:
        if len(value) != 4:
            raise ValueError(f"Each quiz question must have exactly 4 options, got {len(value)}.")
        return value

    @field_validator("correct_index")
    @classmethod
    def correct_index_in_range(cls, value: int) -> int:
        if not (0 <= value <= 3):
            raise ValueError(f"correct_index must be between 0 and 3, got {value}.")
        return value


class Quiz(BaseModel):
    questions: List[QuizQuestion] = Field(
        description=f"Exactly {NUM_QUIZ_QUESTIONS} multiple-choice questions covering this step's material."
    )

    @field_validator("questions")
    @classmethod
    def must_have_n_questions(cls, value: List[QuizQuestion]) -> List[QuizQuestion]:
        if len(value) != NUM_QUIZ_QUESTIONS:
            raise ValueError(f"Quiz must contain exactly {NUM_QUIZ_QUESTIONS} questions, got {len(value)}.")
        return value


class WorkAid(BaseModel):
    title: str = Field(description="Short title for this reference/cheat-sheet.")
    body: str = Field(
        description=(
            "A concise, scannable reference for this step's topic — key terms and definitions, "
            "important syntax, or a short checklist, whichever fits best. Plain text with short "
            "lines and clear structure (e.g. one 'term: definition' per line); this is displayed "
            "as-is, not rendered as markdown, so avoid markdown syntax like '#' or '**'."
        )
    )


class ReviewVerdict(BaseModel):
    passed: bool = Field(description="Whether the submission meets the lab's success criteria.")
    feedback: str = Field(
        description="Specific, constructive feedback for the learner, referencing the success criteria and the source material."
    )
    unmet_criteria: List[str] = Field(
        default_factory=list, description="Which success criteria were not satisfied; empty if passed."
    )


T = TypeVar("T", bound=BaseModel)


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class OnboardingState(TypedDict):
    repo_summary: str
    syllabus: Optional[dict]
    current_step_index: int
    lab: Optional[dict]
    quiz: Optional[dict]
    work_aid: Optional[dict]
    lab_context: str
    attempt_count: int
    review_passed: Optional[bool]
    review_feedback: Optional[str]
    completed_steps: List[dict]
    messages: Annotated[List[BaseMessage], add_messages]


def initial_state() -> OnboardingState:
    """A fresh, fully-populated starting state for a new onboarding session."""
    return {
        "repo_summary": "",
        "syllabus": None,
        "current_step_index": 0,
        "lab": None,
        "quiz": None,
        "work_aid": None,
        "lab_context": "",
        "attempt_count": 0,
        "review_passed": None,
        "review_feedback": None,
        "completed_steps": [],
        "messages": [],
    }


# ---------------------------------------------------------------------------
# LLM / provider factory
# ---------------------------------------------------------------------------

def get_chat_model(provider: str, model_name: Optional[str] = None, temperature: float = 0.2):
    """
    Build a chat model for the agents to reason with.

    Supported providers:
        "ollama"    — local, offline inference via Ollama (default model: llama3.1).
        "openai"    — OpenAI's chat API. Requires OPENAI_API_KEY.
        "anthropic" — Anthropic's Claude API. Requires ANTHROPIC_API_KEY and an
                      explicit --chat-model (no default is assumed, since valid
                      model IDs change over time).

    Raises:
        ImportError: if the required integration package isn't installed.
        EnvironmentError: if a required API key is missing.
        ValueError: if `provider` is unrecognized or a required model isn't given.
    """
    provider = provider.lower().strip()

    if provider == "ollama":
        try:
            from langchain_ollama import ChatOllama
        except ImportError as exc:
            raise ImportError(
                "Local chat models require the 'langchain-ollama' package. Install with:\n"
                "    pip install langchain-ollama\n"
                "and make sure the Ollama daemon is running with the model pulled, e.g.:\n"
                "    ollama pull llama3.1"
            ) from exc

        model = model_name or "llama3.1"
        logger.info("Using local Ollama chat model: %s", model)
        return ChatOllama(model=model, temperature=temperature)

    if provider == "openai":
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as exc:
            raise ImportError(
                "OpenAI chat models require the 'langchain-openai' package. Install with:\n"
                "    pip install langchain-openai"
            ) from exc

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Add it to your .env file or export it before "
                "running with --chat-provider openai."
            )

        model = model_name or "gpt-4o-mini"
        logger.info("Using OpenAI chat model: %s", model)
        return ChatOpenAI(model=model, api_key=api_key, temperature=temperature)

    if provider == "anthropic":
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as exc:
            raise ImportError(
                "Anthropic chat models require the 'langchain-anthropic' package. Install with:\n"
                "    pip install langchain-anthropic"
            ) from exc

        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "ANTHROPIC_API_KEY is not set. Add it to your .env file or export it before "
                "running with --chat-provider anthropic."
            )

        if not model_name:
            raise ValueError(
                "Anthropic requires an explicit --chat-model (current Claude model IDs change "
                "over time — check https://docs.claude.com/en/docs/about-claude/models)."
            )

        logger.info("Using Anthropic chat model: %s", model_name)
        return ChatAnthropic(model=model_name, api_key=api_key, temperature=temperature)

    raise ValueError(f"Unsupported chat provider: '{provider}' (expected 'ollama', 'openai', or 'anthropic')")


# ---------------------------------------------------------------------------
# Checkpointer
# ---------------------------------------------------------------------------

def get_checkpointer(db_path: Path) -> Tuple["SqliteSaver", sqlite3.Connection]:
    """
    Build a SQLite-backed checkpointer so graph state — including any
    in-progress human-in-the-loop interrupt — is durable across process
    restarts. Returns both the checkpointer and the raw connection so the
    caller can close the connection on shutdown.

    Only used by main.py's local CLI path — the web deployment (server.py /
    agent_service.py) uses the Postgres/Redis checkpointer instead, so this
    import is deferred to runtime rather than happening at module load time:
    'langgraph-checkpoint-sqlite' is deliberately NOT in requirements.txt
    (it would be dead weight in the Vercel bundle), so importing it
    unconditionally at the top of this file would break the entire module
    — and therefore the whole deployed app — the moment that package isn't
    installed, even though nothing in the web path ever calls this function.

    Raises:
        ImportError: if 'langgraph-checkpoint-sqlite' isn't installed.
    """
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as exc:
        raise ImportError(
            "The local CLI checkpointer requires the 'langgraph-checkpoint-sqlite' package. "
            "Install with:\n    pip install langgraph-checkpoint-sqlite\n"
            "(This is intentionally NOT in requirements.txt, since the web deployment uses "
            "the Postgres/Redis checkpointer instead and doesn't need it.)"
        ) from exc

    try:
        db_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(f"Could not create directory for checkpoint database '{db_path}': {exc}") from exc

    try:
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
    except sqlite3.Error as exc:
        raise RuntimeError(f"Could not open checkpoint database at '{db_path}': {exc}") from exc

    checkpointer = SqliteSaver(conn)
    return checkpointer, conn


# ---------------------------------------------------------------------------
# Vector store — Pinecone (serverless)
# ---------------------------------------------------------------------------
#
# All configuration is read from os.environ so the exact same code works
# locally (via a .env file, loaded by python-dotenv in ingest.py/server.py)
# and on a hosting platform like Vercel, where you set these as Project
# Environment Variables instead of a file:
#
#   PINECONE_API_KEY    (required) — from app.pinecone.io
#   PINECONE_INDEX_NAME  (optional) — physical Pinecone index name.
#                         Must be lowercase alphanumeric + hyphens only.
#   PINECONE_NAMESPACE   (optional) — logical partition within the index;
#                         lets you keep multiple "collections" (e.g. one per
#                         repo) in a single index without hitting a free-tier
#                         index-count limit. Free-form string.
#   PINECONE_CLOUD       (optional) — serverless cloud, e.g. "aws".
#   PINECONE_REGION      (optional) — serverless region, e.g. "us-east-1".
#                         Check your Pinecone account for which cloud/region
#                         combinations your plan (including the free tier)
#                         currently supports.
#   PINECONE_METRIC      (optional) — distance metric, e.g. "cosine".

DEFAULT_PINECONE_INDEX_NAME = "onboarding-engineer"
DEFAULT_NAMESPACE = "onboarding_docs"
DEFAULT_PINECONE_CLOUD = "aws"
DEFAULT_PINECONE_REGION = "us-east-1"
DEFAULT_PINECONE_METRIC = "cosine"

_INDEX_NAME_PATTERN = re.compile(r"^[a-z0-9-]+$")


def _field(obj, name: str, default=None):
    """Pinecone's SDK response objects have supported both dict-style and
    attribute-style access across versions; handle either defensively
    rather than betting on one."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _pinecone_client():
    try:
        from pinecone import Pinecone
    except ImportError as exc:
        raise ImportError(
            "Pinecone support requires the 'pinecone' and 'langchain-pinecone' packages. Install with:\n"
            "    pip install pinecone langchain-pinecone"
        ) from exc

    api_key = os.environ.get("PINECONE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "PINECONE_API_KEY is not set. Add it to your .env file (locally) or to your hosting "
            "provider's environment variables (e.g. Vercel Project Settings -> Environment Variables)."
        )

    try:
        return Pinecone(api_key=api_key)
    except Exception as exc:
        raise AgentError(f"Failed to initialize the Pinecone client: {exc}") from exc


def _resolve_index_name(index_name: Optional[str]) -> str:
    name = index_name or os.environ.get("PINECONE_INDEX_NAME", DEFAULT_PINECONE_INDEX_NAME)
    if not _INDEX_NAME_PATTERN.match(name):
        raise ValueError(
            f"Invalid Pinecone index name '{name}'. Pinecone index names must contain only "
            "lowercase letters, numbers, and hyphens (e.g. 'onboarding-engineer')."
        )
    return name


def _resolve_namespace(namespace: Optional[str]) -> str:
    return namespace if namespace is not None else os.environ.get("PINECONE_NAMESPACE", DEFAULT_NAMESPACE)


def _infer_embedding_dimension(embedding_function: Embeddings) -> int:
    try:
        vector = embedding_function.embed_query("dimension probe")
    except Exception as exc:
        raise AgentError(f"Could not determine the embedding dimension (a test embed_query call failed): {exc}") from exc
    return len(vector)


def _wait_for_index_ready(pc, index_name: str, timeout_seconds: int = 120, poll_seconds: float = 2.0) -> None:
    waited = 0.0
    while waited < timeout_seconds:
        try:
            desc = pc.describe_index(index_name)
            if _field(_field(desc, "status"), "ready"):
                return
        except Exception:
            pass  # index may not be visible yet immediately after create_index
        time.sleep(poll_seconds)
        waited += poll_seconds
    raise AgentError(f"Pinecone index '{index_name}' did not become ready within {timeout_seconds}s.")


def reset_namespace(index_name: Optional[str] = None, namespace: Optional[str] = None) -> None:
    """
    Delete all vectors in a Pinecone namespace, leaving the index itself
    intact. No-op if the index or namespace doesn't exist yet.
    """
    pc = _pinecone_client()
    resolved_index_name = _resolve_index_name(index_name)
    resolved_namespace = _resolve_namespace(namespace)

    try:
        if not pc.has_index(resolved_index_name):
            logger.debug("Index '%s' doesn't exist yet — nothing to reset.", resolved_index_name)
            return
    except Exception as exc:
        raise AgentError(f"Failed to reach Pinecone while checking for index '{resolved_index_name}': {exc}") from exc

    index = pc.Index(resolved_index_name)
    try:
        index.delete(delete_all=True, namespace=resolved_namespace)
        logger.info("Cleared namespace '%s' in index '%s' (--reset).", resolved_namespace, resolved_index_name)
    except Exception as exc:
        # A namespace that doesn't exist yet raises here in some SDK versions — treat as a no-op.
        logger.debug("Namespace '%s' reset skipped (likely didn't exist yet): %s", resolved_namespace, exc)


def get_vector_store(
    embedding_function: Embeddings,
    *,
    index_name: Optional[str] = None,
    namespace: Optional[str] = None,
    create_if_missing: bool = False,
    require_populated: bool = False,
):
    """
    Open a Pinecone-backed LangChain vector store, creating the serverless
    index first if it doesn't exist and `create_if_missing` is True.

    Raises:
        ImportError: if 'pinecone' / 'langchain-pinecone' aren't installed.
        EnvironmentError: if PINECONE_API_KEY is missing.
        ValueError: if the resolved index name is invalid.
        AgentError: for any other Pinecone-side failure — the index doesn't
        exist and create_if_missing is False, index creation/readiness
        times out, the existing index's dimension doesn't match the
        embedding function in use, or (when require_populated=True) the
        namespace has no vectors in it yet.
    """
    from langchain_pinecone import PineconeVectorStore
    from pinecone import ServerlessSpec

    pc = _pinecone_client()
    resolved_index_name = _resolve_index_name(index_name)
    resolved_namespace = _resolve_namespace(namespace)
    cloud = os.environ.get("PINECONE_CLOUD", DEFAULT_PINECONE_CLOUD)
    region = os.environ.get("PINECONE_REGION", DEFAULT_PINECONE_REGION)
    metric = os.environ.get("PINECONE_METRIC", DEFAULT_PINECONE_METRIC)

    try:
        index_exists = pc.has_index(resolved_index_name)
    except Exception as exc:
        raise AgentError(f"Failed to reach Pinecone while checking for index '{resolved_index_name}': {exc}") from exc

    if not index_exists:
        if not create_if_missing:
            raise AgentError(
                f"Pinecone index '{resolved_index_name}' does not exist. Run ingest.py first — it "
                "creates the index automatically — before starting the server."
            )

        dimension = _infer_embedding_dimension(embedding_function)
        logger.info(
            "Creating Pinecone serverless index '%s' (dimension=%d, metric=%s, cloud=%s, region=%s)...",
            resolved_index_name, dimension, metric, cloud, region,
        )
        try:
            pc.create_index(
                name=resolved_index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud=cloud, region=region),
            )
        except Exception as exc:
            raise AgentError(
                f"Failed to create Pinecone index '{resolved_index_name}': {exc}. If you're on Pinecone's "
                "free tier, double check that your account supports this cloud/region combination "
                "(PINECONE_CLOUD/PINECONE_REGION)."
            ) from exc

        _wait_for_index_ready(pc, resolved_index_name)
    else:
        # Guard against silently querying/writing an existing index with the
        # wrong embedding model — Pinecone indexes have a fixed dimension.
        expected_dim = _infer_embedding_dimension(embedding_function)
        actual_dim = _field(pc.describe_index(resolved_index_name), "dimension")
        if actual_dim is not None and actual_dim != expected_dim:
            raise AgentError(
                f"Pinecone index '{resolved_index_name}' has dimension {actual_dim}, but the current "
                f"embedding provider produces {expected_dim}-dimensional vectors. Use the same "
                "embedding provider/model you originally ingested with, or point to a different index."
            )

    try:
        index = pc.Index(resolved_index_name)
    except Exception as exc:
        raise AgentError(f"Failed to connect to Pinecone index '{resolved_index_name}': {exc}") from exc

    if require_populated:
        try:
            stats = index.describe_index_stats()
            namespaces = _field(stats, "namespaces", {}) or {}
            ns_info = namespaces.get(resolved_namespace) if isinstance(namespaces, dict) else None
            count = _field(ns_info, "vector_count", 0) or 0
        except Exception as exc:
            raise AgentError(f"Failed to read index stats for '{resolved_index_name}': {exc}") from exc

        if count == 0:
            raise AgentError(
                f"Namespace '{resolved_namespace}' in index '{resolved_index_name}' is empty. Run "
                "ingest.py against your repository before starting an onboarding session."
            )
        logger.info(
            "Opened Pinecone index '%s' namespace '%s' (%d vectors).",
            resolved_index_name, resolved_namespace, count,
        )

    return PineconeVectorStore(index=index, embedding=embedding_function, namespace=resolved_namespace)


# ---------------------------------------------------------------------------
# Structured-output helper (robust to local models that don't reliably
# support tool-calling based structured output)
# ---------------------------------------------------------------------------

def _extract_json_object(text: str) -> dict:
    """Pull the first JSON object out of `text`, tolerating markdown code
    fences and trailing commentary."""
    text = text.strip()

    fence_match = re.search(r"```(?:json)?\s*(\{.*)\s*```", text, re.DOTALL)
    candidate = fence_match.group(1) if fence_match else text

    start = candidate.find("{")
    if start == -1:
        raise ValueError("No JSON object found in the model's response.")
    candidate = candidate[start:]

    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(candidate)
    if not isinstance(obj, dict):
        raise ValueError("Parsed JSON was not an object.")
    return obj


def invoke_structured(
    llm,
    system_prompt: str,
    human_prompt: str,
    schema: Type[T],
    max_retries: int = 2,
) -> T:
    """
    Invoke `llm` and coerce its response into `schema`.

    Tries the model's native structured-output support first
    (`with_structured_output`). If that fails or isn't supported, falls
    back to asking for raw JSON and parsing it manually, retrying with an
    explicit correction message on failure.

    Raises:
        AgentError: if no valid structured output could be obtained after
        `max_retries` additional attempts.
    """
    messages: List[BaseMessage] = [SystemMessage(content=system_prompt), HumanMessage(content=human_prompt)]
    last_error: Optional[Exception] = None
    total_attempts = max_retries + 1

    for attempt in range(1, total_attempts + 1):
        try:
            structured_llm = llm.with_structured_output(schema)
            result = structured_llm.invoke(messages)
            if isinstance(result, schema):
                return result
            return schema.model_validate(result)
        except Exception as structured_exc:  # provider/model may not support this path
            last_error = structured_exc
            logger.debug("with_structured_output failed on attempt %d: %s", attempt, structured_exc)

        raw_content = ""
        try:
            raw_response = llm.invoke(messages)
            raw_content = raw_response.content if hasattr(raw_response, "content") else str(raw_response)
            data = _extract_json_object(raw_content)
            return schema.model_validate(data)
        except (ValidationError, ValueError, json.JSONDecodeError) as parse_exc:
            last_error = parse_exc
            logger.warning(
                "Structured output parse attempt %d/%d failed: %s", attempt, total_attempts, parse_exc
            )
            if attempt < total_attempts:
                correction = HumanMessage(
                    content=(
                        "Your previous response could not be parsed as valid JSON matching the "
                        f"required schema:\n{json.dumps(schema.model_json_schema())}\n\n"
                        "Respond with ONLY a single valid JSON object matching this schema — "
                        "no markdown code fences, no explanation, no extra text."
                    )
                )
                messages = messages + [AIMessage(content=raw_content), correction]
        except Exception as invoke_exc:
            # Genuine connection/provider failure — not worth retrying with a
            # correction message since the model never responded.
            last_error = invoke_exc
            logger.warning("LLM invocation failed on attempt %d/%d: %s", attempt, total_attempts, invoke_exc)

    raise AgentError(
        f"Failed to get valid structured output for '{schema.__name__}' after {total_attempts} "
        f"attempt(s). This usually means the LLM provider is unreachable/misconfigured, or the "
        f"model can't reliably follow the required schema. Last error: {last_error}"
    ) from last_error


# ---------------------------------------------------------------------------
# Retrieval helper
# ---------------------------------------------------------------------------

def _format_docs(docs: List[Document]) -> str:
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        parts.append(f"--- Source: {source} ---\n{doc.page_content}")
    return "\n\n".join(parts)


def retrieve_context(vector_store, queries: List[str], k_per_query: int = 4, max_chars: int = 7000) -> str:
    """
    Run several similarity-search queries against the vector store, dedupe
    the results, and return a single context string capped at `max_chars`.

    Raises:
        AgentError: if the vector store can't be queried, or if it returns
        no results at all (most likely meaning ingest.py hasn't been run,
        or was run with a different/incompatible embedding provider).
    """
    seen: set = set()
    docs: List[Document] = []

    for query in queries:
        try:
            results = vector_store.similarity_search(query, k=k_per_query)
        except Exception as exc:
            raise AgentError(
                "Failed to query the vector store. Make sure Pinecone is reachable, the "
                "namespace was ingested, and you're using the same --embedding-provider "
                f"here as you used with ingest.py. Original error: {exc}"
            ) from exc

        for doc in results:
            key = (doc.metadata.get("source"), doc.page_content[:80])
            if key not in seen:
                seen.add(key)
                docs.append(doc)

    if not docs:
        raise AgentError(
            "The vector store returned no results for any query. Run ingest.py against your "
            "repository before starting an onboarding session."
        )

    return _format_docs(docs)[:max_chars]


# ---------------------------------------------------------------------------
# Node 1 — Syllabus Designer Agent
# ---------------------------------------------------------------------------

GENERAL_TOPIC_QUERIES = [
    "overview, purpose, and what this is used for",
    "core concepts, terminology, and how the pieces fit together",
    "getting started — the simplest possible first step",
    "key features and capabilities",
]


def syllabus_designer_node(state: OnboardingState, *, vector_store, llm) -> dict:
    logger.info("Syllabus Designer: retrieving source context...")
    context = retrieve_context(vector_store, GENERAL_TOPIC_QUERIES, k_per_query=4, max_chars=7000)

    system_prompt = (
        "You are the Syllabus Designer for an AI Onboarding Engineer. Your job is to study "
        "excerpts from source material (this could be a software repository's docs and code, "
        "or a documentation/reference site — treat both the same way) and design a focused, "
        "progressive 3-step learning path for someone completely new to this subject.\n\n"
        "Two hard rules:\n"
        "1. The learner has NO access to any repository, file system, or external material — "
        "only this conversation. Never assume they can open, browse, or navigate to anything. "
        "Every concept must be teachable through text alone.\n"
        "2. Start from true fundamentals. Step 1 must be genuinely introductory — assume the "
        "learner has never encountered this subject before, and needs the most basic building "
        "blocks and vocabulary before anything else. Do not open with advanced, niche, or "
        "deep-implementation topics just because they appeared prominently in the retrieved "
        "context. Step 2 builds on step 1, and step 3 is the most advanced of the three — but "
        "'advanced' here still means appropriate early material, not expert-level depth.\n\n"
        "Ground every step strictly in the actual context provided — do not invent concepts "
        "that aren't supported by it."
    )
    human_prompt = (
        f"Source context (retrieved from the documentation/repository):\n\n{context}\n\n"
        "Design exactly 3 learning steps, ordered from most basic to least basic, for someone "
        "brand new to this subject. Respond in the required structured format."
    )

    syllabus = invoke_structured(llm, system_prompt, human_prompt, Syllabus)

    summary_lines = [f"Project summary: {syllabus.project_summary}", "", "Learning path:"]
    for i, step in enumerate(syllabus.steps, start=1):
        summary_lines.append(f"{i}. {step.title} — {step.objective}")

    logger.info("Syllabus Designer: produced a 3-step syllabus.")
    return {
        "repo_summary": syllabus.project_summary,
        "syllabus": syllabus.model_dump(),
        "current_step_index": 0,
        "messages": [AIMessage(content="\n".join(summary_lines))],
    }


# ---------------------------------------------------------------------------
# Node 2 — Lab Generator Agent
# ---------------------------------------------------------------------------

def lab_generator_node(state: OnboardingState, *, vector_store, llm) -> dict:
    syllabus = state.get("syllabus")
    if not syllabus:
        raise AgentError("Lab Generator was invoked before a syllabus existed.")

    idx = state.get("current_step_index", 0)
    steps = syllabus["steps"]
    if idx >= len(steps):
        raise AgentError(f"Lab Generator: step index {idx} is out of range for a {len(steps)}-step syllabus.")

    step = steps[idx]
    logger.info("Lab Generator: building lab for step %d/%d — %s", idx + 1, len(steps), step["title"])

    queries = [step["title"]] + list(step.get("key_concepts", []))
    lab_context = retrieve_context(vector_store, queries, k_per_query=3, max_chars=6000)

    system_prompt = (
        "You are the Lab Generator for an AI Onboarding Engineer. Given one step of a "
        "learner's onboarding syllabus and grounded excerpts from the source material, write "
        "the lesson text, then a single short-answer question that directly tests something "
        "the lesson just covered.\n\n"
        "Hard constraints on the challenge question:\n"
        "- It must be answerable by typing a few sentences (or a short code snippet) into a "
        "plain text box — that is the ONLY way the learner can respond, so never ask for "
        "anything else.\n"
        "- NEVER ask the learner to draw, sketch, diagram, or produce anything visual — they "
        "have no way to submit an image.\n"
        "- NEVER ask for a long essay, a multi-part answer, or a multi-file project. One "
        "focused question, one short answer.\n"
        "- It must test material from the lesson you just wrote, not something tangential.\n\n"
        "The learner has NO access to any repository or file system: never tell them to open, "
        "find, or navigate to a specific file or path. Everything they need — any code, "
        "config, or reference snippet — must be written directly into the lesson or challenge "
        "text itself. Early steps should be simple and confidence-building, not deep or "
        "advanced. Include clear, specific success criteria appropriate for a short answer."
    )
    human_prompt = (
        f"Learning step {idx + 1} of {len(steps)}: {step['title']}\n"
        f"Objective: {step['objective']}\n"
        f"Key concepts to cover: {', '.join(step.get('key_concepts', [])) or 'N/A'}\n\n"
        f"Relevant source context:\n\n{lab_context}\n\n"
        "Create a hands-on exercise for this step. Respond in the required structured format."
    )

    quiz_system_prompt = (
        "You are the Quiz Generator for an AI Onboarding Engineer. Given one step of a "
        "learner's onboarding syllabus and grounded excerpts from the source material, write "
        f"exactly {NUM_QUIZ_QUESTIONS} multiple-choice questions (4 options each, exactly one "
        "correct) that check understanding of this step's material. Questions must be "
        "answerable purely from the provided context and general reasoning — never assume "
        "access to any repository or file. Keep each question clear and unambiguous, with "
        "only one defensibly correct option; wrong options should be plausible, not silly."
    )
    quiz_human_prompt = (
        f"Learning step {idx + 1} of {len(steps)}: {step['title']}\n"
        f"Objective: {step['objective']}\n"
        f"Key concepts to cover: {', '.join(step.get('key_concepts', [])) or 'N/A'}\n\n"
        f"Relevant source context:\n\n{lab_context}\n\n"
        "Write the quiz. Respond in the required structured format."
    )

    workaid_system_prompt = (
        "You are the Work Aid Generator for an AI Onboarding Engineer. Given one step of a "
        "learner's onboarding syllabus and grounded excerpts from the source material, write "
        "a short, scannable reference sheet the learner can consult while working through this "
        "step — key terms and their definitions, important syntax, or a short checklist, "
        "whichever best fits this step's topic. Keep it concise; this is a quick-reference "
        "companion, not a restatement of the full lesson."
    )
    workaid_human_prompt = (
        f"Learning step {idx + 1} of {len(steps)}: {step['title']}\n"
        f"Key concepts to cover: {', '.join(step.get('key_concepts', [])) or 'N/A'}\n\n"
        f"Relevant source context:\n\n{lab_context}\n\n"
        "Write the work aid. Respond in the required structured format."
    )

    # The lab, quiz, and work aid are three independent LLM calls grounded in
    # the same retrieved context — running them concurrently instead of
    # sequentially keeps this node's wall-clock time close to that of a
    # single call rather than the sum of three, which matters for staying
    # under the deployment's function timeout.
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        lab_future = executor.submit(invoke_structured, llm, system_prompt, human_prompt, Lab)
        quiz_future = executor.submit(invoke_structured, llm, quiz_system_prompt, quiz_human_prompt, Quiz)
        workaid_future = executor.submit(
            invoke_structured, llm, workaid_system_prompt, workaid_human_prompt, WorkAid
        )
        lab = lab_future.result()
        quiz = quiz_future.result()
        work_aid = workaid_future.result()

    content = (
        f"Step {idx + 1}/{len(steps)}: {lab.title}\n\n"
        f"{lab.lesson}\n\n"
        f"Your challenge:\n{lab.challenge}\n\n"
    )
    if lab.starter_code.strip():
        content += f"Starter code:\n{lab.starter_code}\n\n"
    content += "You'll be evaluated on:\n" + "\n".join(f"- {c}" for c in lab.success_criteria)

    return {
        "lab": lab.model_dump(),
        "quiz": quiz.model_dump(),
        "work_aid": work_aid.model_dump(),
        "lab_context": lab_context,
        "attempt_count": 0,
        "review_passed": None,
        "review_feedback": None,
        "messages": [AIMessage(content=content)],
    }


# ---------------------------------------------------------------------------
# Node 3 — Reviewer Agent (human-in-the-loop)
# ---------------------------------------------------------------------------

MAX_ATTEMPTS_PER_STEP = 3


def reviewer_node(state: OnboardingState, *, vector_store, llm) -> dict:
    lab = state.get("lab")
    if not lab:
        raise AgentError("Reviewer was invoked before a lab existed.")

    syllabus = state["syllabus"]
    idx = state["current_step_index"]
    total_steps = len(syllabus["steps"])
    attempt_count = state.get("attempt_count", 0) + 1

    interrupt_payload = {
        "type": "lab_submission_request",
        "step": idx + 1,
        "total_steps": total_steps,
        "lab_title": lab["title"],
        "challenge": lab["challenge"],
        "attempt": attempt_count,
        "max_attempts": MAX_ATTEMPTS_PER_STEP,
        "previous_feedback": state.get("review_feedback"),
    }

    # Pauses the graph here; on resume, `interrupt(...)` returns the value
    # passed via Command(resume=...) instead of pausing again.
    user_response = interrupt(interrupt_payload)
    user_response = "" if user_response is None else str(user_response)

    logger.info("Reviewer: evaluating attempt %d/%d for step %d.", attempt_count, MAX_ATTEMPTS_PER_STEP, idx + 1)

    system_prompt = (
        "You are the Reviewer Agent for an AI Onboarding Engineer. You evaluate a learner's "
        "submission against the source material and against the lab's stated success "
        "criteria. Be strict but fair and specific: cite what was correct and what was "
        "missing or wrong, and reference the source context when relevant. Do not pass a "
        "submission that fails to meet the success criteria, even if it is well-written. "
        "Remember the learner has no access to any repository or file system — only what was "
        "given to them in the lesson and challenge text — so never fault them for not knowing "
        "something that wasn't provided to them."
    )
    human_prompt = (
        f"Lab: {lab['title']}\n"
        f"Challenge given to the learner:\n{lab['challenge']}\n\n"
        "Success criteria:\n" + "\n".join(f"- {c}" for c in lab["success_criteria"]) + "\n\n"
        f"Relevant source context to check the submission against:\n\n{state.get('lab_context', '')}\n\n"
        f"Learner's submission:\n{user_response or '(no response submitted)'}\n\n"
        "Evaluate this submission. Respond in the required structured format."
    )

    verdict = invoke_structured(llm, system_prompt, human_prompt, ReviewVerdict)

    feedback_text = f"Passed! {verdict.feedback}" if verdict.passed else f"Not quite. {verdict.feedback}"
    if verdict.unmet_criteria:
        feedback_text += "\n\nStill missing:\n" + "\n".join(f"- {c}" for c in verdict.unmet_criteria)

    messages: List[BaseMessage] = [HumanMessage(content=user_response), AIMessage(content=feedback_text)]

    if verdict.passed:
        completed_record = {"step": idx + 1, "title": lab["title"], "attempts": attempt_count}
        return {
            "attempt_count": attempt_count,
            "review_passed": True,
            "review_feedback": verdict.feedback,
            "current_step_index": idx + 1,
            "completed_steps": state.get("completed_steps", []) + [completed_record],
            "messages": messages,
        }

    if attempt_count >= MAX_ATTEMPTS_PER_STEP:
        messages.append(
            AIMessage(
                content=(
                    f"You've used all {MAX_ATTEMPTS_PER_STEP} attempts for this step. That's "
                    "completely fine — this is exactly what pairing with a teammate is for. "
                    f"Flag '{lab['title']}' in your onboarding channel, and you can restart "
                    "this step anytime."
                )
            )
        )

    return {
        "attempt_count": attempt_count,
        "review_passed": False,
        "review_feedback": verdict.feedback,
        "messages": messages,
    }


# ---------------------------------------------------------------------------
# Routing
# ---------------------------------------------------------------------------

def route_after_review(state: OnboardingState) -> str:
    if state.get("review_passed"):
        total_steps = len(state["syllabus"]["steps"])
        if state["current_step_index"] >= total_steps:
            return "complete"
        return "next_step"

    if state.get("attempt_count", 0) >= MAX_ATTEMPTS_PER_STEP:
        return "escalate"

    return "retry"


# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph(vector_store, llm, checkpointer: SqliteSaver):
    """
    Assemble and compile the onboarding StateGraph:

        START -> syllabus_designer -> lab_generator -> reviewer
                                            ^              |
                                            |   retry ----+---> next_step
                                            +--------------      |
                                                                  v
                                                        complete/escalate -> END
    """
    builder = StateGraph(OnboardingState)

    builder.add_node("syllabus_designer", functools.partial(syllabus_designer_node, vector_store=vector_store, llm=llm))
    builder.add_node("lab_generator", functools.partial(lab_generator_node, vector_store=vector_store, llm=llm))
    builder.add_node("reviewer", functools.partial(reviewer_node, vector_store=vector_store, llm=llm))

    builder.add_edge(START, "syllabus_designer")
    builder.add_edge("syllabus_designer", "lab_generator")
    builder.add_edge("lab_generator", "reviewer")
    builder.add_conditional_edges(
        "reviewer",
        route_after_review,
        {
            "retry": "reviewer",
            "next_step": "lab_generator",
            "complete": END,
            "escalate": END,
        },
    )

    return builder.compile(checkpointer=checkpointer)