#!/usr/bin/env python3
"""
ingest.py — Data ingestion pipeline for the AI Onboarding Engineer.

Scans a local repository for Markdown (.md) and Python (.py) files, splits
them into semantically-aware chunks, generates vector embeddings (locally
via sentence-transformers, or remotely via the OpenAI API), and upserts
everything into a Pinecone serverless index for later retrieval. Pinecone
(rather than a local ChromaDB directory) is used specifically so the
retrieval side has no local-disk dependency, which is what lets the agent
backend run as a stateless/serverless deployment (e.g. behind Vercel).

Usage
-----
    # Ingest the current directory using local (offline) embeddings
    python ingest.py --source-dir . --embedding-provider local

    # Ingest a specific repo using OpenAI embeddings (requires OPENAI_API_KEY)
    python ingest.py --source-dir /path/to/repo --embedding-provider openai

    # Wipe and rebuild the namespace from scratch
    python ingest.py --source-dir . --reset

Required environment variables (see agents.py's Pinecone section for the
full list): PINECONE_API_KEY, and optionally PINECONE_INDEX_NAME,
PINECONE_NAMESPACE, PINECONE_CLOUD, PINECONE_REGION.

Requirements
------------
    pip install langchain langchain-community langchain-text-splitters \
                pinecone langchain-pinecone python-dotenv
    # For local embeddings:
    pip install langchain-huggingface sentence-transformers
    # For OpenAI embeddings:
    pip install langchain-openai
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Iterable, List, Optional

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

import agents

# ---------------------------------------------------------------------------
# Configuration defaults
# ---------------------------------------------------------------------------

DEFAULT_SOURCE_DIR = "."
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200
DEFAULT_BATCH_SIZE = 100
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BACKOFF_SECONDS = 2.0

# Extensions we care about and the file "type" tag we attach as metadata.
TARGET_EXTENSIONS = {
    ".md": "markdown",
    ".py": "python",
}

# Directories we never want to walk into (VCS metadata, virtualenvs, caches,
# and build artifacts).
EXCLUDED_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    "node_modules",
    "dist",
    "build",
    ".idea",
    ".vscode",
}

logger = logging.getLogger("ingest")


# ---------------------------------------------------------------------------
# 1. Directory scanning
# ---------------------------------------------------------------------------

def scan_directory(source_dir: Path) -> List[Path]:
    """
    Recursively scan `source_dir` for files with a target extension
    (.md, .py), skipping known noise directories.

    Raises:
        FileNotFoundError: if `source_dir` does not exist.
        NotADirectoryError: if `source_dir` is not a directory.
    """
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory does not exist: {source_dir}")
    if not source_dir.is_dir():
        raise NotADirectoryError(f"Source path is not a directory: {source_dir}")

    matched_files: List[Path] = []

    for root, dirnames, filenames in os.walk(source_dir):
        # Prune excluded directories in-place so os.walk doesn't descend into them.
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIR_NAMES and not d.startswith(".")]

        for filename in filenames:
            ext = Path(filename).suffix.lower()
            if ext in TARGET_EXTENSIONS:
                matched_files.append(Path(root) / filename)

    return sorted(matched_files)


# ---------------------------------------------------------------------------
# 2. Document loading
# ---------------------------------------------------------------------------

def load_documents(file_paths: Iterable[Path], source_dir: Path) -> List[Document]:
    """
    Read each file into a LangChain Document, attaching useful metadata.
    Individual file failures (permission errors, bad encodings, files that
    vanish mid-scan, empty files) are logged and skipped rather than
    aborting the whole run.
    """
    documents: List[Document] = []
    skipped = 0

    for path in file_paths:
        text: Optional[str] = None

        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            logger.warning("UTF-8 decode failed for %s; retrying with latin-1", path)
            try:
                text = path.read_text(encoding="latin-1")
            except (OSError, UnicodeDecodeError) as exc:
                logger.error("Skipping %s — could not decode file: %s", path, exc)
                skipped += 1
                continue
        except FileNotFoundError:
            logger.error("Skipping %s — file disappeared before it could be read.", path)
            skipped += 1
            continue
        except PermissionError:
            logger.error("Skipping %s — permission denied.", path)
            skipped += 1
            continue
        except OSError as exc:
            logger.error("Skipping %s — OS error while reading file: %s", path, exc)
            skipped += 1
            continue

        if text is None or not text.strip():
            logger.debug("Skipping %s — file is empty or whitespace-only.", path)
            skipped += 1
            continue

        ext = path.suffix.lower()
        file_type = TARGET_EXTENSIONS.get(ext, "unknown")

        try:
            relative_source = str(path.relative_to(source_dir))
        except ValueError:
            relative_source = str(path)

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": relative_source,
                    "absolute_path": str(path.resolve()),
                    "file_name": path.name,
                    "file_type": file_type,
                },
            )
        )

    if skipped:
        logger.info("Loaded %d file(s); skipped %d unreadable/empty file(s).", len(documents), skipped)
    else:
        logger.info("Loaded %d file(s).", len(documents))

    return documents


# ---------------------------------------------------------------------------
# 3. Chunking
# ---------------------------------------------------------------------------

def split_documents(
    documents: List[Document],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Document]:
    """
    Split documents into chunks using a language-aware splitter: Python
    files are split respecting function/class boundaries, Markdown files
    are split respecting heading/section structure. Falls back to a plain
    recursive character splitter for anything else.
    """
    python_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.PYTHON,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    markdown_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.MARKDOWN,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    fallback_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    all_chunks: List[Document] = []

    for doc in documents:
        file_type = doc.metadata.get("file_type")

        if file_type == "python":
            splitter = python_splitter
        elif file_type == "markdown":
            splitter = markdown_splitter
        else:
            splitter = fallback_splitter

        try:
            chunks = splitter.split_documents([doc])
        except Exception as exc:  # a malformed file shouldn't kill the whole run
            logger.error(
                "Failed to split %s (%s); falling back to plain splitter: %s",
                doc.metadata.get("source"),
                file_type,
                exc,
            )
            try:
                chunks = fallback_splitter.split_documents([doc])
            except Exception as inner_exc:
                logger.error("Fallback split also failed for %s: %s", doc.metadata.get("source"), inner_exc)
                continue

        for i, chunk in enumerate(chunks):
            if not chunk.page_content.strip():
                continue
            chunk.metadata["chunk_index"] = i
            chunk.metadata["chunk_count"] = len(chunks)
            all_chunks.append(chunk)

    logger.info("Split %d document(s) into %d chunk(s).", len(documents), len(all_chunks))
    return all_chunks


# ---------------------------------------------------------------------------
# 4. Embeddings
# ---------------------------------------------------------------------------

def get_embedding_function(provider: str, model_name: Optional[str]) -> Embeddings:
    """
    Build an Embeddings instance for the requested provider.

    Supported providers:
        "local"  — sentence-transformers via langchain-huggingface (offline,
                   no API key required). Default model: all-MiniLM-L6-v2.
        "openai" — OpenAI's embeddings API. Requires OPENAI_API_KEY.

    Raises:
        ImportError: if the required integration package isn't installed.
        EnvironmentError: if a required API key is missing.
        ValueError: if `provider` is not recognized.
    """
    provider = provider.lower().strip()

    if provider == "local":
        try:
            from langchain_huggingface import HuggingFaceEmbeddings
        except ImportError as exc:
            raise ImportError(
                "Local embeddings require the 'langchain-huggingface' and "
                "'sentence-transformers' packages. Install them with:\n"
                "    pip install langchain-huggingface sentence-transformers"
            ) from exc

        model = model_name or "sentence-transformers/all-MiniLM-L6-v2"
        logger.info("Using local embedding model: %s", model)
        try:
            return HuggingFaceEmbeddings(model_name=model)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load local embedding model '{model}'. If this is the "
                "first run, it may need to download the model weights — check your "
                f"network connection. Original error: {exc}"
            ) from exc

    if provider == "openai":
        try:
            from langchain_openai import OpenAIEmbeddings
        except ImportError as exc:
            raise ImportError(
                "OpenAI embeddings require the 'langchain-openai' package. "
                "Install it with:\n    pip install langchain-openai"
            ) from exc

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Add it to your .env file or export it "
                "in your shell before running with --embedding-provider openai."
            )

        model = model_name or "text-embedding-3-small"
        logger.info("Using OpenAI embedding model: %s", model)
        return OpenAIEmbeddings(model=model, api_key=api_key)

    raise ValueError(f"Unsupported embedding provider: '{provider}' (expected 'local' or 'openai')")


# ---------------------------------------------------------------------------
# 5. Vector store (Pinecone — see agents.py for the connection code)
# ---------------------------------------------------------------------------

def add_documents_with_retry(
    vector_store,
    chunks: List[Document],
    batch_size: int,
    max_retries: int,
    backoff_base: float,
) -> int:
    """
    Upsert chunks into the vector store in batches, retrying transient
    failures (network hiccups, rate limits, temporary embedding-API/Pinecone
    connection errors) with exponential backoff. Raises RuntimeError if a
    batch still fails after all retries are exhausted.

    Each chunk is given an explicit UUID as its Pinecone vector ID (rather
    than relying on an auto-generated one), matching Pinecone's documented
    LangChain integration pattern.
    """
    total = len(chunks)
    added = 0

    for start in range(0, total, batch_size):
        batch = chunks[start : start + batch_size]
        batch_ids = [str(uuid.uuid4()) for _ in batch]
        batch_num = start // batch_size + 1
        total_batches = (total + batch_size - 1) // batch_size

        attempt = 0
        while True:
            attempt += 1
            try:
                vector_store.add_documents(batch, ids=batch_ids)
                added += len(batch)
                logger.info(
                    "Embedded and upserted batch %d/%d (%d chunks, %d/%d total).",
                    batch_num, total_batches, len(batch), added, total,
                )
                break
            except Exception as exc:
                if attempt >= max_retries:
                    raise RuntimeError(
                        f"Batch {batch_num}/{total_batches} failed after {max_retries} attempts "
                        f"(likely an embedding API / Pinecone connection issue): {exc}"
                    ) from exc

                wait = backoff_base * (2 ** (attempt - 1))
                logger.warning(
                    "Batch %d/%d failed (attempt %d/%d): %s — retrying in %.1fs",
                    batch_num, total_batches, attempt, max_retries, exc, wait,
                )
                time.sleep(wait)

    return added


# ---------------------------------------------------------------------------
# CLI / orchestration
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ingest Markdown and Python files from a repo into a Pinecone serverless index.",
    )
    parser.add_argument(
        "--source-dir", type=str, default=DEFAULT_SOURCE_DIR,
        help=f"Directory to scan recursively for .md/.py files (default: {DEFAULT_SOURCE_DIR})",
    )
    parser.add_argument(
        "--index-name", type=str, default=None,
        help=(
            "Pinecone index name (lowercase alphanumeric + hyphens only). Defaults to "
            f"PINECONE_INDEX_NAME or '{agents.DEFAULT_PINECONE_INDEX_NAME}'."
        ),
    )
    parser.add_argument(
        "--namespace", type=str, default=None,
        help=(
            "Pinecone namespace within the index (acts like a collection name). Defaults to "
            f"PINECONE_NAMESPACE or '{agents.DEFAULT_NAMESPACE}'."
        ),
    )
    parser.add_argument(
        "--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
        help=f"Target chunk size in characters (default: {DEFAULT_CHUNK_SIZE})",
    )
    parser.add_argument(
        "--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP,
        help=f"Chunk overlap in characters (default: {DEFAULT_CHUNK_OVERLAP})",
    )
    parser.add_argument(
        "--embedding-provider", type=str, default="local", choices=["local", "openai"],
        help="Embedding backend to use (default: local, offline, no API key required)",
    )
    parser.add_argument(
        "--embedding-model", type=str, default=None,
        help="Override the default embedding model name for the chosen provider.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=DEFAULT_BATCH_SIZE,
        help=f"Number of chunks embedded/stored per batch (default: {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument(
        "--max-retries", type=int, default=DEFAULT_MAX_RETRIES,
        help=f"Max retries per batch on embedding/connection failure (default: {DEFAULT_MAX_RETRIES})",
    )
    parser.add_argument(
        "--reset", action="store_true",
        help="Delete all vectors in this namespace before ingesting.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable debug-level logging.",
    )
    return parser.parse_args(argv)


def configure_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    configure_logging(args.verbose)
    load_dotenv()

    start_time = time.monotonic()
    source_dir = Path(args.source_dir).expanduser().resolve()

    logger.info("Scanning '%s' for .md/.py files...", source_dir)
    try:
        file_paths = scan_directory(source_dir)
    except (FileNotFoundError, NotADirectoryError) as exc:
        logger.error(str(exc))
        return 1

    if not file_paths:
        logger.warning("No .md or .py files found under '%s'. Nothing to ingest.", source_dir)
        return 0

    md_count = sum(1 for p in file_paths if p.suffix.lower() == ".md")
    py_count = sum(1 for p in file_paths if p.suffix.lower() == ".py")
    logger.info("Found %d file(s): %d Markdown, %d Python.", len(file_paths), md_count, py_count)

    documents = load_documents(file_paths, source_dir)
    if not documents:
        logger.error("No documents could be loaded (all files were empty, unreadable, or inaccessible).")
        return 1

    chunks = split_documents(documents, args.chunk_size, args.chunk_overlap)
    if not chunks:
        logger.error("Document splitting produced zero chunks — nothing to embed.")
        return 1

    try:
        embedding_function = get_embedding_function(args.embedding_provider, args.embedding_model)
    except (ImportError, EnvironmentError, ValueError, RuntimeError) as exc:
        logger.error("Could not initialize embedding function: %s", exc)
        return 1

    if args.reset:
        try:
            agents.reset_namespace(index_name=args.index_name, namespace=args.namespace)
        except (ImportError, EnvironmentError, agents.AgentError) as exc:
            logger.error("Could not reset namespace: %s", exc)
            return 1

    try:
        vector_store = agents.get_vector_store(
            embedding_function,
            index_name=args.index_name,
            namespace=args.namespace,
            create_if_missing=True,
        )
    except (ImportError, EnvironmentError, ValueError, agents.AgentError) as exc:
        logger.error("Could not initialize vector store: %s", exc)
        return 1

    logger.info(
        "Embedding %d chunk(s) in batches of %d using the '%s' provider...",
        len(chunks), args.batch_size, args.embedding_provider,
    )
    try:
        added = add_documents_with_retry(
            vector_store=vector_store,
            chunks=chunks,
            batch_size=args.batch_size,
            max_retries=args.max_retries,
            backoff_base=DEFAULT_RETRY_BACKOFF_SECONDS,
        )
    except RuntimeError as exc:
        logger.error(
            "Ingestion aborted while embedding chunks — this usually means the embedding "
            "API/service or Pinecone is unreachable or misconfigured: %s", exc,
        )
        return 1

    elapsed = time.monotonic() - start_time
    index_name = args.index_name or os.environ.get("PINECONE_INDEX_NAME", agents.DEFAULT_PINECONE_INDEX_NAME)
    namespace = args.namespace or os.environ.get("PINECONE_NAMESPACE", agents.DEFAULT_NAMESPACE)
    logger.info(
        "Done. Ingested %d chunk(s) from %d file(s) into Pinecone index '%s' namespace '%s' in %.1fs.",
        added, len(documents), index_name, namespace, elapsed,
    )
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        sys.exit(130)
    except Exception:
        logger.exception("Unhandled error during ingestion.")
        sys.exit(1)