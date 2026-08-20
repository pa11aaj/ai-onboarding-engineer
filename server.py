"""
server.py — FastAPI web backend for the AI Onboarding Engineer.

A thin HTTP layer over agent_service.py: creates onboarding sessions and
advances them by resuming the human-in-the-loop LangGraph interrupt with
the learner's submission.

Run locally:
    uvicorn server:app --reload --port 8000

Deployment: with CHECKPOINTER=postgres (or redis) and PINECONE_* set, this
has no per-instance state at all and runs fine as a Vercel Python
serverless function — every request reads/writes the same external
checkpointer and vector store, so it doesn't matter which instance handles
it. With the default CHECKPOINTER=memory, session state lives only in this
process's RAM, which only works for a single long-running process (e.g.
`uvicorn` on your own machine or a host like Render/Fly/Railway) — see
agent_service.py's module docstring for why.

Environment variables (see .env.example):
    PINECONE_API_KEY, PINECONE_INDEX_NAME, PINECONE_NAMESPACE,
    PINECONE_CLOUD, PINECONE_REGION, EMBEDDING_PROVIDER, EMBEDDING_MODEL,
    CHAT_PROVIDER, CHAT_MODEL, OPENAI_API_KEY / ANTHROPIC_API_KEY,
    CHECKPOINTER (memory | postgres | redis), DATABASE_URL (postgres),
    REDIS_URL (redis), FRONTEND_ORIGINS
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import agent_service
from agents import AgentError

logger = logging.getLogger("server")


@asynccontextmanager
async def lifespan(app: FastAPI):
    agent_service.initialize()
    yield
    agent_service.shutdown()


app = FastAPI(title="AI Onboarding Engineer API", version="1.0.0", lifespan=lifespan)

_origins_env = os.environ.get("FRONTEND_ORIGINS", "*").strip()
_allowed_origins = ["*"] if _origins_env == "*" else [o.strip() for o in _origins_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class AnswerRequest(BaseModel):
    answer: str


@app.get("/api/health")
def health():
    ok = agent_service.STARTUP_ERROR is None
    return {"status": "ok" if ok else "degraded", "detail": agent_service.STARTUP_ERROR}


@app.post("/api/session")
def create_session():
    try:
        return agent_service.start_session()
    except AgentError as exc:
        logger.error("Failed to start session: %s", exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/api/session/{thread_id}/answer")
def answer_session(thread_id: str, body: AnswerRequest):
    try:
        return agent_service.submit_answer(thread_id, body.answer)
    except agent_service.SessionNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except AgentError as exc:
        logger.error("Failed to advance session %s: %s", thread_id, exc)
        raise HTTPException(status_code=503, detail=str(exc)) from exc