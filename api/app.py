import json
import logging
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException, Request
from openai import OpenAI
from pydantic import BaseModel, Field

from agent.agent import runAgent
from agent.nomad import loadStore, unverifiedQuotes
from api.auth import requireApiKey
from api.sessions import SessionStore, trimToLastTurns
from constants import EMBEDDING_MODEL
from observability.context import newRequestId, startRequestStats
from observability.logging_config import setupLogging
from retrieval.embeddings import embedTexts

load_dotenv()
setupLogging()
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MAX_TURNS = int(os.environ.get("MAX_TURNS", "10"))


def vectorCount(store) -> int:
    return store.index.index.ntotal if store.index is not None else 0


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail closed: without a key the protected routes would reject everything
    if not os.environ.get("API_KEY"):
        raise RuntimeError("API_KEY is not set")

    app.state.store = loadStore()  # serving set: local copy, or downloaded from S3
    catalog = app.state.store.catalog
    
    embedTexts(["warmup"], EMBEDDING_MODEL, kind="query")

    logger.info("Startup complete: %s listings in %s (snapshot %s), %s vectors",
                len(catalog.listings), catalog.city, catalog.snapshot_date,
                vectorCount(app.state.store))
    
    app.state.client = OpenAI(api_key=OPENAI_API_KEY)
    app.state.sessions = SessionStore(
        ttl_seconds=int(os.environ.get("SESSION_TTL_SECONDS", "1800")),
        max_sessions=int(os.environ.get("MAX_SESSIONS", "500")),
    )
    yield

    app.state.client.close()

app = FastAPI(lifespan=lifespan)


@app.middleware("http")
async def addRequestId(request: Request, call_next):
    # Every log line of this request carries the id; the client gets it back
    # in the header to report a problem
    request_id = newRequestId()
    startRequestStats()  # every model call of this request adds to it (observability/llm.py)
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


class AgentRequest(BaseModel):
    message: str = Field(min_length=1,max_length=2000)
    session_id: str | None = None


@app.get("/health")
def healthcheck(http: Request) -> dict:
    store = http.app.state.store
    return {
        "status": "alive",
        "vectors": vectorCount(store),
        "city": store.catalog.city,
        "listings": len(store.catalog.listings),
        "snapshot_date": store.catalog.snapshot_date,
    }


def toolOutputs(items: list) -> list[dict]:
    """Tool outputs of this turn: what the answer may quote from."""
    return [json.loads(item["output"]) for item in items
            if item.get("type") == "function_call_output"]


@app.post("/agent/run", dependencies=[Depends(requireApiKey)])
def askAgent(request: AgentRequest, http: Request) -> dict:
    store = http.app.state.store
    client = http.app.state.client
    sessions = http.app.state.sessions

    if request.session_id:
        context = sessions.get(request.session_id)
        if context is None:
            raise HTTPException(status_code=404, detail="Session not found or expired")
        session_id = request.session_id
    else:
        session_id, context = sessions.create()

    previous_context = list(context)

    context.append({"role": "user", "content": request.message})
    trimToLastTurns(context, MAX_TURNS)
    turn_start = len(context)

    try:
        answer = runAgent(context, client, store, 5)
    except Exception:
        logger.exception("Agent execution failed session_id=%s", session_id)
        context.clear()
        context.extend(previous_context)

        raise HTTPException(status_code=500, detail="Agent execution failed")

    response = {"session_id": session_id, "answer": answer}
    # Citations the model made up or edited are reported, never silently passed on
    unverified = unverifiedQuotes(answer, toolOutputs(list(context)[turn_start:]))
    if unverified:
        logger.warning("unverified quotes session_id=%s quotes=%s", session_id, unverified)
        response["warnings"] = {"unverified_quotes": unverified}
    return response


@app.get("/agent/sessions/{session_id}", dependencies=[Depends(requireApiKey)])
def getSessionHistory(session_id: str, http: Request, include_tools: bool = True) -> dict:
    context = http.app.state.sessions.get(session_id)
    if context is None:
        raise HTTPException(status_code=404, detail="Session not found or expired")

    context = list(context)

    if not include_tools:
        context = [
            item for item in context if item.get("role") in ("user", "assistant")
        ]

    return {"session_id": session_id, "history": context}
