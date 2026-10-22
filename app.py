from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel, Field
from rag import loadOrBuildIndex
from openai import OpenAI
from agent import runAgent
from data import HOTELS
from dotenv import load_dotenv
import os
import logging
from constants import *
from contextlib import asynccontextmanager
from logging_config import setupLogging
from sessions import SessionStore, trimToLastTurns
from auth import requireApiKey

load_dotenv()
setupLogging()
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
MAX_TURNS = int(os.environ.get("MAX_TURNS", "10"))

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail closed: without a key the protected routes would reject everything
    if not os.environ.get("API_KEY"):
        raise RuntimeError("API_KEY is not set")

    app.state.v_store = loadOrBuildIndex()
    logger.info("Startup complete: %s vectors loaded", app.state.v_store.index.ntotal)
    app.state.client = OpenAI(api_key=OPENAI_API_KEY)
    app.state.sessions = SessionStore(
        ttl_seconds=int(os.environ.get("SESSION_TTL_SECONDS", "1800")),
        max_sessions=int(os.environ.get("MAX_SESSIONS", "500")),
    )
    yield

    app.state.client.close()

app = FastAPI(lifespan=lifespan)


class AgentRequest(BaseModel):
    message: str = Field(min_length=1,max_length=2000)
    session_id: str | None = None


@app.get("/health")
def healthcheck(http: Request) -> dict:
    return {"status": "alive", "vectors": http.app.state.v_store.index.ntotal}



@app.post("/agent/run", dependencies=[Depends(requireApiKey)])
def askAgent(request: AgentRequest, http: Request) -> dict:
    v_store = http.app.state.v_store 
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

    try:
        answer = runAgent(context, client, v_store, 5)

        return {"session_id": session_id, "answer": answer}

    except Exception:
        logger.exception("Agent execution failed session_id=%s", session_id)
        context.clear()
        context.extend(previous_context)

        raise HTTPException(status_code=500, detail="Agent execution failed")


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
