import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag_pipeline import RAGPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

_pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    logger.info("Loading RAG pipeline...")
    _pipeline = RAGPipeline()
    logger.info("Pipeline ready.")
    yield
    _pipeline = None


app = FastAPI(
    title="NUST Bank Intelligent Customer Assistant",
    version="1.0.0",
    lifespan=lifespan,
)


class ChatRequest(BaseModel):
    query: str


class ChatResponse(BaseModel):
    answer: str


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest) -> ChatResponse:
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query must not be empty.")
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    answer = _pipeline.answer(request.query.strip())
    return ChatResponse(answer=answer)


@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": _pipeline is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
