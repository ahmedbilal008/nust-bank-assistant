import argparse
import logging
import os
import sys

# Ensure the parent directory is in the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from pyngrok import ngrok, conf

from src.rag_pipeline import RAGPipeline

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define the FastAPI app
app = FastAPI(
    title="NUST Bank RAG API",
    description="RAG Pipeline served via Ngrok from Google Colab",
    version="1.0"
)

# Initialize pipeline lazily to save memory during startup/docs building
pipeline = None

class QueryRequest(BaseModel):
    query: str

class QueryResponse(BaseModel):
    answer: str

@app.on_event("startup")
async def startup_event():
    global pipeline
    logger.info("Initializing RAG Pipeline...")
    pipeline = RAGPipeline()
    logger.info("RAG Pipeline initialized successfully.")

@app.post("/chat", response_model=QueryResponse)
async def chat(request: QueryRequest):
    if not pipeline:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        answer = pipeline.answer(request.query)
        return QueryResponse(answer=answer)
    except Exception as e:
        logger.error(f"Error generating answer: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health():
    return {"status": "ok", "pipeline_loaded": pipeline is not None}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Serve RAG Pipeline via ngrok")
    parser.add_argument("--port", type=int, default=8000, help="Port to run FastAPI on")
    parser.add_argument("--ngrok-token", type=str, required=True, help="Your ngrok authtoken")
    parser.add_argument("--region", type=str, default="us", help="ngrok region (e.g. us, eu, ap)")
    
    args = parser.parse_args()
    
    # Configure ngrok
    logger.info("Setting up ngrok...")
    conf.get_default().auth_token = args.ngrok_token
    conf.get_default().region = args.region
    
    # Open ngrok tunnel
    public_url = ngrok.connect(args.port).public_url
    logger.info(f"===============================================================")
    logger.info(f"🚀 Public URL: {public_url}")
    logger.info(f"API Docs available at: {public_url}/docs")
    logger.info(f"===============================================================")
    
    # Run uvicorn server
    uvicorn.run(app, host="0.0.0.0", port=args.port)
