import json
import logging
import os
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHUNKS_PATH = DATA_DIR / "chunks.json"
INDEX_PATH = DATA_DIR / "faiss.index"
CHUNKS_STORE_PATH = DATA_DIR / "chunks_store.pkl"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
SIMILARITY_THRESHOLD = 0.35


class Retriever:
    def __init__(self, index_path: Path = INDEX_PATH, chunks_path: Path = CHUNKS_STORE_PATH):
        self.model = SentenceTransformer(EMBED_MODEL)
        self.index = faiss.read_index(str(index_path))
        with open(chunks_path, "rb") as f:
            self.chunks = pickle.load(f)
        logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")

    def _embed(self, texts: list[str]) -> np.ndarray:
        embs = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return embs.astype("float32")

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        q_emb = self._embed([query])
        distances, indices = self.index.search(q_emb, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            similarity = float(dist)  # inner product = cosine sim when normalized
            results.append({
                "text": self.chunks[idx]["text"],
                "source": self.chunks[idx].get("source", ""),
                "product": self.chunks[idx].get("product", ""),
                "score": similarity,
            })
        return results

    def is_in_domain(self, results: list[dict]) -> bool:
        if not results:
            return False
        return results[0]["score"] >= SIMILARITY_THRESHOLD


def build_index(chunks_path: Path = CHUNKS_PATH) -> None:
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)

    texts = [c["text"] for c in chunks]
    logger.info(f"Embedding {len(texts)} chunks...")

    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    embeddings = embeddings.astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # inner product with normalized vectors = cosine
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_PATH))
    logger.info(f"FAISS index saved to {INDEX_PATH}")

    with open(CHUNKS_STORE_PATH, "wb") as f:
        pickle.dump(chunks, f)
    logger.info(f"Chunks store saved to {CHUNKS_STORE_PATH}")


def test_retrieval() -> None:
    retriever = Retriever()
    test_queries = [
        "What is the daily transfer limit?",
        "How do I apply for auto finance?",
        "Is there an account for women?",
        "Tell me about the stock market",
        "What is the weather today?",
        "How to reset MPIN?",
    ]
    for query in test_queries:
        results = retriever.retrieve(query, top_k=3)
        in_domain = retriever.is_in_domain(results)
        top_score = results[0]["score"] if results else 0.0
        print(f"\nQuery: {query}")
        print(f"  In-domain: {in_domain} | Top score: {top_score:.4f}")
        if results:
            print(f"  Top match: {results[0]['text'][:120]}...")


if __name__ == "__main__":
    build_index()
    test_retrieval()
