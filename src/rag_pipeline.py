import logging
import os
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from retriever import Retriever

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT / "models"
ADAPTER_DIR = MODELS_DIR / "phi3-lora-adapter"
BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"

SYSTEM_PROMPT = (
    "You are a helpful, caring, and professional customer support assistant for NUST Bank. "
    "Answer only based on the provided context. "
    "If the question is unrelated to NUST Bank products or services, politely decline to answer."
)

REFUSAL_RESPONSE = (
    "I'm sorry, I can only assist with NUST Bank-related queries. "
    "Please ask me about our products, services, or account features."
)

MAX_NEW_TOKENS = 256
TEMPERATURE = 0.3
TOP_K = 5


def _build_prompt(question: str, context_chunks: list[dict]) -> str:
    context_text = "\n\n".join(
        f"[Source: {c['product']}]\n{c['text']}" for c in context_chunks
    )
    return (
        f"<|system|>\n{SYSTEM_PROMPT}<|end|>\n"
        f"<|user|>\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {question}<|end|>\n"
        f"<|assistant|>\n"
    )


class RAGPipeline:
    def __init__(self):
        logger.info("Loading retriever...")
        self.retriever = Retriever()

        logger.info("Loading language model...")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")

        load_kwargs = dict(attn_implementation='eager')
        if self.device == "cuda":
            load_kwargs["dtype"] = torch.float16
            load_kwargs["device_map"] = "auto"
        else:
            # bfloat16 on CPU: ~7.6 GB RAM instead of ~15 GB (float32)
            load_kwargs["dtype"] = torch.bfloat16
            # No device_map on CPU — causes meta-tensor issues with PEFT

        base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **load_kwargs)

        if ADAPTER_DIR.exists():
            logger.info(f"Loading LoRA adapter from {ADAPTER_DIR}")
            self.model = PeftModel.from_pretrained(base, str(ADAPTER_DIR))
        else:
            logger.warning("LoRA adapter not found. Using base model.")
            self.model = base

        self.model.eval()
        logger.info("RAG pipeline ready.")

    def answer(self, question: str) -> str:
        results = self.retriever.retrieve(question, top_k=TOP_K)

        if not self.retriever.is_in_domain(results):
            logger.info(f"Out-of-domain query rejected: {question}")
            return REFUSAL_RESPONSE

        prompt = _build_prompt(question, results[:3])
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                temperature=TEMPERATURE,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.convert_tokens_to_ids("<|end|>"),
            )

        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        response = self.tokenizer.decode(generated, skip_special_tokens=True)
        return response.strip()


def run_interactive() -> None:
    pipeline = RAGPipeline()

    print("\n=== NUST Bank Intelligent Customer Assistant ===")
    print("Type 'quit' to exit.\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        answer = pipeline.answer(query)
        print(f"\nAssistant: {answer}\n")


if __name__ == "__main__":
    run_interactive()
