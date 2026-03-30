import logging
import os
import time
from pathlib import Path

import torch
from langchain_core.prompts import PromptTemplate
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from guardrails import evaluate_input, evaluate_output
from retriever import Retriever
from tracking import MLflowTracker

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
RETRIEVE_K = 12


PROMPT_TEMPLATE = PromptTemplate.from_template(
    "<|system|>\n{system_prompt}<|end|>\n"
    "<|user|>\n"
    "Context:\n{context}\n\n"
    "Question: {question}<|end|>\n"
    "<|assistant|>\n"
)


def _build_prompt(question: str, context_chunks: list[dict]) -> str:
    context_text = "\n\n".join(
        f"[Source: {c['product']}]\n{c['text']}" for c in context_chunks
    )
    return PROMPT_TEMPLATE.format(
        system_prompt=SYSTEM_PROMPT,
        context=context_text,
        question=question,
    )


class RAGPipeline:
    def __init__(self):
        logger.info("Loading retriever...")
        self.retriever = Retriever()
        self.tracker = MLflowTracker(enabled=True)

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
            self.used_adapter = True
        else:
            logger.warning("LoRA adapter not found. Using base model.")
            self.model = base
            self.used_adapter = False

        self.model.eval()
        logger.info("RAG pipeline ready.")

    def answer(self, question: str) -> str:
        start = time.perf_counter()

        input_decision = evaluate_input(question)
        if not input_decision.allowed:
            logger.info(f"Input guardrail blocked request: {input_decision.reason}")
            self.tracker.log_inference(
                query=question,
                in_domain=False,
                top_dense_score=0.0,
                retrieved_count=0,
                reranked_count=0,
                latency_ms=int((time.perf_counter() - start) * 1000),
                blocked_reason=input_decision.reason,
                used_adapter=self.used_adapter,
                base_model=BASE_MODEL,
            )
            return input_decision.message

        dense_results = self.retriever.retrieve(question, top_k=RETRIEVE_K)
        top_dense_score = dense_results[0]["score"] if dense_results else 0.0
        in_domain = self.retriever.is_in_domain(dense_results)

        if not in_domain:
            logger.info(f"Out-of-domain query rejected: {question}")
            self.tracker.log_inference(
                query=question,
                in_domain=False,
                top_dense_score=top_dense_score,
                retrieved_count=len(dense_results),
                reranked_count=0,
                latency_ms=int((time.perf_counter() - start) * 1000),
                blocked_reason="ood",
                used_adapter=self.used_adapter,
                base_model=BASE_MODEL,
            )
            return REFUSAL_RESPONSE

        results = self.retriever.retrieve_with_rerank(
            question,
            retrieve_k=RETRIEVE_K,
            top_k=TOP_K,
        )
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
        output_decision = evaluate_output(response)
        if not output_decision.allowed:
            logger.info(f"Output guardrail blocked response: {output_decision.reason}")
            self.tracker.log_inference(
                query=question,
                in_domain=True,
                top_dense_score=top_dense_score,
                retrieved_count=len(dense_results),
                reranked_count=len(results),
                latency_ms=int((time.perf_counter() - start) * 1000),
                blocked_reason=output_decision.reason,
                used_adapter=self.used_adapter,
                base_model=BASE_MODEL,
            )
            return output_decision.message

        self.tracker.log_inference(
            query=question,
            in_domain=True,
            top_dense_score=top_dense_score,
            retrieved_count=len(dense_results),
            reranked_count=len(results),
            latency_ms=int((time.perf_counter() - start) * 1000),
            blocked_reason="",
            used_adapter=self.used_adapter,
            base_model=BASE_MODEL,
        )
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
