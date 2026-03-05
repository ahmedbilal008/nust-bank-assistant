import logging
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.WARNING)

MODEL = "microsoft/Phi-3.5-mini-instruct"

print("Downloading tokenizer...")
AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
print("Tokenizer done.")

print("Downloading model weights (this is the big one)...")
AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True)
print("Model fully cached. Ready for inference.")
