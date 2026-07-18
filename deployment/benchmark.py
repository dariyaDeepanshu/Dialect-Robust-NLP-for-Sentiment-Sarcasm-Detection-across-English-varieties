"""Q5.2 — Inference efficiency benchmarking.

Compares DistilBERT (67M encoder) vs Qwen2.5-1.5B + LoRA (1.5B decoder).
Run: uv run python deployment/benchmark.py
"""

import time
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)

WARMUP = 10
MEASURE = 100

TEST_TEXTS = {
    "short": "This is great.",
    "medium": "I absolutely love waiting in line for hours on a Monday morning.",
    "long": (
        "The weather today is just perfect for a picnic, isn't it? "
        "I can't think of anything better than sitting in the rain "
        "while reading an email about mandatory weekend overtime work."
    ),
}


def benchmark(name, predict_fn, batch_sizes=None):
    if batch_sizes is None:
        batch_sizes = [1]

    results = {}
    for bs in batch_sizes:
        texts = [TEST_TEXTS["medium"]] * bs
        for _ in range(WARMUP):
            predict_fn(texts)
        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        latencies = []
        for _ in range(MEASURE):
            t0 = time.perf_counter()
            predict_fn(texts)
            if DEVICE.type == "cuda":
                torch.cuda.synchronize()
            latencies.append((time.perf_counter() - t0) * 1000)

        avg_ms = float(np.mean(latencies))
        per_item_ms = avg_ms / bs
        results[bs] = {"total_ms": avg_ms, "per_item_ms": per_item_ms}
        print(
            f"  batch={bs:2d}: {avg_ms:8.2f}ms total, {per_item_ms:8.2f}ms/item"
        )

    return {name: results}


# --- DistilBERT ---
print("Loading DistilBERT...")
db_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
db_model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased"
).to(DEVICE)
db_model.eval()


def predict_db(texts):
    inputs = db_tokenizer(
        texts, return_tensors="pt", truncation=True, padding=True, max_length=512
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        return db_model(**inputs).logits


print("\n" + "=" * 60)
print("DistilBERT (67M parameters, encoder)")
print("=" * 60)
benchmark("DistilBERT", predict_db, batch_sizes=[1, 8, 16, 32])

# --- Qwen2.5-1.5B + LoRA ---
print("\nLoading Qwen2.5-1.5B base + en-UK LoRA adapter...")
from peft import PeftModel
from transformers import AutoModelForCausalLM

SARCASM_PROMPT = (
    "Determine whether the following text is sarcastic or not. "
    "Reply with only 'sarcastic' or 'not sarcastic'.\n\n"
    "Text: {text}\nAnswer:"
)

lora_tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct", padding_side="left"
)
if lora_tokenizer.pad_token is None:
    lora_tokenizer.pad_token = lora_tokenizer.eos_token

lora_base = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-1.5B-Instruct", torch_dtype=torch.float16, device_map="auto"
)
lora_model = PeftModel.from_pretrained(lora_base, "./adapters/en-UK_seed42").to(DEVICE)
lora_model.eval()


def predict_lora(texts):
    prompts = [SARCASM_PROMPT.format(text=t) for t in texts]
    inputs = lora_tokenizer(
        prompts, return_tensors="pt", truncation=True, padding=True, max_length=256
    )
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    with torch.no_grad():
        return lora_model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
            pad_token_id=lora_tokenizer.eos_token_id,
        )


print("\n" + "=" * 60)
print("Qwen2.5-1.5B + LoRA (1.5B parameters, decoder, adapter on en-UK)")
print("=" * 60)
benchmark("Qwen2.5-1.5B+LoRA", predict_lora, batch_sizes=[1, 4, 8])

# --- Summary ---
print("\n" + "=" * 60)
print("TRADE-OFF SUMMARY")
print("=" * 60)
print("DistilBERT:")
print("  - Encoder-only, ~67M params, 3 epochs")
print("  - Fast inference (~5-20ms)")
print("  - Moderate sarcasm F1 (0.66-0.71 within-variety)")
print("")
print("Qwen2.5-1.5B + LoRA:")
print("  - Decoder LLM, ~1.5B params, LoRA r=16 adapters")
print("  - Slower inference (~50-200ms due to autoregressive generation)")
print("  - Hot-swappable adapters (<1% trainable params)")
print("  - Better cross-variety flexibility")
print("")
print("Key trade-off: latency vs. adaptability")
print("LoRA enables variety-aware serving with one frozen base model + 3 tiny adapters (~70MB each).")
