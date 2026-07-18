"""Q5.1 — Gradio web service for variety-aware Sarcasm detection.

Uses Qwen2.5-1.5B + LoRA adapters with hot-swap per English variety.
Run: uv run python deployment/app.py
"""

import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Prompt (matching Q2.3 notebook)
SARCASM_PROMPT = """Determine whether the following text is sarcastic or not. Reply with only 'sarcastic' or 'not sarcastic'.

Text: {text}
Answer:"""

# Global state for lazy loading
_tokenizer = None
_base_model = None
_adapters = {}

# Use best seed per variety (based on within-variety Macro-F1)
ADAPTER_PATHS = {
    "en-UK": "Dush91/besstie-sarcasm-lora-en-UK_seed123",   # 0.71 vs 0.48 (seed42 collapsed)
    "en-AU": "Dush91/besstie-sarcasm-lora-en-AU_seed42",    # 0.74 (both seeds similar)
    "en-IN": "Dush91/besstie-sarcasm-lora-en-IN_seed42",    # 0.60 vs 0.48
}

# Logit-based classification (tokens after "Answer: ")
SARCASTIC_TOKEN = 82267   # " sarcast"
NOT_SARCASTIC_TOKEN = 537  # " not"


def get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, padding_side="left")
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
    return _tokenizer


def get_base_model():
    global _base_model
    if _base_model is None:
        _base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.float16, device_map="auto"
        )
    return _base_model


def get_adapter(variety):
    if variety not in _adapters:
        base = get_base_model()
        path = ADAPTER_PATHS[variety]
        _adapters[variety] = PeftModel.from_pretrained(base, path).to(DEVICE)
    return _adapters[variety]


def predict_sarcasm(text, variety):
    tokenizer = get_tokenizer()
    model = get_adapter(variety)

    prompt = SARCASM_PROMPT.format(text=text)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=256)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # last token position
        sarc_logit = logits[SARCASTIC_TOKEN].item()
        not_logit = logits[NOT_SARCASTIC_TOKEN].item()
        return "Sarcastic" if sarc_logit > not_logit else "Not Sarcastic"


def classify(text, variety):
    if not text.strip():
        return "Please enter some text."
    result = predict_sarcasm(text, variety)
    return f"Prediction: **{result}**  \n*({variety} adapter)*"


demo = gr.Blocks(title="Sarcasm Detection across English Varieties")

with demo:
    gr.Markdown(
        """
        # Sarcasm Detection across English Varieties
        Enter text and select the English variety. Uses a LoRA-adapted
        Qwen2.5-1.5B with hot-swapping adapters per variety.
        """
    )

    with gr.Row():
        text_input = gr.Textbox(
            label="Input Text",
            placeholder="Type a sentence to check for sarcasm...",
            lines=3,
        )

    with gr.Row():
        variety_dropdown = gr.Dropdown(
            choices=["en-UK", "en-AU", "en-IN"],
            value="en-UK",
            label="English Variety",
        )
        submit_btn = gr.Button("Detect Sarcasm", variant="primary")

    output = gr.Markdown(label="Result")

    submit_btn.click(
        fn=classify, inputs=[text_input, variety_dropdown], outputs=output
    )
    text_input.submit(
        fn=classify, inputs=[text_input, variety_dropdown], outputs=output
    )

    gr.Examples(
        examples=[
            ["Oh great, another meeting. Just what I needed.", "en-UK"],
            ["Yeah right, like that's ever going to happen.", "en-AU"],
            ["What a wonderful day to be stuck in traffic.", "en-IN"],
            ["Love it when the wifi goes down during a deadline.", "en-UK"],
            ["Thanks so much for holding the door. Really helpful.", "en-AU"],
        ],
        inputs=[text_input, variety_dropdown],
    )

if __name__ == "__main__":
    demo.launch()
