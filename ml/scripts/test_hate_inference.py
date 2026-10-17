"""
test_hate_inference.py

Loads the fine-tuned checkpoint from train_hate_model.py and runs it on a
few example sentences - implicit stereotyping without slurs (per the
project's actual scope) and plain neutral text - before wiring it into
hate_service.py and the live API.

    python test_hate_inference.py
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "hate-bert")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "hate-bert-v2")
TEST_SENTENCES = [
    "People from that community are always causing trouble and can't be trusted.",
    "The city council held its monthly meeting on Tuesday to discuss the new budget.",
    "Women are just too emotional to handle leadership roles.",
    "Local schools will reopen next week after the holiday break.",
    "if english is not imposition then hindi is also not imposition shut up chutiya retards stophindiimposition",  # verbatim from HateXplain's own training set
    "People from that community are always causing trouble and can't be trusted.",
    "The city council held its monthly meeting on Tuesday to discuss the new budget.",
    "Women are just too emotional to handle leadership roles.",
    "Local schools will reopen next week after the holiday break.",
]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(device)
    model.eval()

    for text in TEST_SENTENCES:
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]
        label = "FLAGGED" if probs[1] > probs[0] else "NOT FLAGGED"
        confidence = max(probs).item()

        print(f"\n[{label}] (confidence: {confidence:.2f})")
        print(f"  {text}")


if __name__ == "__main__":
    main()