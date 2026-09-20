"""
test_inference.py

Loads the fine-tuned checkpoint from train_bias_model.py and runs it on a
few example sentences, to sanity-check the saved model actually works
before wiring it into bias_service.py and the live API.

    python test_inference.py
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "bias-bert")

TEST_SENTENCES = [
    "The minister slammed the so-called radical opposition in a shocking outburst.",
    "The city council approved the new budget in a 5-2 vote on Tuesday.",
    "Extremist protesters violently disrupted the peaceful gathering, witnesses say.",
    "The report states unemployment fell by 0.3 percent last quarter.",
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
        label = "BIASED" if probs[1] > probs[0] else "NOT BIASED"
        confidence = max(probs).item()

        print(f"\n[{label}] (confidence: {confidence:.2f})")
        print(f"  {text}")


if __name__ == "__main__":
    main()