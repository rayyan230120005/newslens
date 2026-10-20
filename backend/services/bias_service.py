"""
services/bias_service.py

Stage 3, real version. Loads the BERT model fine-tuned on BABE (see
ml/scripts/train_bias_model.py) once, at import time, then scores new text
at inference time. This replaces the earlier placeholder lexicon heuristic.

The route (routes/bias.py) and schema (BiasScoreResponse) are unchanged -
only what's inside score_bias() changed, exactly as planned when the
placeholder was first built.

Model checkpoint expected at ../../models/bias-bert relative to this file
(i.e. newslens/models/bias-bert) - that's where train_bias_model.py saved
it. If this file fails to import with a "can't find model" error, the
checkpoint isn't where this expects - check that path first.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models", "bias-bert")

_device = "cuda" if torch.cuda.is_available() else "cpu"
_tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
_model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(_device)
_model.eval()


def score_bias(text: str) -> dict:
    inputs = _tokenizer(text, return_tensors="pt", truncation=True, max_length=128).to(_device)
    with torch.no_grad():
        logits = _model(**inputs).logits
    probs = torch.softmax(logits, dim=-1)[0]

    bias_score = probs[1].item()  # model's probability that this is biased

    return {
        "bias_score": round(bias_score, 2),
        # word-level explanation is the LLM layer's job (a later stage),
        # not this model's - left empty here, not removed from the schema,
        # so the response shape stays identical either way.
        "matched_signals": [],
        "note": "bert-base-uncased fine-tuned on BABE (v1, ~82% test accuracy)",
    }