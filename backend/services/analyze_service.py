"""
services/analyze_service.py

Single contract used by the browser extension. It keeps the extension from
knowing about the internal claim and bias endpoints, so the ML model can be
swapped later without rewriting the content script.
"""

from typing import Optional

from services.claim_service import check_claim
from services.bias_service import score_bias

BIAS_REVIEW_THRESHOLD = 0.55
BIAS_BLUR_THRESHOLD = 0.7


def analyze_text(text: str, tweet_id: Optional[str] = None) -> dict:
    cleaned_text = text.strip()
    claim_result = check_claim(cleaned_text)
    matched_signals = list(claim_result["matched_signals"])

    if not claim_result["is_checkable_claim"]:
        return {
            "tweet_id": tweet_id,
            "is_checkable_claim": False,
            "claim_confidence": claim_result["confidence"],
            "bias_score": None,
            "label": "not_checkable",
            "should_blur": False,
            "explanation": "No checkable factual claim was detected.",
            "matched_signals": matched_signals,
        }

    bias_result = score_bias(cleaned_text)
    bias_score = bias_result["bias_score"]
    matched_signals.extend(bias_result.get("matched_signals", []))

    if bias_score >= BIAS_BLUR_THRESHOLD:
        label = "possible_framing_bias"
        should_blur = True
        explanation = "This checkable claim appears to use strongly framed language."
    elif bias_score >= BIAS_REVIEW_THRESHOLD:
        label = "review_framing"
        should_blur = False
        explanation = "This checkable claim may contain some framing cues."
    else:
        label = "low_framing_signal"
        should_blur = False
        explanation = "This checkable claim has a low framing-bias signal."

    return {
        "tweet_id": tweet_id,
        "is_checkable_claim": True,
        "claim_confidence": claim_result["confidence"],
        "bias_score": bias_score,
        "label": label,
        "should_blur": should_blur,
        "explanation": explanation,
        "matched_signals": matched_signals,
    }
