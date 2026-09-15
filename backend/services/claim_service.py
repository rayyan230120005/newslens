"""
services/claim_service.py

Stage 2: a lightweight, ClaimBuster-style gate that decides whether a post
even contains a checkable factual claim, before any of the heavier bias/hate
models run on it (see project doc, "Claim-relevance gating").

This first version is a plain rule-based heuristic, not a trained
classifier - deliberately, so the endpoint contract can be proven end to
end before any model or training data enters the picture. Swap the body of
check_claim() for a real ClaimBuster-style model later without touching the
route or the schema.
"""
CLAIM_KEYWORDS = [
    "said","according to","reported","announced","confirmed",
    "study shows","study found","data shows","statistics",
    "percent","%","claims","minister","governemnt","scientists"
]

def check_claim(text:str) -> dict:
    text_lower = text.lower()
    matched = [kw for kw in CLAIM_KEYWORDS if kw in text_lower]

    if any(char.isdigit() for char in text):
        matched.append("contains a number")

    is_checkable = bool(matched)
    confidence = min(1.0,0.3 + 0.2 * len(matched)) if is_checkable else 0.1
    return{
        "is_checkable_claim":is_checkable,
        "confidence":round(confidence,2),
        "matched_signals":matched
    }