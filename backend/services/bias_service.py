"""
services/bias_service.py

Stage 3: framing-bias scoring. The real version (per project doc, section
3a) is a BERT model fine-tuned on BABE/MBIC, adapted toward Indian political
framing, with an LLM layer generating a plain-language explanation.

This first version is a placeholder lexicon heuristic - counts loaded/
charged words often associated with selective framing (not the same as
sentiment, and nowhere near as reliable as a trained model). It exists
purely to prove the /bias/score endpoint contract before any dataset or
training work starts. score_bias() is the only thing that changes when the
real model is swapped in - the route and schema stay identical.
"""

LOADED_LANGUAGE = [
    "slammed","blasted","radical","so-called","regime",
    "extremist","shocking","outrageous","sensational",
    "furious","blasts","blast","slams","destroys","annihilates"
]

def score_bias(text:str) -> dict:
    text_lower = text.lower()
    matched = [word for word in LOADED_LANGUAGE if word in text_lower]

    word_count = max(len(text.split()),1)
    bias_score = min(1.0, len(matched) / word_count * 5)

    return{
        "bias_score":round(bias_score,2),
        "matched_signals":matched,
        "note":"placeholder heuristic - not the trained BERT model yet",
    }
