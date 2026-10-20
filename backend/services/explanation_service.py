"""
services/explanation_service.py

Stage 3, LLM half. Takes text plus its bias_score (already computed by
bias_service.py) and asks an LLM to explain *why* the language might read
as biased - never asking it to rule on whether the underlying claim is
true, per the project's "corroboration over verdicts" principle.

Loads the Groq client once at import time, same pattern as bias_service
loading the BERT model once - not per-request.
"""

import os
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

_api_key = os.environ.get("GROQ_API_KEY")
_client  = Groq(api_key = _api_key) if _api_key else None

MODEL_NAME = "openai/gpt-oss-120b"
SYSTEM_PROMPT = (
        "You are a media-literacy assistant that explains framing bias in one "
    "or two plain sentences. You are given a piece of text and a bias "
    "score from 0 (neutral) to 1 (strongly framed), produced by a "
    "separate classifier. Explain what in the wording might justify that "
    "score - word choice, tone, emphasis. Never state whether the "
    "underlying claim in the text is true or false; you are explaining "
    "framing, not verifying facts."
)

def generate_explanation(text:str,bias_score:float) -> str:
    if _client is None:
        return "Explaination unavailable - GROQ_API_KEY not configured"
    try:
        response = _client.chat.completions.create(
            model = MODEL_NAME,
            messages=[
                {"role":"system","content":SYSTEM_PROMPT},
                {
                    "role":"user",
                    "content":(
                        f"Bias score : {bias_score:.2f}\n"
                        f"Text: \"{text}\""
                    )
                }
            ]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
         # Don't let a rate limit or network blip crash the whole request -
        # the bias score itself is still valid even if the explanation isn't.
        return f"Explaination  unavailable right now ({type(e).__name__})"