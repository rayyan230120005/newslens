"""
routes/analyze.py

Extension-facing endpoint for the first MVP slice:
tweet text in, one compact analysis response out.
"""

from fastapi import APIRouter

from services.analyze_service import analyze_text
from shared.schemas import AnalyzeRequest, AnalyzeResponse

router = APIRouter(tags=["analysis"])


@router.post("/analyze", response_model=AnalyzeResponse)
def analyze_route(payload: AnalyzeRequest):
    result = analyze_text(payload.text, tweet_id=payload.tweet_id)
    return AnalyzeResponse(**result)
