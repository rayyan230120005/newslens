"""
routes/bias.py

Stage 3. /bias/score (already live) returns just the number from the
fine-tuned BERT model. /bias/explain (new) runs both bias_service and
explanation_service, returning a score plus a plain-language explanation
of it - the full "BERT + LLM explanation" pipeline from the project doc.
"""

from fastapi import APIRouter
from shared.schemas import BiasScoreRequest, BiasScoreResponse, BiasExplainRequest, BiasExplainResponse
from services.bias_service import score_bias
from services.explanation_service import generate_explanation

router = APIRouter(prefix="/bias", tags=["bias"])


@router.post("/score", response_model=BiasScoreResponse)
def score_bias_route(payload: BiasScoreRequest):
    result = score_bias(payload.text)
    return BiasScoreResponse(**result)


@router.post("/explain", response_model=BiasExplainResponse)
def explain_bias_route(payload: BiasExplainRequest):
    score_result = score_bias(payload.text)
    explanation = generate_explanation(payload.text, score_result["bias_score"])
    return BiasExplainResponse(
        bias_score=score_result["bias_score"],
        explanation=explanation,
    )