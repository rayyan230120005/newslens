"""
routes/bias.py

Stage 3 endpoint, placeholder version. Same shape it'll have once the real
BERT model is wired in via services/bias_service.py - only the internals of
score_bias() change later, not this route.
"""

from fastapi import APIRouter
from shared.schemas import BiasScoreRequest,BiasScoringResponse
from services.bias_service import score_bias

router = APIRouter(prefix="/bias", tags = ["bias"])

@router.post("/score",response_model=BiasScoringResponse)
def score_bias_route(payload: BiasScoreRequest):
    result = score_bias(payload.text)
    return BiasScoringResponse(**result)
