"""
routes/claim.py

Stage 2 endpoint. Takes raw text, runs it through the claim-detection gate,
and returns whether it's worth passing further down the pipeline (bias/hate
scoring). Nothing here talks to the extension yet - that wiring is a later
step, once this contract is proven with curl.
"""

from fastapi import APIRouter
from shared.schemas import ClaimCheckRequest , ClaimCheckResponse
from services.claim_service import check_claim

router = APIRouter(prefix="/claim",tags=["claim"])

@router.post("/check", response_model=ClaimCheckResponse)
def check_claim_route(payload: ClaimCheckRequest):
    result = check_claim(payload.text)
    return ClaimCheckResponse(**result)
