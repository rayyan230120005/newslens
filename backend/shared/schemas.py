from typing import Optional

from pydantic import BaseModel


class ClaimCheckRequest(BaseModel):
    text: str


class ClaimCheckResponse(BaseModel):
    is_checkable_claim: bool
    confidence: float
    matched_signals: list[str]


class BiasScoreRequest(BaseModel):
    text: str


class BiasScoreResponse(BaseModel):
    bias_score: float
    matched_signals: list[str]
    note: str


class BiasExplainRequest(BaseModel):
    text: str


class BiasExplainResponse(BaseModel):
    bias_score: float
    explanation: str


class AnalyzeRequest(BaseModel):
    text: str
    tweet_id: Optional[str] = None
    source: Optional[str] = None


class AnalyzeResponse(BaseModel):
    tweet_id: Optional[str]
    is_checkable_claim: bool
    claim_confidence: float
    bias_score: Optional[float]
    label: str
    should_blur: bool
    explanation: str
    matched_signals: list[str]
