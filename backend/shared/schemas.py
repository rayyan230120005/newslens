from pydantic import BaseModel


class ClaimCheckRequest(BaseModel):
    text:str

class ClaimCheckResponse(BaseModel):
    is_checkable_claim:bool
    confidence:float
    matched_signals:list[str]

class BiasScoreRequest(BaseModel):
    text:str

class BiasScoringResponse(BaseModel):
    bias_score:float
    matched_signals:list[str]
    note:str
