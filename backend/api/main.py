"""
api/main.py

The single FastAPI app the extension will eventually call. Stage 1 was
just /health. This adds Stage 2: POST /claim/check. Bias, hate, and
pseudoscience routes get wired in the same way, one at a time.
"""
"""
api/main.py

The single FastAPI app the extension will eventually call.
Stage 1: /health. Stage 2: /claim/check. This adds Stage 3: /bias/score
(placeholder scorer for now). Hate and pseudoscience routes get added the
same way.
"""
from fastapi import FastAPI
from api.routes import health,claim,bias

app = FastAPI(title="NewsLens API")

app.include_router(health.router)
app.include_router(claim.router)
app.include_router(bias.router)
