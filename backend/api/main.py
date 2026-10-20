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
from fastapi import FastAPI  # type: ignore[import-not-found]
from fastapi.middleware.cors import CORSMiddleware

from api.routes import analyze, articles, bias, claim, health

app = FastAPI(title="NewsLens API")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=(
        r"^(chrome-extension://.*|http://localhost(:\d+)?|"
        r"http://127\.0\.0\.1(:\d+)?|https://x\.com|"
        r"https://twitter\.com|https://mobile\.twitter\.com)$"
    ),
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(claim.router)
app.include_router(bias.router)
app.include_router(analyze.router)
app.include_router(articles.router)
