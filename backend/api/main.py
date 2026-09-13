"""
api/main.py

The single FastAPI app the extension will eventually call. Right now it
does nothing but expose /health - routes for claim detection, bias, hate,
and pseudoscience get added one at a time in later steps, each wired in
here the same way health is.
"""

from fastapi import FastAPI
from api.routes import health

app = FastAPI(title="NewsLens API")

app.include_router(health.router)
