"""Backend-only article URL extraction endpoint."""

from fastapi import APIRouter, HTTPException

from services.article_extraction_service import ArticleExtractionError, extract_article
from shared.schemas import ArticleExtractRequest, ArticleExtractResponse

router = APIRouter(prefix="/articles", tags=["articles"])


@router.post("/extract", response_model=ArticleExtractResponse)
def extract_article_route(payload: ArticleExtractRequest):
    try:
        return extract_article(payload.url, max_words=payload.max_words)
    except ArticleExtractionError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.message) from exc
