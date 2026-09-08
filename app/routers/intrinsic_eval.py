from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from app.services.intrinsic_eval import (
    FLORES_LANGUAGES,
    flores_gini_metrics,
    per_text_metrics,
)
from app.services.registry import registry

router = APIRouter(prefix="/api/intrinsic-eval", tags=["intrinsic-eval"])


class PerTextRequest(BaseModel):
    text: str


class FloresRequest(BaseModel):
    language_codes: list[str] = Field(
        default_factory=lambda: [l["code"] for l in FLORES_LANGUAGES],
        description="FLORES+ language codes to evaluate",
    )
    n_samples: int = Field(200, ge=1, le=1012)


@router.get("/languages")
async def list_flores_languages():
    """Return the curated list of supported FLORES+ languages."""
    return {"languages": FLORES_LANGUAGES}


@router.post("/{tokenizer_id}/per-text")
async def per_text(tokenizer_id: str, req: PerTextRequest):
    adapter = registry.get(tokenizer_id)
    if adapter is None:
        raise HTTPException(status_code=404, detail=f"Tokenizer '{tokenizer_id}' not loaded")
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")

    metrics = per_text_metrics(adapter, req.text)
    return {"tokenizer_id": tokenizer_id, "metrics": metrics}


@router.post("/{tokenizer_id}/flores")
async def flores_eval(tokenizer_id: str, req: FloresRequest):
    adapter = registry.get(tokenizer_id)
    if adapter is None:
        raise HTTPException(status_code=404, detail=f"Tokenizer '{tokenizer_id}' not loaded")
    if not req.language_codes:
        raise HTTPException(status_code=422, detail="language_codes must not be empty")

    result = flores_gini_metrics(adapter, req.language_codes, req.n_samples)
    return {"tokenizer_id": tokenizer_id, **result}
