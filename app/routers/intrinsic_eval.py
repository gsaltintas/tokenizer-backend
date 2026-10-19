from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel, Field

from app.services.intrinsic_eval import (
    FLORES_LANGUAGES,
    flores_gini_metrics,
    per_text_metrics,
)
from app.services.registry import registry
from app.services.ustat import compute_ustat

router = APIRouter(prefix="/api/intrinsic-eval", tags=["intrinsic-eval"])


class PerTextRequest(BaseModel):
    text: str


class FloresRequest(BaseModel):
    language_codes: list[str] = Field(
        default_factory=lambda: [l["code"] for l in FLORES_LANGUAGES],
        description="FLORES+ language codes to evaluate",
    )
    n_samples: int = Field(200, ge=1, le=1012)


class UStatRequest(BaseModel):
    language_code: str | None = Field(None, description="FLORES+ language code to use as corpus")
    text: str | None = Field(None, description="Custom corpus (used if language_code is not set)")
    n_samples: int = Field(500, ge=1, le=1012)
    lam: float = Field(4.0, ge=0, le=100, description="Weight on branching entropy (paper: 4.0)")
    across_spaces: bool = True


@router.post("/{tokenizer_id:path}/ustat")
async def ustat(tokenizer_id: str, req: UStatRequest):
    """U-statistic boundary metrics (Lopardo et al., arXiv:2610.21679)."""
    adapter = registry.get(tokenizer_id)
    if adapter is None:
        raise HTTPException(status_code=404, detail=f"Tokenizer '{tokenizer_id}' not loaded")

    if req.language_code:
        try:
            from datasets import load_dataset

            ds = load_dataset("openlanguagedata/flores_plus", req.language_code, split="devtest")
            texts = list(ds["text"][: req.n_samples])
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Could not load FLORES+ corpus: {exc}")
    elif req.text and req.text.strip():
        texts = [ln for ln in req.text.splitlines() if ln.strip()]
    else:
        raise HTTPException(status_code=422, detail="Provide language_code or text")

    try:
        result = compute_ustat(adapter, texts, req.lam, req.across_spaces)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return {"tokenizer_id": tokenizer_id, **result}


@router.get("/languages")
async def list_flores_languages():
    """Return the curated list of supported FLORES+ languages."""
    return {"languages": FLORES_LANGUAGES}


@router.post("/{tokenizer_id:path}/per-text")
async def per_text(tokenizer_id: str, req: PerTextRequest):
    adapter = registry.get(tokenizer_id)
    if adapter is None:
        raise HTTPException(status_code=404, detail=f"Tokenizer '{tokenizer_id}' not loaded")
    if not req.text.strip():
        raise HTTPException(status_code=422, detail="text must not be empty")

    metrics = per_text_metrics(adapter, req.text)
    return {"tokenizer_id": tokenizer_id, "metrics": metrics}


@router.post("/{tokenizer_id:path}/flores")
async def flores_eval(tokenizer_id: str, req: FloresRequest):
    adapter = registry.get(tokenizer_id)
    if adapter is None:
        raise HTTPException(status_code=404, detail=f"Tokenizer '{tokenizer_id}' not loaded")
    if not req.language_codes:
        raise HTTPException(status_code=422, detail="language_codes must not be empty")

    result = flores_gini_metrics(adapter, req.language_codes, req.n_samples)
    return {"tokenizer_id": tokenizer_id, **result}
