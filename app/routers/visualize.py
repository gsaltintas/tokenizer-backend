from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.services.registry import registry
from app.services.visualize import default_samples, visualize

router = APIRouter(prefix="/api/visualize", tags=["visualize"])

MAX_TEXT_CHARS = 50_000


class VisualizeRequest(BaseModel):
    tokenizer_ids: list[str] = Field(..., min_length=1, max_length=8)
    text: str


@router.get("/samples")
async def samples():
    """Built-in samples from tokenizer-visualize: Python code, Unicode math, multilingual."""
    return {"samples": default_samples()}


@router.post("")
def visualize_text(req: VisualizeRequest):
    """Token boundaries on the source text, per tokenizer."""
    if not req.text:
        raise HTTPException(status_code=422, detail="text must not be empty")
    if len(req.text) > MAX_TEXT_CHARS:
        raise HTTPException(status_code=422, detail=f"text longer than {MAX_TEXT_CHARS} chars")

    results = []
    for tid in req.tokenizer_ids:
        adapter = registry.get(tid)
        if adapter is None:
            results.append({"tokenizer_id": tid, "error": f"Tokenizer '{tid}' not loaded"})
            continue
        try:
            results.append({"tokenizer_id": tid, "name": adapter.name, **visualize(adapter, req.text)})
        except Exception as e:
            results.append({"tokenizer_id": tid, "error": str(e)})
    return {"results": results}
