from fastapi import APIRouter, HTTPException

from app.services.registry import registry
from app.services.sanity_check import sanity_check

router = APIRouter(prefix="/api/sanity-check", tags=["sanity-check"])


@router.post("/{tokenizer_id:path}")
def run_sanity_check(tokenizer_id: str):
    """Run the 16 TokEval sanity checks on a loaded tokenizer (built-in probes)."""
    adapter = registry.get(tokenizer_id)
    if adapter is None:
        raise HTTPException(status_code=404, detail=f"Tokenizer '{tokenizer_id}' not loaded")
    return {"tokenizer_id": tokenizer_id, **sanity_check(adapter)}
