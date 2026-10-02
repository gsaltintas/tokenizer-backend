from fastapi import APIRouter, HTTPException

from app.models.schemas import LanguageCompositionResponse, ScriptCategory
from app.services.cache import memo
from app.services.language import compute_language_composition
from app.services.registry import registry

router = APIRouter(prefix="/api/language", tags=["language"])


@router.get("/{tok_id:path}", response_model=LanguageCompositionResponse)
async def get_language_composition(tok_id: str):
    adapter = registry.get(tok_id)
    if adapter is None:
        raise HTTPException(status_code=404, detail=f"Tokenizer '{tok_id}' not loaded")
    data = memo(adapter, "language", lambda: compute_language_composition(adapter))
    return LanguageCompositionResponse(
        categories=[ScriptCategory(**c) for c in data["categories"]],
        total_tokens=data["total_tokens"],
        mixed_script_count=data["mixed_script_count"],
    )
