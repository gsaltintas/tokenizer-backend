from fastapi import APIRouter, HTTPException

from app.models.schemas import LanguageCompositionResponse, ScriptCategory
from app.services.language import compute_language_composition
from app.services.registry import registry

router = APIRouter(prefix="/api/language", tags=["language"])

_language_cache: dict[str, dict] = {}


@router.get("/{tok_id:path}", response_model=LanguageCompositionResponse)
async def get_language_composition(tok_id: str):
    if tok_id not in _language_cache:
        adapter = registry.get(tok_id)
        if adapter is None:
            raise HTTPException(status_code=404, detail=f"Tokenizer '{tok_id}' not loaded")
        _language_cache[tok_id] = compute_language_composition(adapter)

    data = _language_cache[tok_id]
    return LanguageCompositionResponse(
        categories=[ScriptCategory(**c) for c in data["categories"]],
        total_tokens=data["total_tokens"],
        mixed_script_count=data["mixed_script_count"],
    )
