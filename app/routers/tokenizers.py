from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    LoadTokenizerRequest,
    LoadTokenizerResponse,
    TokenizerInfo,
    TokenizerListResponse,
)
from app.services.registry import registry

router = APIRouter(prefix="/api/tokenizers", tags=["tokenizers"])


@router.get("", response_model=TokenizerListResponse)
async def list_tokenizers():
    """List available and loaded tokenizers."""
    loaded = registry.list_loaded()
    available = registry.list_available()
    # Merge: loaded first, then available that aren't loaded
    loaded_ids = {t["id"] for t in loaded}
    all_tokenizers = loaded + [t for t in available if t["id"] not in loaded_ids]
    return TokenizerListResponse(
        tokenizers=[TokenizerInfo(**t) for t in all_tokenizers]
    )


@router.post("/load", response_model=LoadTokenizerResponse)
async def load_tokenizer(req: LoadTokenizerRequest):
    """Load a tokenizer by name, HuggingFace model ID, or file path."""
    try:
        adapter = registry.load(req.name)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load tokenizer: {e}")

    return LoadTokenizerResponse(
        tokenizer=TokenizerInfo(
            id=req.name,
            name=adapter.name,
            tokenizer_type=adapter.tokenizer_type,
            vocab_size=adapter.vocab_size(),
            source=adapter.source,
        )
    )
