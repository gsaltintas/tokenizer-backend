from fastapi import APIRouter, HTTPException, Query

from app.models.schemas import UndertrainedResponse, UndertrainedToken
from app.services.cache import memo
from app.services.registry import registry
from app.services.undertrained import detect_undertrained_tokens

router = APIRouter(prefix="/api/undertrained", tags=["undertrained"])


@router.get("/{tok_id:path}", response_model=UndertrainedResponse)
async def get_undertrained(
    tok_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000),
):
    adapter = registry.get(tok_id)
    if adapter is None:
        raise HTTPException(status_code=404, detail=f"Tokenizer '{tok_id}' not loaded")

    bpe_available = adapter.tokenizer_type == "bpe"

    if not bpe_available:
        return UndertrainedResponse(
            tokens=[], total=0, page=1, page_size=page_size, bpe_available=False
        )

    all_tokens = memo(adapter, "undertrained", lambda: detect_undertrained_tokens(adapter))
    total = len(all_tokens)
    start = (page - 1) * page_size
    end = start + page_size
    page_tokens = all_tokens[start:end]

    return UndertrainedResponse(
        tokens=[UndertrainedToken(**t) for t in page_tokens],
        total=total,
        page=page,
        page_size=page_size,
        bpe_available=True,
    )
