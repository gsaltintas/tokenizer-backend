from fastapi import APIRouter, HTTPException, Query

from app.models.schemas import MultiplicityGroup, MultiplicityResponse, VariantInfo
from app.services.multiplicity import compute_multiplicity_groups, search_multiplicity_groups
from app.services.registry import registry

router = APIRouter(prefix="/api/multiplicity", tags=["multiplicity"])

# Cache multiplicity results per tokenizer
_multiplicity_cache: dict[str, list[dict]] = {}


def _get_groups(tok_id: str) -> list[dict]:
    if tok_id not in _multiplicity_cache:
        adapter = registry.get(tok_id)
        if adapter is None:
            raise HTTPException(status_code=404, detail=f"Tokenizer '{tok_id}' not loaded")
        _multiplicity_cache[tok_id] = compute_multiplicity_groups(adapter)
    return _multiplicity_cache[tok_id]



@router.get("/search/{tok_id:path}", response_model=MultiplicityResponse)
async def search_multiplicity(
    tok_id: str,
    query: str = Query("", min_length=1),
):
    adapter = registry.get(tok_id)
    if adapter is None:
        raise HTTPException(status_code=404, detail=f"Tokenizer '{tok_id}' not loaded")

    results = search_multiplicity_groups(adapter, query)

    return MultiplicityResponse(
        groups=[
            MultiplicityGroup(
                base_form=g["base_form"],
                variants=[VariantInfo(**v) for v in g["variants"]],
                count=g["count"],
            )
            for g in results[:100]  # Cap at 100 results
        ],
        total_groups=len(results),
        page=1,
        page_size=len(results),
    )

# put general route second
@router.get("/{tok_id:path}", response_model=MultiplicityResponse)
async def get_multiplicity(
    tok_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
):
    groups = _get_groups(tok_id)
    total = len(groups)
    start = (page - 1) * page_size
    end = start + page_size
    page_groups = groups[start:end]

    return MultiplicityResponse(
        groups=[
            MultiplicityGroup(
                base_form=g["base_form"],
                variants=[VariantInfo(**v) for v in g["variants"]],
                count=g["count"],
            )
            for g in page_groups
        ],
        total_groups=total,
        page=page,
        page_size=page_size,
    )
