from fastapi import APIRouter, HTTPException, Query

from app.models.schemas import MorphemeAnalysisResponse, MorphemeBreakdown
from app.services.morphemes import compute_morpheme_analysis
from app.services.registry import registry

router = APIRouter(prefix="/api/morphemes", tags=["morphemes"])

_morpheme_cache: dict[str, list[dict]] = {}


@router.get("/{tok_id:path}", response_model=MorphemeAnalysisResponse)
async def get_morphemes(
    tok_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000),
    type_filter: str = Query(""),
):
    if tok_id not in _morpheme_cache:
        adapter = registry.get(tok_id)
        if adapter is None:
            raise HTTPException(status_code=404, detail=f"Tokenizer '{tok_id}' not loaded")
        _morpheme_cache[tok_id] = compute_morpheme_analysis(adapter)

    all_results = _morpheme_cache[tok_id]

    # Compute type distribution from full results
    type_dist: dict[str, int] = {}
    for r in all_results:
        t = r["morpheme_type"]
        type_dist[t] = type_dist.get(t, 0) + 1

    # Apply filter
    if type_filter:
        filtered = [r for r in all_results if r["morpheme_type"] == type_filter]
    else:
        filtered = all_results

    total = len(filtered)
    start = (page - 1) * page_size
    end = start + page_size
    page_results = filtered[start:end]

    return MorphemeAnalysisResponse(
        breakdowns=[MorphemeBreakdown(**r) for r in page_results],
        total=total,
        page=page,
        page_size=page_size,
        type_distribution=type_dist,
    )
