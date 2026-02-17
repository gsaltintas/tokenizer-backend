from fastapi import APIRouter, HTTPException

from app.models.schemas import MergeTreeComparisonResponse, MergeTreeRequest
from app.services.merge_tree import compare_merge_trees
from app.services.registry import registry

router = APIRouter(prefix="/api/merge-tree", tags=["merge-tree"])


@router.post("/compare", response_model=MergeTreeComparisonResponse)
async def merge_tree_compare(req: MergeTreeRequest):
    adapters = []
    for tid in req.tokenizer_ids:
        adapter = registry.get(tid)
        if adapter is None:
            raise HTTPException(404, f"Tokenizer '{tid}' not loaded")
        adapters.append(adapter)

    ranks_a = adapters[0].get_merge_ranks()
    ranks_b = adapters[1].get_merge_ranks()

    if ranks_a is None:
        raise HTTPException(400, f"Tokenizer '{req.tokenizer_ids[0]}' does not expose BPE merge ranks")
    if ranks_b is None:
        raise HTTPException(400, f"Tokenizer '{req.tokenizer_ids[1]}' does not expose BPE merge ranks")

    result = compare_merge_trees(
        text=req.text,
        ranks_a=ranks_a,
        ranks_b=ranks_b,
        name_a=adapters[0].name,
        name_b=adapters[1].name,
    )
    return result
