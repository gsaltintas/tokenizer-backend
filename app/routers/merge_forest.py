from fastapi import APIRouter, HTTPException, Query

from app.models.schemas import (
    MergeForestEntry,
    MergeForestResponse,
    MergeForestSubtreeNode,
    MergeForestSubtreeResponse,
)
from app.services.merge_forest import get_cached_entries, get_subtree
from app.services.registry import registry

router = APIRouter(prefix="/api/merge-forest", tags=["merge-forest"])


def _entry_to_schema(e) -> MergeForestEntry:
    left_str = e.left_bytes.decode("utf-8", errors="replace") if e.left_bytes else None
    right_str = e.right_bytes.decode("utf-8", errors="replace") if e.right_bytes else None
    return MergeForestEntry(
        token=e.token_str(),
        token_hex=e.token_hex(),
        rank=e.rank,
        byte_length=len(e.token_bytes),
        is_leaf=e.is_leaf,
        is_root=e.is_root,
        left=left_str,
        left_hex=e.left_bytes.hex() if e.left_bytes else None,
        left_rank=e.left_rank if e.left_bytes else None,
        right=right_str,
        right_hex=e.right_bytes.hex() if e.right_bytes else None,
        right_rank=e.right_rank if e.right_bytes else None,
    )


def _get_adapter_and_ranks(tok_id: str):
    adapter = registry.get(tok_id)
    if adapter is None:
        raise HTTPException(404, f"Tokenizer '{tok_id}' not loaded")
    ranks = adapter.get_merge_ranks()
    if ranks is None:
        raise HTTPException(400, "Tokenizer does not support BPE merge forest visualization")
    return adapter, ranks


def _count_tree(node: dict) -> tuple[int, int]:
    """Return (depth, node_count) of a subtree dict."""
    if node.get("is_leaf"):
        return 1, 1
    ld, lc = _count_tree(node["left"])
    rd, rc = _count_tree(node["right"])
    return 1 + max(ld, rd), 1 + lc + rc


def _dict_to_schema(d: dict) -> MergeForestSubtreeNode:
    return MergeForestSubtreeNode(
        token=d["token"],
        token_hex=d["token_hex"],
        rank=d["rank"],
        is_leaf=d["is_leaf"],
        left=_dict_to_schema(d["left"]) if d.get("left") else None,
        right=_dict_to_schema(d["right"]) if d.get("right") else None,
    )


# Subtree endpoint MUST be defined before the catch-all {tok_id:path} route
@router.get("/subtree/{rank}", response_model=MergeForestSubtreeResponse)
async def get_merge_forest_subtree(
    rank: int,
    tok_id: str = Query(..., description="Tokenizer ID"),
):
    _, ranks = _get_adapter_and_ranks(tok_id)

    # Find the token with the given rank
    target_bytes: bytes | None = None
    for token_bytes, r in ranks.items():
        if r == rank:
            target_bytes = token_bytes
            break

    if target_bytes is None:
        raise HTTPException(404, f"No token found with rank {rank}")

    tree = get_subtree(ranks, target_bytes)
    depth, node_count = _count_tree(tree)

    return MergeForestSubtreeResponse(
        root=_dict_to_schema(tree),
        depth=depth,
        node_count=node_count,
    )


@router.get("/{tok_id:path}", response_model=MergeForestResponse)
async def get_merge_forest(
    tok_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=500),
    search: str = "",
    sort_by: str = Query("rank", pattern="^(rank|byte_length|token)$"),
    sort_dir: str = Query("asc", pattern="^(asc|desc)$"),
    filter: str = Query("all", pattern="^(all|leaves|merges|roots)$"),
):
    _, ranks = _get_adapter_and_ranks(tok_id)
    entries = get_cached_entries(tok_id, ranks)

    # Filter
    if filter == "leaves":
        filtered = [e for e in entries if e.is_leaf]
    elif filter == "merges":
        filtered = [e for e in entries if not e.is_leaf]
    elif filter == "roots":
        filtered = [e for e in entries if e.is_root]
    else:
        filtered = entries

    # Search
    if search:
        search_lower = search.lower()
        filtered = [
            e for e in filtered
            if search_lower in e.token_str().lower() or search_lower in e.token_hex()
        ]

    # Sort
    reverse = sort_dir == "desc"
    if sort_by == "rank":
        filtered.sort(key=lambda e: e.rank, reverse=reverse)
    elif sort_by == "byte_length":
        filtered.sort(key=lambda e: len(e.token_bytes), reverse=reverse)
    elif sort_by == "token":
        filtered.sort(key=lambda e: e.token_str(), reverse=reverse)

    total = len(filtered)
    start = (page - 1) * page_size
    page_entries = filtered[start : start + page_size]

    total_leaves = sum(1 for e in entries if e.is_leaf)
    total_merges = sum(1 for e in entries if not e.is_leaf)
    total_roots = sum(1 for e in entries if e.is_root)

    return MergeForestResponse(
        entries=[_entry_to_schema(e) for e in page_entries],
        total=total,
        page=page,
        page_size=page_size,
        total_leaves=total_leaves,
        total_merges=total_merges,
        total_roots=total_roots,
    )
