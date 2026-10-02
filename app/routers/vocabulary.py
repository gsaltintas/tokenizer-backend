import unicodedata

from fastapi import APIRouter, HTTPException, Query

from app.models.schemas import VocabEntry, VocabResponse, VocabStatsResponse
from app.services.cache import memo
from app.services.registry import registry

router = APIRouter(prefix="/api/vocab", tags=["vocabulary"])


def _classify_script(token_str: str) -> str:
    """Classify the dominant Unicode script of a token."""
    scripts: dict[str, int] = {}
    for ch in token_str:
        cat = unicodedata.category(ch)
        if cat.startswith("L"):
            try:
                script = unicodedata.name(ch, "").split(" ")[0]
            except ValueError:
                script = "Unknown"
        elif cat.startswith("N"):
            script = "Digit"
        elif cat.startswith("P") or cat.startswith("S"):
            script = "Punctuation"
        elif cat.startswith("Z") or cat.startswith("C"):
            script = "Control/Space"
        else:
            script = "Other"
        scripts[script] = scripts.get(script, 0) + 1

    if not scripts:
        return "Empty"
    return max(scripts, key=lambda s: scripts[s])



# (id, token_str, token_bytes_hex, byte_length, script), in vocab order
_Row = tuple[int, str, str, int, str]
_SORT_KEYS = {
    "id": lambda r: r[0],
    "byte_length": lambda r: r[3],
    "token_str": lambda r: r[1],
}


def _vocab_rows(adapter) -> list[_Row]:
    def build() -> list[_Row]:
        rows = []
        for token_str, token_id in adapter.get_vocab().items():
            token_bytes = token_str.encode("utf-8", errors="replace")
            rows.append(
                (token_id, token_str, token_bytes.hex(), len(token_bytes), _classify_script(token_str))
            )
        return rows

    return memo(adapter, "vocab_rows", build)


def _sorted_rows(adapter, sort_by: str, reverse: bool) -> list[_Row]:
    key = _SORT_KEYS.get(sort_by)
    if key is None:
        return _vocab_rows(adapter)
    return memo(
        adapter,
        f"vocab_rows:{sort_by}:{reverse}",
        lambda: sorted(_vocab_rows(adapter), key=key, reverse=reverse),
    )


def _compute_stats(adapter) -> VocabStatsResponse:
    rows = _vocab_rows(adapter)
    length_dist: dict[int, int] = {}
    script_dist: dict[str, int] = {}
    total_length = 0
    max_length = 0

    for _, _, _, b_len, script in rows:
        total_length += b_len
        max_length = max(max_length, b_len)
        length_dist[b_len] = length_dist.get(b_len, 0) + 1
        script_dist[script] = script_dist.get(script, 0) + 1

    vocab_size = len(rows)
    return VocabStatsResponse(
        vocab_size=vocab_size,
        avg_token_length=total_length / max(vocab_size, 1),
        max_token_length=max_length,
        length_distribution=length_dist,
        script_distribution=script_dist,
    )


@router.get("/stats/{tok_id:path}", response_model=VocabStatsResponse)
async def get_vocab_stats(tok_id: str):
    adapter = registry.get(tok_id)
    if adapter is None:
        raise HTTPException(status_code=404, detail=f"Tokenizer '{tok_id}' not loaded")
    return memo(adapter, "vocab_stats", lambda: _compute_stats(adapter))


@router.get("/{tok_id:path}", response_model=VocabResponse)
async def get_vocab(
    tok_id: str,
    page: int = Query(1, ge=1),
    page_size: int = Query(100, ge=1, le=1000),
    search: str = Query(""),
    sort_by: str = Query("id"),
    sort_dir: str = Query("asc"),
):
    adapter = registry.get(tok_id)
    if adapter is None:
        raise HTTPException(status_code=404, detail=f"Tokenizer '{tok_id}' not loaded")

    # Sorting is stable, so filtering the cached sorted list gives the same
    # order as filtering first and sorting after.
    rows = _sorted_rows(adapter, sort_by, sort_dir == "desc")
    if search:
        search_lower = search.lower()
        rows = [r for r in rows if search_lower in r[1].lower()]

    total = len(rows)
    start = (page - 1) * page_size
    page_entries = [
        VocabEntry(id=i, token_str=s, token_bytes_hex=h, byte_length=n, script=sc)
        for i, s, h, n, sc in rows[start : start + page_size]
    ]

    return VocabResponse(entries=page_entries, total=total, page=page, page_size=page_size)
