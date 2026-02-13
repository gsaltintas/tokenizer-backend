import unicodedata

from fastapi import APIRouter, HTTPException, Query

from app.models.schemas import VocabEntry, VocabResponse, VocabStatsResponse
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




@router.get("/stats/{tok_id:path}", response_model=VocabStatsResponse)
async def get_vocab_stats(tok_id: str):
    adapter = registry.get(tok_id)
    if adapter is None:
        raise HTTPException(status_code=404, detail=f"Tokenizer '{tok_id}' not loaded")

    vocab = adapter.get_vocab()
    length_dist: dict[int, int] = {}
    script_dist: dict[str, int] = {}
    total_length = 0
    max_length = 0

    for token_str in vocab:
        b_len = len(token_str.encode("utf-8", errors="replace"))
        total_length += b_len
        max_length = max(max_length, b_len)
        length_dist[b_len] = length_dist.get(b_len, 0) + 1

        script = _classify_script(token_str)
        script_dist[script] = script_dist.get(script, 0) + 1

    vocab_size = len(vocab)
    return VocabStatsResponse(
        vocab_size=vocab_size,
        avg_token_length=total_length / max(vocab_size, 1),
        max_token_length=max_length,
        length_distribution=length_dist,
        script_distribution=script_dist,
    )


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

    vocab = adapter.get_vocab()
    entries = []
    for token_str, token_id in vocab.items():
        token_bytes = token_str.encode("utf-8", errors="replace")
        entries.append(
            VocabEntry(
                id=token_id,
                token_str=token_str,
                token_bytes_hex=token_bytes.hex(),
                byte_length=len(token_bytes),
                script=_classify_script(token_str),
            )
        )

    # Filter by search
    if search:
        search_lower = search.lower()
        entries = [e for e in entries if search_lower in e.token_str.lower()]

    # Sort
    reverse = sort_dir == "desc"
    if sort_by == "id":
        entries.sort(key=lambda e: e.id, reverse=reverse)
    elif sort_by == "byte_length":
        entries.sort(key=lambda e: e.byte_length, reverse=reverse)
    elif sort_by == "token_str":
        entries.sort(key=lambda e: e.token_str, reverse=reverse)

    total = len(entries)
    start = (page - 1) * page_size
    end = start + page_size
    page_entries = entries[start:end]

    return VocabResponse(entries=page_entries, total=total, page=page, page_size=page_size)
