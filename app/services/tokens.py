from app.models.schemas import TokenInfo
from app.services.adapter import TokenizerAdapter


def build_token_infos(adapter: TokenizerAdapter, text: str) -> list[TokenInfo]:
    """Encode text into per-token display info.

    Byte-level tokenizers can split one character across several tokens
    (e.g. the 4 UTF-8 bytes of an emoji over 3 tokens), so no single token
    decodes to valid text.  Consecutive tokens are grouped until their
    concatenated bytes decode cleanly; tokens in a multi-token group are
    marked partial and share the group's decoded string and character span.
    """
    token_ids = adapter.encode(text)
    raw = [adapter.raw_token_bytes(tid) for tid in token_ids]
    if any(b is None for b in raw):
        return _build_from_decode(adapter, text, token_ids)

    tokens: list[TokenInfo] = []
    offset = 0
    group_id = 0
    i = 0
    while i < len(token_ids):
        # Extend the group until its bytes form complete UTF-8 characters
        j = i
        buf = b""
        group_str = None
        while j < len(token_ids):
            buf += raw[j]
            j += 1
            try:
                group_str = buf.decode("utf-8")
                break
            except UnicodeDecodeError:
                pass
        if group_str is None:  # incomplete sequence at end of input
            group_str = buf.decode("utf-8", errors="replace")

        start, end = _locate(text, group_str, offset)
        offset = end
        multi = j - i > 1
        for k in range(i, j):
            b = raw[k]
            tokens.append(
                TokenInfo(
                    id=token_ids[k],
                    token_str=group_str if not multi else b.decode("utf-8", errors="backslashreplace"),
                    token_bytes_hex=b.hex(),
                    byte_length=len(b),
                    start=start,
                    end=end,
                    is_partial=multi,
                    group_id=group_id if multi else None,
                    group_str=group_str if multi else None,
                )
            )
        if multi:
            group_id += 1
        i = j
    return tokens


def _locate(text: str, s: str, offset: int) -> tuple[int, int]:
    """Character span of s in text at/after offset.  SentencePiece-style
    tokenizers add a leading space (▁) that isn't in the input, so also try
    without it."""
    start = text.find(s, offset)
    if start != -1:
        return start, start + len(s)
    stripped = s.lstrip(" ")
    if stripped and stripped != s:
        start = text.find(stripped, offset)
        if start != -1:
            return start, start + len(stripped)
    return offset, min(offset + len(s), len(text))


def _build_from_decode(adapter: TokenizerAdapter, text: str, token_ids: list[int]) -> list[TokenInfo]:
    """Fallback for backends without byte-level access (e.g. WordPiece)."""
    tokens: list[TokenInfo] = []
    offset = 0
    prev_decoded = ""
    for tid in token_ids:
        # Decode the growing prefix and diff against the previous one to keep
        # context (e.g. SentencePiece ▁ → space).
        curr_decoded = adapter.decode(token_ids[: len(tokens) + 1])
        token_str = curr_decoded[len(prev_decoded):]
        prev_decoded = curr_decoded

        token_bytes = token_str.encode("utf-8", errors="replace")
        start = text.find(token_str, offset)
        if start == -1:
            start = offset
        end = start + len(token_str)
        offset = end
        tokens.append(
            TokenInfo(
                id=tid,
                token_str=token_str,
                token_bytes_hex=token_bytes.hex(),
                byte_length=len(token_bytes),
                start=start,
                end=end,
            )
        )
    return tokens
