"""
Token boundaries on source text, as in tokenizer-intrinsic-evals' `tokenizer-visualize`.

The library's CLI prints ANSI-coloured text; this returns the same analysis as
data. Each source character is owned by the token whose (gap-filled) offset
span covers it. A character claimed by more than one token's raw offset is a
multi-byte character split across byte-tokens (red in the CLI). Whitespace,
newline and indentation statistics follow the CLI's definitions.

The source is returned as runs of consecutive characters sharing an owner, so
the client never has to index into the text (Python counts code points,
JavaScript UTF-16 units).
"""

from typing import Any

from app.services.adapter import TokenizerAdapter
from app.services.tokeval_wrapper import to_tokeval_wrapper


def default_samples() -> list[dict[str, str]]:
    from tokenizer_analysis.cli.visualize_tokenization import _DEFAULT_SAMPLES

    return [{"label": label, "text": text} for label, text in _DEFAULT_SAMPLES]


def visualize(adapter: TokenizerAdapter, text: str) -> dict[str, Any]:
    from tokenizer_analysis.cli.visualize_tokenization import (
        _UNASSIGNED,
        _build_char_owner,
        _fill_offsets,
        _get_offsets,
    )

    wrapper = to_tokeval_wrapper(adapter)
    ids = list(wrapper.encode(text))
    raw_tokens = wrapper.convert_ids_to_tokens(ids)
    special_ids = wrapper.get_special_token_ids()
    raw_offsets = _get_offsets(wrapper, text, ids)

    if not raw_offsets:
        # No offsets from this backend: the CLI falls back to raw token strings.
        return {
            "n_tokens": len(ids),
            "has_offsets": False,
            "tokens": [
                {"id": tid, "raw": raw, "special": tid in special_ids}
                for tid, raw in zip(ids, raw_tokens)
            ],
            "segments": None,
            "stats": None,
        }

    raw_offsets = [tuple(o) for o in raw_offsets]
    offsets = _fill_offsets(raw_offsets)
    n = len(text)
    owner, _ = _build_char_owner(offsets, n)
    # Counted from the raw offsets: byte-tokens of one split character all
    # carry that character's span, which _fill_offsets collapses to zero length.
    _, tcount = _build_char_owner(raw_offsets, n)

    tokens = []
    for tid, raw, (s, e), (rs, re_) in zip(ids, raw_tokens, offsets, raw_offsets):
        tokens.append({
            "id": tid,
            "raw": raw,
            "text": text[s:e],
            "start": s,
            "end": e,
            # A declared special id, or an empty span reported by the tokenizer
            # itself (SentencePiece BOS/EOS). Not an empty *filled* span: that
            # is also what a split character's continuation byte-tokens get.
            "special": tid in special_ids or rs == re_,
        })

    segments: list[dict[str, Any]] = []
    for i, ch in enumerate(text):
        tok = owner[i] if owner[i] != _UNASSIGNED else None
        split = tcount[i] if tcount[i] > 1 else 0
        last = segments[-1] if segments else None
        if last and last["token"] == tok and last["split"] == split:
            last["text"] += ch
        else:
            segments.append({"text": ch, "token": tok, "split": split})

    return {
        "n_tokens": len(ids),
        "has_offsets": True,
        "tokens": tokens,
        "segments": segments,
        "stats": _stats(text, tokens, owner, tcount, _UNASSIGNED),
    }


def _stats(text, tokens, owner, tcount, unassigned) -> dict[str, Any]:
    ws_only = newline_toks = newline_indent_toks = 0
    for t in tokens:
        sp = t["text"]
        if not sp:
            continue
        if "\n" in sp:
            newline_toks += 1
            after_last_nl = sp[sp.rfind("\n") + 1:]
            if after_last_nl and after_last_nl.isspace():
                newline_indent_toks += 1
        elif sp.isspace():
            ws_only += 1

    # For each indented line, the distinct tokens owning its leading whitespace.
    indent_patterns: dict[tuple[int, ...], int] = {}
    depth_tokens: dict[int, list[int]] = {}
    total_indent_toks = 0
    pos = 0
    for line in text.split("\n"):
        leading = len(line) - len(line.lstrip())
        if leading:
            owners: list[int] = []
            for ci in range(pos, pos + leading):
                o = owner[ci]
                if o != unassigned and (not owners or owners[-1] != o):
                    owners.append(o)
            total_indent_toks += len(owners)
            depth_tokens.setdefault(leading, []).append(len(owners))
            pattern = tuple(
                sum(1 for ci in range(pos, pos + leading) if owner[ci] == o) for o in owners
            )
            indent_patterns[pattern] = indent_patterns.get(pattern, 0) + 1
        pos += len(line) + 1

    return {
        "whitespace_tokens": ws_only,
        "newline_tokens": newline_toks,
        "newline_indent_tokens": newline_indent_toks,
        "indentation_tokens": total_indent_toks,
        "special_tokens": sum(t["special"] for t in tokens),
        "split_chars": sum(1 for c in tcount if c > 1),
        "hidden_tokens": sum(c - 1 for c in tcount if c > 1),
        "indent_patterns": [
            {"spaces_per_token": list(p), "count": c}
            for p, c in sorted(indent_patterns.items(), key=lambda x: -x[1])
        ],
        "tokens_per_indent_depth": [
            {"depth": d, "avg_tokens": sum(v) / len(v)} for d, v in sorted(depth_tokens.items())
        ],
    }
