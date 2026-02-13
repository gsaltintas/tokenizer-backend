import re
import unicodedata

from app.services.adapter import TokenizerAdapter


def _normalize(token: str) -> str:
    """Normalize a token to its base form for grouping."""
    # Strip leading space markers (Ġ = U+0120, or actual space/NBSP)
    s = token.lstrip(" \u0120\u00a0")
    # Lowercase
    s = s.lower()
    # Strip punctuation
    s = re.sub(r"[^\w]", "", s, flags=re.UNICODE)
    return s


def _detect_casing(token: str) -> str:
    """Detect casing pattern of a token."""
    letters = [c for c in token if c.isalpha()]
    if not letters:
        return "lower"
    if all(c.isupper() for c in letters):
        return "upper"
    if all(c.islower() for c in letters):
        return "lower"
    if letters[0].isupper() and all(c.islower() for c in letters[1:]):
        return "title"
    return "mixed"


def _has_punctuation(token: str) -> bool:
    return any(unicodedata.category(c).startswith("P") for c in token)


def _has_space_prefix(token: str) -> bool:
    return token.startswith((" ", "\u0120", "\u00a0", "▁"))


def compute_multiplicity_groups(
    adapter: TokenizerAdapter,
) -> list[dict]:
    """Compute all multiplicity groups for a tokenizer's vocabulary."""
    vocab = adapter.get_vocab()
    groups: dict[str, list[dict]] = {}

    for token_str, token_id in vocab.items():
        base = _normalize(token_str)
        if not base or len(base) < 1:
            continue

        variant = {
            "token_id": token_id,
            "token_str": token_str,
            "has_space_prefix": _has_space_prefix(token_str),
            "casing": _detect_casing(token_str),
            "has_punctuation": _has_punctuation(token_str),
        }

        if base not in groups:
            groups[base] = []
        groups[base].append(variant)

    # Only keep groups with more than one variant
    result = []
    for base_form, variants in groups.items():
        if len(variants) > 1:
            result.append(
                {
                    "base_form": base_form,
                    "variants": variants,
                    "count": len(variants),
                }
            )

    # Sort by group size (most variants first)
    result.sort(key=lambda g: g["count"], reverse=True)
    return result


def search_multiplicity_groups(
    adapter: TokenizerAdapter, query: str
) -> list[dict]:
    """Search for multiplicity groups matching a query."""
    all_groups = compute_multiplicity_groups(adapter)
    query_lower = query.lower().strip()
    return [g for g in all_groups if query_lower in g["base_form"]]
