"""Build a flat merge forest from a BPE merge-ranks dictionary.

Each multi-byte token is decomposed into its optimal left+right split,
forming a forest of binary trees rooted at tokens that never appear as
children of another merge.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MergeEntry:
    token_bytes: bytes
    rank: int
    left_bytes: bytes | None = None
    right_bytes: bytes | None = None
    left_rank: int = -1
    right_rank: int = -1
    is_leaf: bool = False
    is_root: bool = True

    def token_str(self) -> str:
        return self.token_bytes.decode("utf-8", errors="replace")

    def token_hex(self) -> str:
        return self.token_bytes.hex()


def build_merge_entries(ranks: dict[bytes, int]) -> list[MergeEntry]:
    """Build a flat list of merge entries from a ranks dictionary.

    For each multi-byte token, find the optimal split point where both
    halves exist in ranks and max(left_rank, right_rank) is minimized.
    """
    entries: list[MergeEntry] = []
    children: set[bytes] = set()

    for token_bytes, rank in ranks.items():
        if len(token_bytes) <= 1:
            entries.append(MergeEntry(
                token_bytes=token_bytes,
                rank=rank,
                is_leaf=True,
            ))
            continue

        # Find optimal split
        best_split: tuple[bytes, bytes] | None = None
        best_max_rank = float("inf")

        for i in range(1, len(token_bytes)):
            left = token_bytes[:i]
            right = token_bytes[i:]
            if left in ranks and right in ranks:
                max_rank = max(ranks[left], ranks[right])
                if max_rank < best_max_rank:
                    best_max_rank = max_rank
                    best_split = (left, right)

        if best_split is not None:
            left_b, right_b = best_split
            children.add(left_b)
            children.add(right_b)
            entries.append(MergeEntry(
                token_bytes=token_bytes,
                rank=rank,
                left_bytes=left_b,
                right_bytes=right_b,
                left_rank=ranks[left_b],
                right_rank=ranks[right_b],
            ))
        else:
            # No valid split found — treat as leaf
            entries.append(MergeEntry(
                token_bytes=token_bytes,
                rank=rank,
                is_leaf=True,
            ))

    # Mark non-root entries
    for entry in entries:
        if entry.token_bytes in children:
            entry.is_root = False

    return entries


def get_subtree(ranks: dict[bytes, int], token_bytes: bytes) -> dict:
    """Recursively decompose a token into its full merge tree."""
    rank = ranks.get(token_bytes, -1)
    token_str = token_bytes.decode("utf-8", errors="replace")
    token_hex = token_bytes.hex()

    if len(token_bytes) <= 1:
        return {
            "token": token_str,
            "token_hex": token_hex,
            "rank": rank,
            "is_leaf": True,
        }

    # Find optimal split (same logic as build_merge_entries)
    best_split: tuple[bytes, bytes] | None = None
    best_max_rank = float("inf")

    for i in range(1, len(token_bytes)):
        left = token_bytes[:i]
        right = token_bytes[i:]
        if left in ranks and right in ranks:
            max_rank = max(ranks[left], ranks[right])
            if max_rank < best_max_rank:
                best_max_rank = max_rank
                best_split = (left, right)

    if best_split is None:
        return {
            "token": token_str,
            "token_hex": token_hex,
            "rank": rank,
            "is_leaf": True,
        }

    return {
        "token": token_str,
        "token_hex": token_hex,
        "rank": rank,
        "is_leaf": False,
        "left": get_subtree(ranks, best_split[0]),
        "right": get_subtree(ranks, best_split[1]),
    }


# Module-level cache: tokenizer_id -> list[MergeEntry]
_forest_cache: dict[str, list[MergeEntry]] = {}


def get_cached_entries(tok_id: str, ranks: dict[bytes, int]) -> list[MergeEntry]:
    """Get or build cached merge entries for a tokenizer."""
    if tok_id not in _forest_cache:
        _forest_cache[tok_id] = build_merge_entries(ranks)
    return _forest_cache[tok_id]
