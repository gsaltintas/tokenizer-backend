"""BPE Merge Tree builder — simulates BPE merges and records the full binary tree."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class MergeNode:
    token: bytes
    left: MergeNode | None = None
    right: MergeNode | None = None
    rank: int = -1

    def is_leaf(self) -> bool:
        return self.left is None and self.right is None

    def text(self) -> str:
        return self.token.decode("utf-8", errors="replace")


def build_merge_tree(text: str, ranks: dict[bytes, int]) -> list[MergeNode]:
    """Simulate BPE merges on *text*, returning a forest of merge trees.

    Each root in the returned list is either a leaf (single byte that was
    never merged) or a binary tree recording the actual BPE merges.
    No ghost/virtual intermediate nodes are created.
    """
    nodes = [MergeNode(bytes([b])) for b in text.encode("utf-8")]

    while len(nodes) > 1:
        best_rank = float("inf")
        best_idx = -1
        best_pair: bytes | None = None

        for i in range(len(nodes) - 1):
            pair = nodes[i].token + nodes[i + 1].token
            rank = ranks.get(pair)
            if rank is not None and rank < best_rank:
                best_rank = rank
                best_idx = i
                best_pair = pair

        if best_pair is None:
            break

        merged = MergeNode(
            best_pair,
            left=nodes[best_idx],
            right=nodes[best_idx + 1],
            rank=best_rank,
        )
        nodes = nodes[:best_idx] + [merged] + nodes[best_idx + 2:]

    return nodes


@dataclass
class MergeStep:
    step: int
    merged_token: str
    rank: int
    tokens_after: list[str]


def compute_merge_steps(text: str, ranks: dict[bytes, int]) -> list[MergeStep]:
    """Return the step-by-step merge sequence."""
    nodes = [MergeNode(bytes([b])) for b in text.encode("utf-8")]
    steps: list[MergeStep] = []
    step_num = 1

    while len(nodes) > 1:
        best_rank = float("inf")
        best_idx = -1
        best_pair: bytes | None = None

        for i in range(len(nodes) - 1):
            pair = nodes[i].token + nodes[i + 1].token
            rank = ranks.get(pair)
            if rank is not None and rank < best_rank:
                best_rank = rank
                best_idx = i
                best_pair = pair

        if best_pair is None:
            break

        merged = MergeNode(
            best_pair,
            left=nodes[best_idx],
            right=nodes[best_idx + 1],
            rank=best_rank,
        )
        nodes = nodes[:best_idx] + [merged] + nodes[best_idx + 2:]

        steps.append(MergeStep(
            step=step_num,
            merged_token=best_pair.decode("utf-8", errors="replace"),
            rank=best_rank,
            tokens_after=[n.text() for n in nodes],
        ))
        step_num += 1

    return steps


def node_to_dict(node: MergeNode) -> dict:
    """Serialize a MergeNode tree to a JSON-friendly dict."""
    result: dict = {
        "token": node.text(),
        "rank": node.rank,
        "is_leaf": node.is_leaf(),
    }
    if not node.is_leaf():
        result["left"] = node_to_dict(node.left)
        result["right"] = node_to_dict(node.right)
    return result


def collect_intermediates_from_node(node: MergeNode, acc: set[str]) -> None:
    """Collect all intermediate (non-leaf) token strings from a single tree."""
    if not node.is_leaf():
        acc.add(node.text())
        if node.left is not None:
            collect_intermediates_from_node(node.left, acc)
        if node.right is not None:
            collect_intermediates_from_node(node.right, acc)


def collect_intermediates(forest: list[MergeNode]) -> set[str]:
    """Collect all intermediate (non-leaf) token strings from a forest."""
    acc: set[str] = set()
    for node in forest:
        collect_intermediates_from_node(node, acc)
    return acc


def compare_merge_trees(
    text: str,
    ranks_a: dict[bytes, int],
    ranks_b: dict[bytes, int],
    name_a: str,
    name_b: str,
) -> dict:
    """Build merge trees for both tokenizers and compute conflict analysis."""
    tree_a = build_merge_tree(text, ranks_a)
    tree_b = build_merge_tree(text, ranks_b)

    steps_a = compute_merge_steps(text, ranks_a)
    steps_b = compute_merge_steps(text, ranks_b)

    ints_a = collect_intermediates(tree_a)
    ints_b = collect_intermediates(tree_b)

    shared = sorted(ints_a & ints_b)
    only_a = sorted(ints_a - ints_b)
    only_b = sorted(ints_b - ints_a)

    initial_bytes = [chr(b) if 32 <= b < 127 else f"0x{b:02x}"
                     for b in text.encode("utf-8")]

    return {
        "text": text,
        "initial_bytes": initial_bytes,
        "tokenizer_a": {
            "name": name_a,
            "trees": [node_to_dict(n) for n in tree_a],
            "steps": [
                {
                    "step": s.step,
                    "merged_token": s.merged_token,
                    "rank": s.rank,
                    "tokens_after": s.tokens_after,
                }
                for s in steps_a
            ],
            "final_tokens": [n for n in (steps_a[-1].tokens_after if steps_a else initial_bytes)],
        },
        "tokenizer_b": {
            "name": name_b,
            "trees": [node_to_dict(n) for n in tree_b],
            "steps": [
                {
                    "step": s.step,
                    "merged_token": s.merged_token,
                    "rank": s.rank,
                    "tokens_after": s.tokens_after,
                }
                for s in steps_b
            ],
            "final_tokens": [n for n in (steps_b[-1].tokens_after if steps_b else initial_bytes)],
        },
        "conflict_analysis": {
            "shared_intermediates": shared,
            "only_a": only_a,
            "only_b": only_b,
            "is_compatible": len(only_a) == 0 and len(only_b) == 0,
            "conflict_count": len(only_a) + len(only_b),
        },
    }
