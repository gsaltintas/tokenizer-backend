import unicodedata

from app.services.adapter import TokenizerAdapter


def _has_unusual_bytes(token_bytes: bytes) -> bool:
    """Check if token contains unusual byte sequences."""
    for b in token_bytes:
        # Control characters (except common whitespace)
        if b < 0x20 and b not in (0x09, 0x0A, 0x0D):
            return True
        # DEL character
        if b == 0x7F:
            return True
    # Check for incomplete UTF-8 sequences
    try:
        token_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return True
    return False


def _simulate_bpe(
    token_bytes: bytes,
    merge_ranks: dict[tuple[bytes, bytes], int],
    byte_to_rank: dict[bytes, int],
) -> list[bytes]:
    """
    Simulate BPE encoding of a byte sequence using the given merge rules.
    Returns the list of token pieces after all applicable merges.
    """
    # Start with individual bytes
    pieces = [bytes([b]) for b in token_bytes]

    while len(pieces) > 1:
        # Find the pair with the lowest merge rank
        best_pair = None
        best_rank = float("inf")
        best_idx = -1

        for i in range(len(pieces) - 1):
            pair = (pieces[i], pieces[i + 1])
            rank = merge_ranks.get(pair)
            if rank is not None and rank < best_rank:
                best_rank = rank
                best_pair = pair
                best_idx = i

        if best_pair is None:
            break  # No more merges possible

        # Apply the merge
        merged = best_pair[0] + best_pair[1]
        pieces = pieces[:best_idx] + [merged] + pieces[best_idx + 2 :]

    return pieces


def detect_undertrained_tokens(adapter: TokenizerAdapter) -> list[dict]:
    """
    Detect under-trained tokens using BPE merge reachability analysis.

    A token is "under-trained" if:
    1. It exists in the vocabulary but cannot be produced by the BPE merge process
    2. It contains unusual byte sequences
    3. It's never the result of any merge rule
    """
    merges_raw = adapter.get_merges()
    if merges_raw is None:
        return []

    vocab = adapter.get_vocab()

    # Build merge rank lookup (bytes-based)
    merge_ranks: dict[tuple[bytes, bytes], int] = {}
    merge_results: set[bytes] = set()

    for rank, (left_str, right_str) in enumerate(merges_raw):
        left_bytes = left_str.encode("utf-8", errors="replace")
        right_bytes = right_str.encode("utf-8", errors="replace")
        merge_ranks[(left_bytes, right_bytes)] = rank
        merge_results.add(left_bytes + right_bytes)

    # Track which tokens are base tokens (single bytes)
    byte_ranks: dict[bytes, int] = {}
    for token_str, token_id in vocab.items():
        tb = token_str.encode("utf-8", errors="replace")
        if len(tb) == 1:
            byte_ranks[tb] = token_id

    undertrained = []

    for token_str, token_id in vocab.items():
        token_bytes = token_str.encode("utf-8", errors="replace")

        # Skip single-byte tokens (they're base vocabulary)
        if len(token_bytes) <= 1:
            continue

        # Skip special tokens (usually enclosed in <> or similar)
        if token_str.startswith("<") and token_str.endswith(">"):
            continue

        reasons = []
        confidence = 0.0

        # Check 1: Simulate BPE and see if this token is reachable
        simulated = _simulate_bpe(token_bytes, merge_ranks, byte_ranks)
        simulated_single = len(simulated) == 1 and simulated[0] == token_bytes
        if not simulated_single:
            reasons.append("unreachable via BPE merges")
            confidence = max(confidence, 0.8)

        # Check 2: Is this token ever the result of a merge?
        if token_bytes not in merge_results and len(token_bytes) > 1:
            reasons.append("not a product of any merge rule")
            confidence = max(confidence, 0.6)

        # Check 3: Unusual bytes
        if _has_unusual_bytes(token_bytes):
            reasons.append("contains unusual byte sequences")
            confidence = max(confidence, 0.5)

        # Check 4: Partial UTF-8
        try:
            token_bytes.decode("utf-8")
        except UnicodeDecodeError:
            reasons.append("incomplete UTF-8 sequence")
            confidence = max(confidence, 0.7)

        if not reasons:
            continue

        # Build expected merge path
        expected_path = [
            s.decode("utf-8", errors="replace") for s in simulated
        ]

        undertrained.append(
            {
                "token_id": token_id,
                "token_str": token_str,
                "token_bytes_hex": token_bytes.hex(),
                "reason": "; ".join(reasons),
                "confidence": round(confidence, 2),
                "expected_merge_path": [token_str],
                "actual_merge_result": expected_path,
            }
        )

    # Sort by confidence (highest first)
    undertrained.sort(key=lambda t: t["confidence"], reverse=True)
    return undertrained
