"""
U-statistic boundary metrics (u_mean, u_median, u_mean_weighted, u_mean_log_weighted).

Reference
---------
Lopardo, A., Salim, L. F., Srivastava, S., Jiang, K., Sharma, M., & Arnett, C. (2026).
"No Single Tokenizer Feature Reliably Predicts Downstream Language Model Performance."
arXiv:2610.21679. https://arxiv.org/abs/2610.21679
Reference implementation: https://github.com/AntonioLopardo/tokenizer-metric-multilingual-correlations
(intrinsic-metrics/).

The per-token score is the U-statistic of Hu et al. (2025), "Entropy-Driven
Pre-Tokenization for Byte-Pair Encoding" (Tokenization Workshop):

    Ustat(w) = PMI_min(w) + lambda * min(H_left(w), H_right(w))        (lambda = 4.0)

    PMI(ci, ci+1) = log( f(ci ci+1) * T / (f(ci) * f(ci+1)) )   T = total character bigrams
    PMI_min(w)    = min over adjacent character pairs inside w
    H_left / H_right = Shannon entropy of the characters immediately left / right
                       of w across all occurrences of w in the corpus

Aggregates over the vocabulary V of tokens with a valid score (>= 2 characters and
present in the corpus):

    u_mean, u_median
    u_mean_weighted     = sum f(w) Ustat(w) / sum f(w)
    u_mean_log_weighted = sum log(1+f(w)) Ustat(w) / sum log(1+f(w))

f(w) is the frequency of token w when the corpus is encoded by the tokenizer.

Note: the paper text does not specify how word/space boundaries are handled; the
``across_spaces`` option is this implementation's interpretation. When True (default)
whitespace is an ordinary character and statistics cross word boundaries; when False the
corpus is split on whitespace so bigrams and neighbours never cross a space, and a
word edge counts as its own "boundary" neighbour symbol.
"""

import math
import statistics
from collections import Counter, defaultdict
from typing import Any

from app.services.adapter import TokenizerAdapter

CITATION = (
    "Lopardo, A., Salim, L. F., Srivastava, S., Jiang, K., Sharma, M., & Arnett, C. (2026). "
    "No Single Tokenizer Feature Reliably Predicts Downstream Language Model Performance. "
    "arXiv:2610.21679. U-statistic from Hu, Y. et al. (2025), "
    "Entropy-Driven Pre-Tokenization for Byte-Pair Encoding."
)
PAPER_URL = "https://arxiv.org/abs/2610.21679"
CODE_URL = "https://github.com/AntonioLopardo/tokenizer-metric-multilingual-correlations"

DEFAULT_LAMBDA = 4.0
MAX_TOKEN_CHARS = 32
_EDGE = "\x00"  # sentinel neighbour for word edges when across_spaces=False


def _entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total == 0:
        return 0.0
    return -sum((c / total) * math.log(c / total) for c in counter.values())


def _segments(texts: list[str], across_spaces: bool) -> list[str]:
    if across_spaces:
        return [t for t in texts if t]
    return [w for t in texts for w in t.split()]


def compute_ustat(
    adapter: TokenizerAdapter,
    texts: list[str],
    lam: float = DEFAULT_LAMBDA,
    across_spaces: bool = True,
    top_k: int = 15,
) -> dict[str, Any]:
    texts = [t for t in texts if t and t.strip()]
    if not texts:
        raise ValueError("corpus is empty")

    # 1. Character / bigram statistics
    segments = _segments(texts, across_spaces)
    uni: Counter = Counter()
    bi: Counter = Counter()
    for seg in segments:
        uni.update(seg)
        bi.update(seg[i : i + 2] for i in range(len(seg) - 1))
    total_bigrams = sum(bi.values())
    if total_bigrams == 0:
        raise ValueError("corpus too short to compute character bigram statistics")

    # 2. Token frequencies from encoding the corpus, and token strings
    freq: Counter = Counter()
    for t in texts:
        freq.update(adapter.encode(t))

    token_strs: dict[int, str] = {}
    for tid in set(adapter.get_vocab().values()):
        s = adapter.decode([tid])
        if (
            2 <= len(s) <= MAX_TOKEN_CHARS
            and "�" not in s
            and (across_spaces or not any(ch.isspace() for ch in s))
        ):
            token_strs[tid] = s

    str_to_ids: dict[str, list[int]] = defaultdict(list)
    for tid, s in token_strs.items():
        str_to_ids[s].append(tid)
    wanted = set(str_to_ids)
    lengths = sorted({len(s) for s in wanted})

    # 3. Left / right neighbour counts for each candidate token string
    left: dict[str, Counter] = defaultdict(Counter)
    right: dict[str, Counter] = defaultdict(Counter)
    for seg in segments:
        n = len(seg)
        for L in lengths:
            for i in range(n - L + 1):
                sub = seg[i : i + L]
                if sub not in wanted:
                    continue
                if i > 0:
                    left[sub][seg[i - 1]] += 1
                elif not across_spaces:
                    left[sub][_EDGE] += 1
                if i + L < n:
                    right[sub][seg[i + L]] += 1
                elif not across_spaces:
                    right[sub][_EDGE] += 1

    # 4. Score tokens
    def pmi(a: str, b: str) -> float:
        return math.log(bi[a + b] * total_bigrams / (uni[a] * uni[b]))

    scores: list[tuple[str, int, float, float, float]] = []  # token, f, U, pmi_min, h_min
    for s, ids in str_to_ids.items():
        if s not in left and s not in right:
            continue  # never occurs in corpus -> no valid score
        if any(bi[s[i : i + 2]] == 0 for i in range(len(s) - 1)):
            continue
        pmi_min = min(pmi(s[i], s[i + 1]) for i in range(len(s) - 1))
        h_min = min(_entropy(left[s]), _entropy(right[s]))
        u = pmi_min + lam * h_min
        f = sum(freq.get(i, 0) for i in ids)
        scores.append((s, f, u, pmi_min, h_min))

    if not scores:
        raise ValueError("no vocabulary tokens with a valid U-statistic in this corpus")

    us = [x[2] for x in scores]
    fs = [x[1] for x in scores]
    f_sum = sum(fs)
    lw = [math.log1p(f) for f in fs]

    def row(x: tuple[str, int, float, float, float]) -> dict[str, Any]:
        return {"token": x[0], "freq": x[1], "ustat": x[2], "pmi_min": x[3], "h_min": x[4]}

    ranked = sorted(scores, key=lambda x: x[2])
    return {
        "u_mean": statistics.fmean(us),
        "u_median": statistics.median(us),
        "u_mean_weighted": (sum(f * u for f, u in zip(fs, us)) / f_sum) if f_sum else None,
        "u_mean_log_weighted": sum(w * u for w, u in zip(lw, us)) / sum(lw) if sum(lw) else None,
        "n_scored_tokens": len(scores),
        "n_vocab": len(set(adapter.get_vocab().values())),
        "n_chars": sum(len(t) for t in texts),
        "lambda": lam,
        "across_spaces": across_spaces,
        "highest": [row(x) for x in ranked[::-1][:top_k]],
        "lowest": [row(x) for x in ranked[:top_k]],
        "citation": CITATION,
        "paper_url": PAPER_URL,
        "code_url": CODE_URL,
    }
