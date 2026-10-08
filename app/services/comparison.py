from app.services.adapter import HuggingFaceAdapter, TokenizerAdapter
from app.services.pretokenize import analyze
from app.services.tokens import build_token_infos

DEFAULT_SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models process natural language by breaking text into tokens.",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "El rápido zorro marrón salta sobre el perro perezoso.",
    "日本語のテキストをトークン化するテスト。",
    "Привет мир! Это тестовое предложение на русском языке.",
    "SELECT * FROM users WHERE age > 18 ORDER BY name;",
    "https://example.com/path?key=value&other=123#section",
    "Hello! 😊 How are you doing today? I'm great! 🎉",
    "3.14159265358979323846264338327950288",
]


def compute_overlap(adapters: dict[str, TokenizerAdapter]) -> dict:
    """Compute vocabulary overlap between tokenizers."""
    vocab_sets: dict[str, set[str]] = {}
    for tok_id, adapter in adapters.items():
        vocab_sets[tok_id] = set(adapter.get_vocab().keys())

    all_ids = list(vocab_sets.keys())

    # Compute intersection of all
    shared = set.intersection(*vocab_sets.values()) if vocab_sets else set()
    union = set.union(*vocab_sets.values()) if vocab_sets else set()

    # Unique per tokenizer
    unique_per = {}
    for tok_id, vocab in vocab_sets.items():
        others = set.union(*(v for k, v in vocab_sets.items() if k != tok_id))
        unique_per[tok_id] = len(vocab - others)

    overlap_pct = (len(shared) / max(len(union), 1)) * 100

    return {
        "shared_tokens": len(shared),
        "unique_per_tokenizer": unique_per,
        "total_union": len(union),
        "overlap_percentage": round(overlap_pct, 2),
        "shared_sample": sorted(list(shared))[:50],
        "unique_samples": {
            tok_id: sorted(list(vocab_sets[tok_id] - shared))[:30]
            for tok_id in all_ids
        },
    }


def compare_tokenization(
    adapters: dict[str, TokenizerAdapter], text: str
) -> list[dict]:
    """Compare how different tokenizers tokenize the same text."""
    results = []
    for tok_id, adapter in adapters.items():
        tokens = [t.model_dump() for t in build_token_infos(adapter, text)]
        results.append(
            {
                "tokenizer_id": tok_id,
                "tokens": tokens,
                "token_count": len(tokens),
            }
        )
    return results


def compute_efficiency(
    adapters: dict[str, TokenizerAdapter],
    sample_texts: list[str] | None = None,
) -> list[dict]:
    """Compare tokenization efficiency across tokenizers."""
    texts = sample_texts or DEFAULT_SAMPLE_TEXTS

    results = []
    for tok_id, adapter in adapters.items():
        total_tokens = 0
        total_chars = 0
        total_words = 0

        for text in texts:
            token_ids = adapter.encode(text)
            total_tokens += len(token_ids)
            total_chars += len(text)
            total_words += len(text.split())

        avg_tokens_per_word = total_tokens / max(total_words, 1)
        avg_token_length = total_chars / max(total_tokens, 1)

        results.append(
            {
                "tokenizer_id": tok_id,
                "avg_tokens_per_word": round(avg_tokens_per_word, 3),
                "avg_token_length_chars": round(avg_token_length, 3),
                "total_tokens": total_tokens,
                "total_chars": total_chars,
            }
        )

    return results


def _boundaries(spans: list[tuple[int, int]], text_len: int) -> set[int]:
    """Interior offsets where a span starts or ends."""
    return {b for span in spans for b in span if 0 < b < text_len}


def _boundary_f1(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    return 2 * len(a & b) / (len(a) + len(b))


def _hf_original_spans(
    adapter: HuggingFaceAdapter, text: str
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]] | None:
    """Chunk and token spans in the input text, via the tokenizers library's alignment
    tracking, so they stay right when the normalizer rewrites the text (lowercasing,
    accent stripping, spaces around CJK, ...)."""
    from tokenizers import PreTokenizedString

    try:
        bt = adapter._tokenizer.backend_tokenizer
    except AttributeError:  # slow tokenizer
        return None
    pts = PreTokenizedString(text)
    if bt.normalizer is not None:
        pts.normalize(lambda ns: bt.normalizer.normalize(ns))
    if bt.pre_tokenizer is not None:
        bt.pre_tokenizer.pre_tokenize(pts)
    chunk_spans = [o for _, o, _ in pts.get_splits(offset_referential="original", offset_type="char")]
    token_spans = list(bt.encode(text, add_special_tokens=False).offsets)
    return chunk_spans, token_spans


def compare_pretokenization(adapters: dict[str, TokenizerAdapter], text: str) -> dict:
    """Normalization, pre-token chunks and final tokens per tokenizer, with boundaries
    on the input's character offsets so they can be compared across tokenizers.

    Hugging Face tokenizers report offsets into the input even when normalization
    rewrites the text.  For the others, chunks are found in the normalized text, so
    when normalization changes it their offsets are only approximate."""
    results = []
    for tok_id, adapter in adapters.items():
        pre = analyze(adapter, text)
        tokens = build_token_infos(adapter, text)
        hf_spans = _hf_original_spans(adapter, text) if isinstance(adapter, HuggingFaceAdapter) else None
        if hf_spans is not None:
            chunk_spans, token_spans = hf_spans
            # Show input text: byte-level chunks would otherwise read like 'Ġworld'
            chunk_texts = [text[s:e] for s, e in chunk_spans]
            approximate = False
        else:
            chunk_spans = [tuple(sp) for sp in pre.chunk_spans]
            token_spans = [(t.start, t.end) for t in tokens]
            chunk_texts = pre.chunks
            approximate = pre.normalization.changed

        chunk_bounds = _boundaries(chunk_spans, len(text))
        token_bounds = _boundaries(token_spans, len(text))
        crossing = sum(1 for ts, te in token_spans if any(ts < b < te for b in chunk_bounds))
        results.append(
            {
                "tokenizer_id": tok_id,
                "normalization_type": pre.normalization.type,
                "normalized_text": pre.normalization.normalized_text,
                "normalization_changed": pre.normalization.changed,
                "pretokenizer_type": pre.pretokenizer_type,
                "pretokenizer_description": pre.pretokenizer_description,
                "regex_pattern": pre.regex_pattern,
                "chunks": [
                    {"text": c, "start": s, "end": e} for c, (s, e) in zip(chunk_texts, chunk_spans)
                ],
                "tokens": tokens,
                "token_spans": token_spans,
                "chunk_boundaries": sorted(chunk_bounds),
                "token_boundaries": sorted(token_bounds),
                "offsets_approximate": approximate,
                "tokens_per_chunk": round(len(tokens) / max(len(chunk_spans), 1), 3),
                "chunk_crossing_tokens": crossing,
            }
        )

    def agreement(key: str) -> list[list[float]]:
        sets = [set(r[key]) for r in results]
        return [[round(_boundary_f1(a, b), 4) for b in sets] for a in sets]

    return {
        "text": text,
        "results": results,
        "chunk_agreement": agreement("chunk_boundaries"),
        "token_agreement": agreement("token_boundaries"),
    }
