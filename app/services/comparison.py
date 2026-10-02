from app.services.adapter import TokenizerAdapter
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
