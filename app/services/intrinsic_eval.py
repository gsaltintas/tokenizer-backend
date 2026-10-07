"""
Intrinsic tokenizer evaluation metrics via tokenizer-intrinsic-evals + FLORES+.

Per-text metrics use per_example_all() from the library via a thin shim.
FLORES+ Gini uses the datasets library to load texts, tokenizes with our adapter,
and computes the Gini coefficient ourselves (avoids wrangling the library's full
corpus pipeline with custom adapters).
"""

import math
from typing import Any

from app.services.adapter import TokenizerAdapter

FLORES_LANGUAGES: list[dict] = [
    {"code": "eng_Latn", "name": "English", "script": "Latin"},
    {"code": "fra_Latn", "name": "French", "script": "Latin"},
    {"code": "deu_Latn", "name": "German", "script": "Latin"},
    {"code": "spa_Latn", "name": "Spanish", "script": "Latin"},
    {"code": "por_Latn", "name": "Portuguese", "script": "Latin"},
    {"code": "ita_Latn", "name": "Italian", "script": "Latin"},
    {"code": "cmn_Hans", "name": "Chinese (Simplified)", "script": "Han"},
    {"code": "jpn_Jpan", "name": "Japanese", "script": "Japanese"},
    {"code": "kor_Hang", "name": "Korean", "script": "Hangul"},
    {"code": "arb_Arab", "name": "Arabic", "script": "Arabic"},
    {"code": "pes_Arab", "name": "Farsi", "script": "Arabic"},
    {"code": "hin_Deva", "name": "Hindi", "script": "Devanagari"},
    {"code": "rus_Cyrl", "name": "Russian", "script": "Cyrillic"},
    {"code": "ukr_Cyrl", "name": "Ukrainian", "script": "Cyrillic"},
    {"code": "tha_Thai", "name": "Thai", "script": "Thai"},
    {"code": "vie_Latn", "name": "Vietnamese", "script": "Latin"},
    {"code": "swh_Latn", "name": "Swahili", "script": "Latin"},
    {"code": "tur_Latn", "name": "Turkish", "script": "Latin"},
    {"code": "heb_Hebr", "name": "Hebrew", "script": "Hebrew"},
    {"code": "ben_Beng", "name": "Bengali", "script": "Bengali"},
    {"code": "fin_Latn", "name": "Finnish", "script": "Latin"},
]


class _TokEvalShim:
    """Thin wrapper around TokenizerAdapter for tokenizer_analysis per_example_ functions.

    The library's per_example helpers call encode() and convert_ids_to_tokens() /
    id_to_token().  Our adapters expose encode() and get_vocab(), so we bridge the gap.
    """

    def __init__(self, adapter: TokenizerAdapter):
        self._adapter = adapter
        self._id_to_token: dict[int, str] | None = None

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return self._adapter.encode(text)

    def _ensure_reverse_vocab(self) -> None:
        if self._id_to_token is None:
            vocab = self._adapter.get_vocab()
            self._id_to_token = {v: k for k, v in vocab.items()}

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        self._ensure_reverse_vocab()
        assert self._id_to_token is not None
        return [self._id_to_token.get(i, f"<unk_{i}>") for i in ids]

    def id_to_token(self, i: int) -> str:
        self._ensure_reverse_vocab()
        assert self._id_to_token is not None
        return self._id_to_token.get(i, f"<unk_{i}>")

    def get_vocab(self) -> dict[str, int]:
        return self._adapter.get_vocab()


def per_text_metrics(adapter: TokenizerAdapter, text: str) -> dict[str, Any]:
    """Return per-text intrinsic metrics for the given text using the library."""
    from tokenizer_analysis.per_example import per_example_all

    shim = _TokEvalShim(adapter)
    raw = per_example_all(shim, text)

    # Pick the keys we want to surface and give them friendly names
    return {
        "n_tokens": raw.get("n_tokens"),
        "n_words": raw.get("n_words"),
        "n_bytes": raw.get("n_bytes"),
        "n_chars": raw.get("n_chars"),
        "fertility_words": _safe_float(raw.get("fertility_words")),
        "bytes_per_token": _safe_float(raw.get("bytes_per_token")),
        "chars_per_token": _safe_float(raw.get("chars_per_token")),
        "tokens_per_byte": _safe_float(raw.get("tokens_per_byte")),
        # UTF-8 integrity
        "integrity_rate": _safe_float(raw.get("integrity_rate")),
        "boundary_crossing_rate": _safe_float(raw.get("boundary_crossing_rate")),
        "byte_fallback_rate": _safe_float(raw.get("byte_fallback_rate")),
    }


def flores_gini_metrics(
    adapter: TokenizerAdapter,
    language_codes: list[str],
    n_samples: int = 200,
) -> dict[str, Any]:
    """Load FLORES+ for each language, tokenize, compute per-language fertility
    and compression rate, then return the Gini coefficient across languages."""
    from datasets import load_dataset

    per_language: list[dict[str, Any]] = []

    for lang_code in language_codes:
        try:
            ds = load_dataset(
                "openlanguagedata/flores_plus",
                lang_code,
                split="devtest",
            )
            texts: list[str] = ds["text"][:n_samples]  # type: ignore[index]
        except Exception as exc:
            per_language.append({
                "code": lang_code,
                "error": str(exc),
            })
            continue

        n_tokens_total = 0
        n_bytes_total = 0
        n_words_total = 0
        n_integrity_valid = 0
        n_content_tokens = 0

        shim = _TokEvalShim(adapter)
        for text in texts:
            if not text.strip():
                continue
            ids = adapter.encode(text)
            n_tokens_total += len(ids)
            n_bytes_total += len(text.encode("utf-8"))
            n_words_total += len(text.split())

            # UTF-8 integrity via library
            try:
                from tokenizer_analysis.per_example import per_example_utf8_integrity
                ut = per_example_utf8_integrity(shim, text)
                n_integrity_valid += ut.get("n_valid_complete", 0)
                n_content_tokens += ut.get("n_content_tokens", len(ids))
            except Exception:
                n_integrity_valid += len(ids)
                n_content_tokens += len(ids)

        fertility = (n_tokens_total / n_words_total) if n_words_total else None
        compression = (n_tokens_total / n_bytes_total) if n_bytes_total else None
        integrity = (n_integrity_valid / n_content_tokens) if n_content_tokens else None

        per_language.append({
            "code": lang_code,
            "n_texts": len(texts),
            "n_tokens": n_tokens_total,
            "n_bytes": n_bytes_total,
            "n_words": n_words_total,
            "fertility": _safe_float(fertility),
            "tokens_per_byte": _safe_float(compression),
            "integrity_rate": _safe_float(integrity),
        })

    # Compute Gini over tokens_per_byte values across languages that succeeded
    valid = [r for r in per_language if "tokens_per_byte" in r and r["tokens_per_byte"] is not None]
    gini_value = _gini([r["tokens_per_byte"] for r in valid]) if len(valid) >= 2 else None

    return {
        "gini": _safe_float(gini_value),
        "n_languages": len(valid),
        "per_language": per_language,
    }


def _gini(values: list[float]) -> float:
    """Gini coefficient of a list of non-negative values."""
    n = len(values)
    if n == 0:
        return 0.0
    s = sorted(values)
    total = sum(s)
    if total == 0:
        return 0.0
    cumsum = sum((i + 1) * v for i, v in enumerate(s))
    return (2 * cumsum / (n * total)) - (n + 1) / n


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
        return None if math.isnan(f) or math.isinf(f) else f
    except (TypeError, ValueError):
        return None
