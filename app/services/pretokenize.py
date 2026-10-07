from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

from app.services.adapter import (
    HuggingFaceAdapter,
    ScriptTokAdapter,
    SentencePieceAdapter,
    TiktokenAdapter,
    TokenizerAdapter,
)


@dataclass
class NormalizationInfo:
    type: str
    normalized_text: str
    changed: bool


@dataclass
class PretokenizeResult:
    normalization: NormalizationInfo
    chunks: list[str]
    chunk_spans: list[tuple[int, int]]  # (start, end) char offsets in normalized text
    chunk_count: int
    pretokenizer_type: str
    pretokenizer_description: str
    regex_pattern: str | None


def _norm_info(raw: str, norm_type: str) -> NormalizationInfo:
    form_map = {"NFC": "NFC", "NFD": "NFD", "NFKC": "NFKC", "NFKD": "NFKD"}
    if norm_type in form_map:
        normalized = unicodedata.normalize(form_map[norm_type], raw)
    else:
        normalized = raw
    return NormalizationInfo(type=norm_type, normalized_text=normalized, changed=normalized != raw)


def _spans_from_chunks(text: str, chunks: list[str]) -> list[tuple[int, int]]:
    spans = []
    offset = 0
    for chunk in chunks:
        idx = text.find(chunk, offset)
        if idx == -1:
            idx = offset
        spans.append((idx, idx + len(chunk)))
        offset = idx + len(chunk)
    return spans


def analyze(adapter: TokenizerAdapter, text: str) -> PretokenizeResult:
    if isinstance(adapter, TiktokenAdapter):
        return _analyze_tiktoken(adapter, text)
    if isinstance(adapter, HuggingFaceAdapter):
        return _analyze_hf(adapter, text)
    if isinstance(adapter, SentencePieceAdapter):
        return _analyze_sp(adapter, text)
    if isinstance(adapter, ScriptTokAdapter):
        return _analyze_script_tok(adapter, text)
    # fallback — show tokens as chunks, no normalization
    norm = NormalizationInfo(type="none", normalized_text=text, changed=False)
    ids = adapter.encode(text)
    chunks = [adapter.decode([i]) for i in ids]
    return PretokenizeResult(
        normalization=norm,
        chunks=chunks,
        chunk_spans=_spans_from_chunks(text, chunks),
        chunk_count=len(chunks),
        pretokenizer_type="none",
        pretokenizer_description="No pretokenization information available for this tokenizer.",
        regex_pattern=None,
    )


def _analyze_tiktoken(adapter: TiktokenAdapter, text: str) -> PretokenizeResult:
    import regex  # tiktoken dependency

    enc = adapter._encoding
    norm = NormalizationInfo(type="none", normalized_text=text, changed=False)

    pat_str: str | None = getattr(enc, "_pat_str", None)
    if pat_str:
        chunks = regex.findall(pat_str, text)
        description = "Regex-based word splitting (applied to raw bytes before BPE merges)"
    else:
        # fall back to splitting on whitespace
        chunks = text.split()
        description = "Whitespace split (regex pattern unavailable)"
        pat_str = None

    return PretokenizeResult(
        normalization=norm,
        chunks=chunks,
        chunk_spans=_spans_from_chunks(text, chunks),
        chunk_count=len(chunks),
        pretokenizer_type="regex",
        pretokenizer_description=description,
        regex_pattern=pat_str,
    )


def _analyze_hf(adapter: HuggingFaceAdapter, text: str) -> PretokenizeResult:
    bt = adapter._tokenizer.backend_tokenizer

    # Normalization
    norm_type = "none"
    normalized_text = text
    if bt.normalizer is not None:
        try:
            normalized_text = bt.normalizer.normalize_str(text)
            # Infer type from class name
            cls = type(bt.normalizer).__name__.lower()
            if "nfkc" in cls:
                norm_type = "NFKC"
            elif "nfkd" in cls:
                norm_type = "NFKD"
            elif "nfc" in cls:
                norm_type = "NFC"
            elif "nfd" in cls:
                norm_type = "NFD"
            elif "lowercase" in cls or "bert" in cls:
                norm_type = "lowercase+strip_accents"
            elif "sequence" in cls:
                norm_type = "sequence"
            else:
                norm_type = cls
        except Exception:
            normalized_text = text
    norm = NormalizationInfo(type=norm_type, normalized_text=normalized_text, changed=normalized_text != text)

    # Pretokenization
    pretokenizer_type = "none"
    description = "No pretokenizer"
    chunks: list[str] = []
    spans: list[tuple[int, int]] = []

    if bt.pre_tokenizer is not None:
        cls = type(bt.pre_tokenizer).__name__.lower()
        if "bytelevel" in cls or "byte_level" in cls:
            pretokenizer_type = "byte_level"
            description = "ByteLevel: adds a space prefix, maps characters to byte representations"
        elif "metaspace" in cls:
            pretokenizer_type = "metaspace"
            description = "Metaspace: replaces spaces with ▁, splits on whitespace"
        elif "whitespace" in cls:
            pretokenizer_type = "whitespace"
            description = "Whitespace: splits on whitespace boundaries"
        elif "punctuation" in cls:
            pretokenizer_type = "punctuation"
            description = "Punctuation: splits on punctuation characters"
        elif "sequence" in cls:
            pretokenizer_type = "sequence"
            description = "Sequence of pretokenizers applied in order"
        elif "bert" in cls:
            pretokenizer_type = "bert_whitespace+punctuation"
            description = "BERT: splits on whitespace and punctuation"
        else:
            pretokenizer_type = cls
            description = f"Pretokenizer: {cls}"

        try:
            pairs = bt.pre_tokenizer.pre_tokenize_str(normalized_text)
            chunks = [p[0] for p in pairs]
            spans = [p[1] for p in pairs]
        except Exception:
            chunks = normalized_text.split()
            spans = _spans_from_chunks(normalized_text, chunks)
    else:
        chunks = [normalized_text]
        spans = [(0, len(normalized_text))]

    return PretokenizeResult(
        normalization=norm,
        chunks=chunks,
        chunk_spans=spans,
        chunk_count=len(chunks),
        pretokenizer_type=pretokenizer_type,
        pretokenizer_description=description,
        regex_pattern=None,
    )


def _analyze_sp(adapter: SentencePieceAdapter, text: str) -> PretokenizeResult:
    # SentencePiece applies NFKC normalization internally by default
    normalized = unicodedata.normalize("NFKC", text)
    norm = NormalizationInfo(type="NFKC", normalized_text=normalized, changed=normalized != text)

    # SentencePiece uses Metaspace-style splitting: words separated by ▁
    pieces = adapter._sp.EncodeAsPieces(text)
    # Reconstruct display chunks (pieces without ▁ prefix shown as space-prefixed words)
    chunks = []
    for p in pieces:
        chunks.append(p.replace("▁", " ").lstrip())

    return PretokenizeResult(
        normalization=norm,
        chunks=chunks,
        chunk_spans=_spans_from_chunks(normalized, chunks),
        chunk_count=len(chunks),
        pretokenizer_type="metaspace",
        pretokenizer_description="SentencePiece Metaspace: NFKC normalization + whitespace splitting with ▁ prefix marker",
        regex_pattern=None,
    )


def _analyze_script_tok(adapter: ScriptTokAdapter, text: str) -> PretokenizeResult:
    pt = adapter._pretokenizer
    config = pt.config
    normalized = pt.normalize(text)
    norm = NormalizationInfo(
        type=config.normalization or "none", normalized_text=normalized, changed=normalized != text
    )

    chunks = [pt.decode(chunk) for chunk in pt.pretokenize(text)]
    steps = []
    if config.regex_pattern:
        steps.append("regex split")
    if config.digit_handling:
        steps.append(f"digits split ({config.digit_handling})")
    if adapter.byte_level:
        pretokenizer_type = "utf8"
        units = "UTF-8 bytes"
    else:
        pretokenizer_type = "script_encoding"
        units = "(script block, index) pairs per character (SCRIPT encoding)"
        if getattr(config, "script_split", False):
            steps.append("split where the Unicode script changes")
    if config.enforce_char_boundaries:
        steps.append("merges respect character boundaries")
    description = f"script_tok: {', '.join(steps) or 'no splitting'}; encoded as {units}"

    return PretokenizeResult(
        normalization=norm,
        chunks=chunks,
        chunk_spans=_spans_from_chunks(normalized, chunks),
        chunk_count=len(chunks),
        pretokenizer_type=pretokenizer_type,
        pretokenizer_description=description,
        regex_pattern=config.regex_pattern,
    )
