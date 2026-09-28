"""
Bridge from our TokenizerAdapter to tokenizer-intrinsic-evals' TokenizerWrapper.

The sanity check and the token-boundary visualization in the library run the
tokenizer "faithfully": they read its normalizer, pre-tokenizer and decoder,
and use character offsets from its own encoder. So rather than wrapping our
adapters' encode()/decode(), we hand the library the same underlying object
its CLI would load:

- Hugging Face  -> HuggingFaceTokenizer around the transformers tokenizer
- SentencePiece -> SentencePieceTokenizer around the SentencePieceProcessor
- tiktoken      -> CustomBPETokenizer around a tokenizers.Tokenizer converted
                   from the encoding's ranks and split regex (the conversion
                   transformers uses for tiktoken-based checkpoints)

TokenMonster has no library counterpart, so it gets a generic wrapper over the
adapter; checks that need introspection come out ``unverifiable`` for it.
"""

import re
from functools import lru_cache
from typing import Any, Optional, Set

from app.services.adapter import (
    HuggingFaceAdapter,
    SentencePieceAdapter,
    TiktokenAdapter,
    TokenizerAdapter,
    _gpt2_unicode_to_bytes,
)


def to_tokeval_wrapper(adapter: TokenizerAdapter):
    """Return a cached tokenizer_analysis TokenizerWrapper for *adapter*."""
    return _wrapper_for(adapter)


@lru_cache(maxsize=10)
def _wrapper_for(adapter: TokenizerAdapter):
    from tokenizer_analysis.core.tokenizer_wrapper import (
        CustomBPETokenizer,
        HuggingFaceTokenizer,
        SentencePieceTokenizer,
    )

    name = adapter.name
    if isinstance(adapter, HuggingFaceAdapter):
        return HuggingFaceTokenizer(name, adapter._tokenizer, {})
    if isinstance(adapter, SentencePieceAdapter):
        return SentencePieceTokenizer(name, adapter._sp, {})
    if isinstance(adapter, TiktokenAdapter):
        return CustomBPETokenizer(name, _convert_tiktoken(adapter._encoding), {})
    return _AdapterWrapper(adapter)


def _convert_tiktoken(encoding):
    """Build a byte-level BPE tokenizers.Tokenizer equivalent to a tiktoken encoding.

    Same construction as transformers' TikTokenConverter, but read from the
    loaded Encoding (its ranks, split pattern and special-token ids) instead of
    a .tiktoken file, so token ids match tiktoken's exactly.
    """
    from tokenizers import AddedToken, Regex, Tokenizer, decoders, pre_tokenizers, processors
    from tokenizers.models import BPE

    byte_to_unicode = {b: ch for ch, b in _gpt2_unicode_to_bytes().items()}

    def to_str(b: bytes) -> str:
        return "".join(byte_to_unicode[x] for x in b)

    ranks: dict[bytes, int] = encoding._mergeable_ranks
    vocab = {to_str(tok): rank for tok, rank in ranks.items()}
    merges = []
    for tok, rank in ranks.items():
        for i in range(1, len(tok)):
            left, right = tok[:i], tok[i:]
            if left in ranks and right in ranks:
                merges.append((rank, ranks[left], ranks[right], left, right))
    merges.sort()
    specials: dict[str, int] = dict(getattr(encoding, "_special_tokens", {}))
    vocab.update(specials)

    tok = Tokenizer(BPE(vocab, [(to_str(m[3]), to_str(m[4])) for m in merges], fuse_unk=False))
    if hasattr(tok.model, "ignore_merges"):
        tok.model.ignore_merges = True
    # Oniguruma reads `{m,n}+` as a repeated interval, not a possessive one:
    # cl100k's `\p{N}{1,3}+` would keep "1234567890" as one pre-token instead
    # of splitting it into groups of three. Possessiveness does not change which
    # string that sub-pattern matches, so drop it.
    pattern = re.sub(r"(\{\d+(?:,\d*)?\})\+", r"\1", encoding._pat_str)
    tok.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(Regex(pattern), behavior="isolated", invert=False),
        pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=False),
    ])
    tok.decoder = decoders.ByteLevel()
    tok.post_processor = processors.ByteLevel(trim_offsets=False)
    if specials:
        tok.add_special_tokens([AddedToken(s, normalized=False, special=True) for s in specials])
    return tok


def _base_wrapper_class():
    from tokenizer_analysis.core.tokenizer_wrapper import TokenizerWrapper
    return TokenizerWrapper


class _AdapterWrapper(_base_wrapper_class()):  # type: ignore[misc]
    """Generic wrapper for adapters the library has no loader for (TokenMonster)."""

    def __init__(self, adapter: TokenizerAdapter):
        self._adapter = adapter

    def get_name(self) -> str:
        return self._adapter.name

    def get_vocab_size(self) -> int:
        return self._adapter.vocab_size()

    def get_vocab(self):
        return self._adapter.get_vocab()

    def can_encode(self) -> bool:
        return True

    def encode(self, text: str):
        return list(self._adapter.encode(text))

    def can_decode(self) -> bool:
        return True

    def decode(self, token_ids, skip_special_tokens: bool = True) -> Optional[str]:
        return self._adapter.decode(list(token_ids))

    def can_pretokenize(self) -> bool:
        return False

    def pretokenize(self, text: str):
        raise NotImplementedError(f"{self.get_name()} does not expose a pre-tokenizer")

    def get_special_token_strings(self) -> Optional[Set[str]]:
        # The adapter cannot report them; None makes the library fall back to
        # its generic list and say so.
        return None

    @classmethod
    def from_config(cls, name: str, config: dict[str, Any]):
        raise NotImplementedError("constructed from a loaded adapter only")
