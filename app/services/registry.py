import os
from collections import OrderedDict

from app.services.adapter import (
    HuggingFaceAdapter,
    SentencePieceAdapter,
    TiktokenAdapter,
    TokenizerAdapter,
)

# Known tiktoken encoding names
TIKTOKEN_ENCODINGS = {
    "gpt-4o": "o200k_base",
    "gpt-4": "cl100k_base",
    "gpt-3.5-turbo": "cl100k_base",
    "cl100k_base": "cl100k_base",
    "o200k_base": "o200k_base",
    "p50k_base": "p50k_base",
    "r50k_base": "r50k_base",
    "gpt2": "gpt2",
}


class TokenizerRegistry:
    """Loads and caches tokenizer adapters with an LRU policy."""

    def __init__(self, max_cache_size: int = 10):
        self._cache: OrderedDict[str, TokenizerAdapter] = OrderedDict()
        self._max_cache_size = max_cache_size

    def load(self, name: str) -> TokenizerAdapter:
        """Load a tokenizer by name, HuggingFace model ID, or file path."""
        if name in self._cache:
            self._cache.move_to_end(name)
            return self._cache[name]

        adapter = self._create_adapter(name)
        self._cache[name] = adapter
        if len(self._cache) > self._max_cache_size:
            self._cache.popitem(last=False)
        return adapter

    def _create_adapter(self, name: str) -> TokenizerAdapter:
        # 1. Check if it's a tiktoken encoding
        if name in TIKTOKEN_ENCODINGS:
            encoding_name = TIKTOKEN_ENCODINGS[name]
            return TiktokenAdapter(encoding_name)

        # 2. Check if it's a .model file path (SentencePiece)
        if name.endswith(".model") and os.path.exists(name):
            return SentencePieceAdapter(name)

        # 3. Try as HuggingFace model ID
        try:
            return HuggingFaceAdapter(name)
        except Exception as e:
            raise ValueError(
                f"Could not load tokenizer '{name}'. "
                f"Tried tiktoken presets, file path, and HuggingFace. "
                f"HuggingFace error: {e}"
            )

    def get(self, name: str) -> TokenizerAdapter | None:
        """Get a cached tokenizer, or None if not loaded."""
        if name in self._cache:
            self._cache.move_to_end(name)
            return self._cache[name]
        return None

    def list_loaded(self) -> list[dict]:
        """List all currently loaded tokenizers."""
        return [
            {
                "id": name,
                "name": adapter.name,
                "tokenizer_type": adapter.tokenizer_type,
                "vocab_size": adapter.vocab_size(),
                "source": adapter.source,
            }
            for name, adapter in self._cache.items()
        ]

    def list_available(self) -> list[dict]:
        """List available preset tokenizers."""
        presets = []
        seen_encodings = set()
        presets.extend([{
            "id": "meta-llama/Llama-3.2-1B",
            "name": "meta-llama/Llama-3.2-1B",
            "vocab_size": 0,
            "tokenizer_type": "bpe",
            "source": "huggingface",
        },
        {
            "id": "Qwen/Qwen3-8B",
            "name": "Qwen/Qwen3-8B",
            "vocab_size": 0,
            "tokenizer_type": "bpe",
            "source": "huggingface",
        },
        {
            "id": "google/gemma-2-2b",
            "name": "google/gemma-2-2b",
            "vocab_size": 0,
            "tokenizer_type": "bpe",
            "source": "huggingface",
        },])
        for alias, encoding in TIKTOKEN_ENCODINGS.items():
            if encoding not in seen_encodings:
                presets.append(
                    {
                        "id": alias,
                        "name": alias,
                        "tokenizer_type": "bpe",
                        "vocab_size": 0,  # Unknown until loaded
                        "source": "tiktoken",
                    }
                )
                seen_encodings.add(encoding)
        return presets


# Global singleton
registry = TokenizerRegistry()
