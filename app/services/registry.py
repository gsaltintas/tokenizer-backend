import os
from collections import OrderedDict

from app.services.adapter import (
    HuggingFaceAdapter,
    ScriptTokAdapter,
    SentencePieceAdapter,
    TiktokenAdapter,
    TokenMonsterAdapter,
    TokenizerAdapter,
)

# Known TokenMonster preset vocabulary names
TOKENMONSTER_VOCABS = {
    "english-32000-consistent-v1",
    "english-24000-consistent-v1",
    "englishcode-32000-consistent-v1",
    "englishcode-32000-balanced-v1",
    "code-32000-consistent-v1",
    "fiction-24000-consistent-v1",
}

# Tokenizers trained with script_tok (https://github.com/sanderland/script_tok) are loaded as
# "script_tok:<hub repo>[/<file in repo>]". They are not bundled: the saved .json.gz is
# downloaded from the Hugging Face Hub into HF_HUB_CACHE, which serve.sh puts on the node's
# scratch disk, like every other Hub tokenizer. A local .json/.json.gz path also works.
SCRIPT_TOK_PREFIX = "script_tok:"
SCRIPT_TOK_SUFFIXES = (".json.gz", ".json")
# From "Explicit Boundary Markers for Subword Vocabularies" (arXiv 2608.08847). Only the
# plain ones: the bnd_* tokenizers need pretokenizer classes from script_tok's paper_utils,
# which isn't part of the installed package.
SCRIPT_TOK_PRESETS = [
    f"{SCRIPT_TOK_PREFIX}cmeister/boundary-markers-{lang}-d12-plain-{model}"
    for lang in ("en", "ko", "ru")
    for model in ("bpe", "mingram")
]


def download_script_tok(spec: str) -> str:
    """Download "<repo>" or "<repo>/<file>" from the Hub; returns the local path. With only
    a repo, it must hold exactly one .json/.json.gz file under tokenizer/."""
    from huggingface_hub import hf_hub_download, list_repo_files

    owner, repo, *rest = spec.split("/", 2)
    repo_id = f"{owner}/{repo}"
    if rest:
        filename = rest[0]
    else:
        candidates = [
            f for f in list_repo_files(repo_id)
            if f.startswith("tokenizer/") and f.endswith(SCRIPT_TOK_SUFFIXES)
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"Expected one script_tok tokenizer under tokenizer/ in {repo_id}, "
                f"found {candidates}; name the file as {SCRIPT_TOK_PREFIX}{repo_id}/<file>"
            )
        filename = candidates[0]
    return hf_hub_download(repo_id, filename)


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

    @staticmethod
    def _cache_key(name: str, subfolder: str | None) -> str:
        return f"{name}::{subfolder}" if subfolder else name

    def load(self, name: str, subfolder: str | None = None) -> TokenizerAdapter:
        """Load a tokenizer by name, HuggingFace model ID, or file path."""
        key = self._cache_key(name, subfolder)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]

        adapter = self._create_adapter(name, subfolder)
        self._cache[key] = adapter
        if len(self._cache) > self._max_cache_size:
            self._cache.popitem(last=False)
        return adapter

    def _create_adapter(self, name: str, subfolder: str | None = None) -> TokenizerAdapter:
        # 1. Check if it's a tiktoken encoding
        if name in TIKTOKEN_ENCODINGS:
            encoding_name = TIKTOKEN_ENCODINGS[name]
            return TiktokenAdapter(encoding_name)

        # 2. Check if it's a .model file path (SentencePiece)
        if name.endswith(".model") and os.path.exists(name):
            return SentencePieceAdapter(name)

        # 3. Check if it's a TokenMonster vocab (.vocab file or known preset name)
        if name.endswith(".vocab") or name in TOKENMONSTER_VOCABS:
            return TokenMonsterAdapter(name)

        # 4. Check if it's a script_tok tokenizer (on the Hub, or a saved .json/.json.gz path)
        if name.startswith(SCRIPT_TOK_PREFIX):
            return ScriptTokAdapter(
                download_script_tok(name.removeprefix(SCRIPT_TOK_PREFIX)), name=name
            )
        if name.endswith(SCRIPT_TOK_SUFFIXES) and os.path.exists(name):
            return ScriptTokAdapter(name)

        # 5. Try as HuggingFace model ID
        try:
            return HuggingFaceAdapter(name, subfolder=subfolder)
        except Exception as e:
            raise ValueError(
                f"Could not load tokenizer '{name}'. "
                f"Tried tiktoken presets, file path, and HuggingFace. "
                f"HuggingFace error: {e}"
            )

    def reload(self, name: str, subfolder: str | None = None) -> TokenizerAdapter:
        """Evict a tokenizer from cache and reload it fresh."""
        key = self._cache_key(name, subfolder)
        self._cache.pop(key, None)
        return self.load(name, subfolder)

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
        for tm_name in sorted(TOKENMONSTER_VOCABS):
            presets.append({
                "id": tm_name,
                "name": tm_name,
                "tokenizer_type": "unigram",
                "vocab_size": 0,
                "source": "tokenmonster",
            })
        for st_name in SCRIPT_TOK_PRESETS:
            presets.append({
                "id": st_name,
                "name": st_name,
                "tokenizer_type": st_name.rsplit("-", 1)[1],
                "vocab_size": 0,
                "source": "script_tok",
            })
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
