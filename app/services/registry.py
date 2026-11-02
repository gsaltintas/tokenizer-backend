import logging
import os
import posixpath
import threading
import time
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


def script_tok_files(repo_id: str, folder: str) -> list[str]:
    """The .json/.json.gz files directly in `folder` of a Hub repo."""
    from huggingface_hub import list_repo_files

    folder = folder.strip("/")
    return [
        f for f in list_repo_files(repo_id)
        if posixpath.dirname(f) == folder and f.endswith(SCRIPT_TOK_SUFFIXES)
    ]


def download_script_tok(spec: str, subfolder: str | None = None) -> str:
    """Download "<repo>" or "<repo>/<file>" from the Hub; returns the local path. With only
    a repo, it must hold exactly one .json/.json.gz file in `subfolder` (default tokenizer/)."""
    from huggingface_hub import hf_hub_download

    owner, repo, *rest = spec.split("/", 2)
    repo_id = f"{owner}/{repo}"
    folder = (subfolder or "tokenizer").strip("/")
    if rest:
        filename = posixpath.join(subfolder.strip("/"), rest[0]) if subfolder else rest[0]
    else:
        candidates = script_tok_files(repo_id, folder)
        if len(candidates) != 1:
            raise ValueError(
                f"Expected one script_tok tokenizer in {folder}/ of {repo_id}, "
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


logger = logging.getLogger(__name__)


class TokenizerRegistry:
    """Loads and caches tokenizer adapters with an LRU policy.

    The cache is shared by every visitor. Evicting a tokenizer is always safe: `get`
    loads it again on the next request that names it.
    """

    def __init__(self, max_cache_size: int = 10):
        self._cache: OrderedDict[str, TokenizerAdapter] = OrderedDict()
        self._last_used: dict[str, float] = {}
        self._max_cache_size = max_cache_size
        # sync endpoints run in a thread pool, next to the idle-eviction task
        self._lock = threading.RLock()

    @staticmethod
    def _cache_key(name: str, subfolder: str | None) -> str:
        return f"{name}::{subfolder}" if subfolder else name

    @staticmethod
    def _split_key(key: str) -> tuple[str, str | None]:
        name, sep, subfolder = key.partition("::")
        return name, (subfolder if sep else None)

    def _touch(self, key: str) -> TokenizerAdapter:
        self._cache.move_to_end(key)
        self._last_used[key] = time.monotonic()
        return self._cache[key]

    def load(self, name: str, subfolder: str | None = None) -> TokenizerAdapter:
        """Load a tokenizer by name, HuggingFace model ID, or file path."""
        key = self._cache_key(name, subfolder)
        with self._lock:
            if key in self._cache:
                return self._touch(key)

        # Outside the lock: loading can take a while (Hub downloads)
        adapter = self._create_adapter(name, subfolder)
        with self._lock:
            self._cache[key] = adapter
            self._touch(key)
            while len(self._cache) > self._max_cache_size:
                evicted, _ = self._cache.popitem(last=False)
                self._last_used.pop(evicted, None)
        return adapter

    def unload(self, key: str | None = None) -> list[str]:
        """Drop one tokenizer (by id), or all of them; returns the ids dropped."""
        with self._lock:
            keys = list(self._cache) if key is None else [key] if key in self._cache else []
            for k in keys:
                self._cache.pop(k)
                self._last_used.pop(k, None)
        return keys

    def evict_idle(self, max_idle_s: float) -> list[str]:
        """Drop tokenizers that no request has used for `max_idle_s` seconds."""
        cutoff = time.monotonic() - max_idle_s
        with self._lock:
            idle = [k for k in self._cache if self._last_used.get(k, 0) < cutoff]
            for k in idle:
                self._cache.pop(k)
                self._last_used.pop(k, None)
        return idle

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
        display_name = f"{name}/{subfolder.strip('/')}" if subfolder else name
        if name.startswith(SCRIPT_TOK_PREFIX):
            return ScriptTokAdapter(
                download_script_tok(name.removeprefix(SCRIPT_TOK_PREFIX), subfolder),
                name=display_name,
            )
        if name.endswith(SCRIPT_TOK_SUFFIXES) and os.path.exists(name):
            return ScriptTokAdapter(name)

        # 5. Try as HuggingFace model ID
        try:
            return HuggingFaceAdapter(name, subfolder=subfolder)
        except Exception as e:
            hf_error = e

        # 6. A Hub repo (or subfolder) holding a script_tok tokenizer.json.gz, which
        # AutoTokenizer can't read
        try:
            found = script_tok_files(name, subfolder or "") if name.count("/") == 1 else []
        except Exception:
            found = []
        if len(found) == 1:
            return ScriptTokAdapter(download_script_tok(f"{name}/{found[0]}"), name=display_name)

        raise ValueError(
            f"Could not load tokenizer '{name}'. "
            f"Tried tiktoken presets, file path, HuggingFace and script_tok. "
            f"HuggingFace error: {hf_error}"
        )

    def reload(self, name: str, subfolder: str | None = None) -> TokenizerAdapter:
        """Evict a tokenizer from cache and reload it fresh."""
        self.unload(self._cache_key(name, subfolder))
        return self.load(name, subfolder)

    def get(self, name: str) -> TokenizerAdapter | None:
        """Get a tokenizer by id, loading it again if it was evicted; None if it can't load."""
        with self._lock:
            if name in self._cache:
                return self._touch(name)
        try:
            return self.load(*self._split_key(name))
        except Exception as e:
            logger.warning("Could not reload evicted tokenizer %r: %s", name, e)
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
            for name, adapter in list(self._cache.items())
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
