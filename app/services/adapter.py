from abc import ABC, abstractmethod
from functools import lru_cache


@lru_cache(maxsize=1)
def _gpt2_unicode_to_bytes() -> dict[str, int]:
    """Build the inverse of GPT-2's bytes_to_unicode mapping.

    GPT-2 BPE maps each byte value (0-255) to a printable Unicode character
    so vocabulary strings are displayable.  For example byte 0x20 (space)
    maps to Ġ (U+0120).  This function returns the reverse: char -> byte.
    """
    # Ranges that map to themselves (printable Latin-1 chars)
    bs = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("¡"), ord("¬") + 1))
        + list(range(ord("®"), ord("ÿ") + 1))
    )
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


class TokenizerAdapter(ABC):
    """Unified interface for all tokenizer backends."""

    @abstractmethod
    def encode(self, text: str) -> list[int]:
        ...

    @abstractmethod
    def decode(self, ids: list[int]) -> str:
        ...

    @abstractmethod
    def get_vocab(self) -> dict[str, int]:
        """Return mapping of token_str -> token_id."""
        ...

    @abstractmethod
    def get_merges(self) -> list[tuple[str, str]] | None:
        """Return BPE merge rules if available, else None."""
        ...

    def get_merge_ranks(self) -> dict[bytes, int] | None:
        """Return mapping of byte-sequence -> rank for BPE merge tree building.
        Lower rank = earlier merge. Returns None if not available."""
        return None

    @abstractmethod
    def vocab_size(self) -> int:
        ...

    @abstractmethod
    def token_to_bytes(self, token: str) -> bytes:
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        ...

    @property
    @abstractmethod
    def tokenizer_type(self) -> str:
        """Return 'bpe', 'unigram', or 'wordpiece'."""
        ...

    @property
    @abstractmethod
    def source(self) -> str:
        """Return 'tiktoken', 'huggingface', or 'sentencepiece'."""
        ...

    def decode_single(self, token_id: int) -> str:
        """Decode a single token ID to string."""
        return self.decode([token_id])

    def encode_single_token(self, text: str) -> int | None:
        """Encode text expected to be a single token. Returns None if multi-token."""
        ids = self.encode(text)
        return ids[0] if len(ids) == 1 else None


class TiktokenAdapter(TokenizerAdapter):
    def __init__(self, encoding_name: str):
        import tiktoken
        self._encoding = tiktoken.get_encoding(encoding_name)
        self._name = encoding_name
        self._vocab: dict[str, int] | None = None

    def encode(self, text: str) -> list[int]:
        return self._encoding.encode(text, allowed_special="all")

    def decode(self, ids: list[int]) -> str:
        return self._encoding.decode(ids)

    def get_vocab(self) -> dict[str, int]:
        if self._vocab is None:
            self._vocab = {}
            for token_bytes, token_id in self._encoding._mergeable_ranks.items():
                try:
                    token_str = token_bytes.decode("utf-8", errors="replace")
                except Exception:
                    token_str = repr(token_bytes)
                self._vocab[token_str] = token_id
            if hasattr(self._encoding, "_special_tokens"):
                for token_str, token_id in self._encoding._special_tokens.items():
                    self._vocab[token_str] = token_id
        return self._vocab

    def get_merges(self) -> list[tuple[str, str]] | None:
        # tiktoken stores merges implicitly via rank ordering of _mergeable_ranks
        # We can reconstruct merge order from the ranks
        ranks = self._encoding._mergeable_ranks
        sorted_tokens = sorted(ranks.items(), key=lambda x: x[1])
        # Base tokens (single bytes) are ranks 0-255
        # Everything after is a merge result
        merges = []
        for token_bytes, _rank in sorted_tokens:
            if len(token_bytes) <= 1:
                continue
            # Find the split point that uses the highest-ranked pair
            best_split = None
            best_max_rank = -1
            for i in range(1, len(token_bytes)):
                left = token_bytes[:i]
                right = token_bytes[i:]
                if left in ranks and right in ranks:
                    max_rank = max(ranks[left], ranks[right])
                    if best_split is None or max_rank < best_max_rank:
                        best_max_rank = max_rank
                        best_split = (left, right)
            if best_split:
                left_str = best_split[0].decode("utf-8", errors="replace")
                right_str = best_split[1].decode("utf-8", errors="replace")
                merges.append((left_str, right_str))
        return merges

    def get_merge_ranks(self) -> dict[bytes, int] | None:
        return dict(self._encoding._mergeable_ranks)

    def vocab_size(self) -> int:
        return self._encoding.n_vocab

    def token_to_bytes(self, token: str) -> bytes:
        return token.encode("utf-8")

    @property
    def name(self) -> str:
        return self._name

    @property
    def tokenizer_type(self) -> str:
        return "bpe"

    @property
    def source(self) -> str:
        return "tiktoken"


class HuggingFaceAdapter(TokenizerAdapter):
    def __init__(self, model_name: str):
        from transformers import AutoTokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self._model_name = model_name
        self._vocab: dict[str, int] | None = None
        # Determine type
        self._type = "bpe"
        if hasattr(self._tokenizer, "backend_tokenizer"):
            model = self._tokenizer.backend_tokenizer.model
            model_type = type(model).__name__.lower()
            if "unigram" in model_type:
                self._type = "unigram"
            elif "wordpiece" in model_type:
                self._type = "wordpiece"
        print("hello")
    def encode(self, text: str) -> list[int]:
        return self._tokenizer.encode(text, add_special_tokens=False)

    def decode(self, ids: list[int]) -> str:
        return self._tokenizer.decode(ids)

    def get_vocab(self) -> dict[str, int]:
        if self._vocab is None:
            self._vocab = dict(self._tokenizer.get_vocab())
        return self._vocab

    def get_merges(self) -> list[tuple[str, str]] | None:
        if self._type != "bpe":
            return None
        try:
            if hasattr(self._tokenizer, "backend_tokenizer"):
                model = self._tokenizer.backend_tokenizer.model
                # tokenizers library exposes merges in the model
                if hasattr(model, "merges"):
                    return [(m[0], m[1]) for m in model.merges]
        except Exception:
            pass
        return None

    def get_merge_ranks(self) -> dict[bytes, int] | None:
        if self._type != "bpe":
            return None

        # Build the GPT-2 unicode-to-byte mapping.
        # GPT-2 BPE maps each byte to a printable Unicode char so that
        # vocab strings are displayable.  E.g. byte 0x20 (space) -> Ġ (U+0120).
        unicode_to_byte = _gpt2_unicode_to_bytes()

        vocab = self._tokenizer.get_vocab()
        ranks: dict[bytes, int] = {}
        for token_str, token_id in vocab.items():
            if token_str.startswith("<") and token_str.endswith(">"):
                continue
            try:
                # Try GPT-2 byte decoding first (each char maps to one byte)
                if all(ch in unicode_to_byte for ch in token_str):
                    token_bytes = bytes(unicode_to_byte[ch] for ch in token_str)
                else:
                    token_bytes = token_str.encode("utf-8")
                ranks[token_bytes] = token_id
            except Exception:
                pass
        return ranks if ranks else None

    def vocab_size(self) -> int:
        return self._tokenizer.vocab_size

    def token_to_bytes(self, token: str) -> bytes:
        return token.encode("utf-8")

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def tokenizer_type(self) -> str:
        return self._type

    @property
    def source(self) -> str:
        return "huggingface"


class SentencePieceAdapter(TokenizerAdapter):
    def __init__(self, model_path: str):
        import sentencepiece as spm
        self._sp = spm.SentencePieceProcessor()
        self._sp.Load(model_path)
        self._model_path = model_path
        self._vocab: dict[str, int] | None = None
        # Determine type from model type
        # SentencePiece model types: 1=unigram, 2=bpe
        model_type = self._sp.piece_to_id("▁") >= 0  # just check if it loaded
        self._type = "bpe"  # default, could be unigram
        try:
            import sentencepiece.sentencepiece_model_pb2 as sp_model
            with open(model_path, "rb") as f:
                m = sp_model.ModelProto()
                m.ParseFromString(f.read())
                if m.trainer_spec.model_type == 1:
                    self._type = "unigram"
                elif m.trainer_spec.model_type == 2:
                    self._type = "bpe"
        except Exception:
            pass

    def encode(self, text: str) -> list[int]:
        return self._sp.Encode(text)

    def decode(self, ids: list[int]) -> str:
        return self._sp.Decode(ids)

    def get_vocab(self) -> dict[str, int]:
        if self._vocab is None:
            self._vocab = {}
            for i in range(self._sp.GetPieceSize()):
                self._vocab[self._sp.IdToPiece(i)] = i
        return self._vocab

    def get_merges(self) -> list[tuple[str, str]] | None:
        if self._type != "bpe":
            return None
        try:
            import sentencepiece.sentencepiece_model_pb2 as sp_model
            with open(self._model_path, "rb") as f:
                m = sp_model.ModelProto()
                m.ParseFromString(f.read())
                merges = []
                for piece in m.pieces:
                    if piece.type == 1 and len(piece.piece) > 1:
                        # Try to find best split
                        p = piece.piece
                        for i in range(1, len(p)):
                            left, right = p[:i], p[i:]
                            if left in self._vocab and right in self._vocab:
                                merges.append((left, right))
                                break
                return merges if merges else None
        except Exception:
            return None

    def vocab_size(self) -> int:
        return self._sp.GetPieceSize()

    def token_to_bytes(self, token: str) -> bytes:
        return token.encode("utf-8")

    @property
    def name(self) -> str:
        return self._model_path.split("/")[-1].replace(".model", "")

    @property
    def tokenizer_type(self) -> str:
        return self._type

    @property
    def source(self) -> str:
        return "sentencepiece"
