"""Tokenizers.

``load_tokenizer`` returns the real Hugging Face tokenizer when it can be
loaded (a local path, the HF cache, or Tinker's ``get_tokenizer()``). For
offline development and unit tests, :class:`ApproxQwenTokenizer` mimics Qwen's
pre-tokenisation (every digit is its own token, words carry their leading
space, special chat tokens are atomic) and errs towards more tokens than the
real tokenizer for unusual strings. Budget and parity claims must be re-run
with the real tokenizer at build stage 3 (``iuw render-check --tokenizer``).
"""

from __future__ import annotations

import hashlib
import re
from typing import Optional, Protocol, Sequence

try:  # the `regex` module supports \p{L}; fall back to an ASCII approximation
    import regex as _re

    _PRETOKEN = _re.compile(
        r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
    )
except ImportError:  # pragma: no cover
    _re = re
    _PRETOKEN = re.compile(
        r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\w]?[^\W\d_]+|\d| ?[^\s\w]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
    )

SPECIAL_TOKENS = {
    "<|endoftext|>": 151643,
    "<|im_start|>": 151644,
    "<|im_end|>": 151645,
    "<think>": 151667,
    "</think>": 151668,
}
_SPECIAL_RE = re.compile("(" + "|".join(re.escape(s) for s in SPECIAL_TOKENS) + ")")


class Tokenizer(Protocol):
    name: str

    def encode(self, text: str) -> list[int]: ...

    def decode(self, ids: Sequence[int]) -> str: ...


class ApproxQwenTokenizer:
    """Deterministic offline stand-in for the Qwen tokenizer (not for final claims)."""

    name = "approx-qwen"
    WORD_MAX = 7  # letter runs up to this length count as one token
    CHUNK = 4

    def __init__(self) -> None:
        self._id_to_str: dict[int, str] = {v: k for k, v in SPECIAL_TOKENS.items()}
        for d in range(10):
            self._id_to_str[15 + d] = str(d)

    def _piece_id(self, piece: str) -> int:
        if len(piece) == 1 and piece.isdigit():
            return 15 + int(piece)
        h = int.from_bytes(hashlib.blake2b(piece.encode(), digest_size=6).digest(), "little")
        i = 1000 + h % 150_000
        while i in self._id_to_str and self._id_to_str[i] != piece:
            i = 1000 + (i + 1 - 1000) % 150_000
        self._id_to_str[i] = piece
        return i

    def _split(self, piece: str) -> list[str]:
        letters = sum(c.isalpha() for c in piece)
        if piece.isspace() or len(piece) <= 1:
            return [piece]
        if letters == len(piece.lstrip(" ")) and letters <= self.WORD_MAX:
            return [piece]
        if letters and letters >= len(piece) - 1 and letters <= self.WORD_MAX:
            return [piece]
        size = self.CHUNK if letters else 2
        return [piece[i:i + size] for i in range(0, len(piece), size)]

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for part in _SPECIAL_RE.split(text):
            if not part:
                continue
            if part in SPECIAL_TOKENS:
                ids.append(SPECIAL_TOKENS[part])
                continue
            for m in _PRETOKEN.finditer(part):
                for chunk in self._split(m.group(0)):
                    ids.append(self._piece_id(chunk))
        return ids

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(self._id_to_str.get(int(i), "�") for i in ids)


class HFTokenizer:
    """Wrapper around a transformers tokenizer (special tokens parsed, none added)."""

    def __init__(self, tok, name: str):
        self.tok = tok
        self.name = name

    def encode(self, text: str) -> list[int]:
        return list(self.tok.encode(text, add_special_tokens=False))

    def decode(self, ids: Sequence[int]) -> str:
        return self.tok.decode(list(ids), skip_special_tokens=False)


def load_tokenizer(name: Optional[str] = None) -> Tokenizer:
    """Load a tokenizer by HF name or path; ``None`` or "approx" gives the offline stand-in."""
    if name in (None, "", "approx", "approx-qwen"):
        return ApproxQwenTokenizer()
    from transformers import AutoTokenizer  # optional dependency

    return HFTokenizer(AutoTokenizer.from_pretrained(name), name)


def action_token_ids(tokenizer: Tokenizer, n_actions: int = 6) -> list[int]:
    """Verify that action IDs 0..n-1 are single, distinct tokens; return their token IDs."""
    ids = []
    for a in range(n_actions):
        enc = tokenizer.encode(str(a))
        if len(enc) != 1:
            raise ValueError(f"action {a} is not a single token under {tokenizer.name}: {enc}")
        ids.append(enc[0])
    if len(set(ids)) != n_actions:
        raise ValueError("action tokens are not distinct")
    return ids


def count_tokens(tokenizer: Tokenizer, text: str) -> int:
    return len(tokenizer.encode(text))
