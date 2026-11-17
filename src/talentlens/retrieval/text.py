"""Tokenisation for BM25 and word-window chunking for dense embeddings."""

from __future__ import annotations

import re

# Keeps tech tokens intact: "c++", "c#", ".net", "node.js", "a/r".
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9+#./&-]*[a-z0-9+#]|[a-z0-9]")

# Small, conservative stop-word list. A big list would also remove words like
# "it" and "us", which can matter in job text ("IT support").
STOPWORDS = frozenset(
    "a an and are as at be been by for from has have in is of on or that the this to was were will with "
    "you your we our they their he she his her its into than then there these those such can may".split()
)


def tokenize(text: str) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    return [t.strip(".") for t in tokens if t not in STOPWORDS]


def chunk_words(text: str, size: int, overlap: int) -> list[str]:
    """Split into overlapping windows of `size` words. Short texts give one chunk."""
    if size <= overlap:
        raise ValueError("chunk size must be larger than overlap")
    words = text.split()
    if len(words) <= size:
        return [" ".join(words)] if words else [""]
    step = size - overlap
    chunks = []
    for start in range(0, len(words), step):
        chunks.append(" ".join(words[start : start + size]))
        if start + size >= len(words):
            break
    return chunks
