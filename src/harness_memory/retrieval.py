"""Hybrid retrieval sidecar: BM25 + optional embeddings + entity match, fused with reciprocal-rank fusion.

This is the optional "phase 2 of the build" index from the checklist (step 16). It sits *behind*
the file abstraction: documents are always files; this module only ranks them. Defaults follow
Mem0 v3 (``top_k=20``, ``threshold=0.1``, rerank off). "Current truth" is resolved at read time
from timestamps and validity windows, never by destructive overwrite.
"""

from __future__ import annotations

import math
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable, Iterable, Protocol, Sequence

_TOKEN = re.compile(r"[A-Za-z0-9_]+(?:[.\-][A-Za-z0-9_]+)*")
_STOPWORDS = frozenset(
    "a an and are as at be but by for from has have i in is it its of on or that the this to was were will with you your we our".split()
)
_RECENCY_HINT = re.compile(r"\b(latest|current|currently|now|today|recent|recently|newest|still)\b", re.I)


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for tok in _TOKEN.findall(text.lower()):
        if tok in _STOPWORDS:
            continue
        tokens.append(tok)
        if "." in tok or "-" in tok:  # also index the parts of dotted/hyphenated identifiers
            tokens.extend(p for p in re.split(r"[.\-]", tok) if p and p not in _STOPWORDS)
    return tokens


class Embedder(Protocol):
    def __call__(self, texts: Sequence[str]) -> list[list[float]]: ...


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


class BM25Index:
    """Incremental Okapi BM25 over in-memory postings. Thread-safe."""

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self._docs: dict[str, Counter[str]] = {}
        self._lengths: dict[str, int] = {}
        self._df: Counter[str] = Counter()
        self._lock = threading.RLock()

    def __len__(self) -> int:
        return len(self._docs)

    def add(self, doc_id: str, text: str) -> None:
        with self._lock:
            self.remove(doc_id)
            tf = Counter(tokenize(text))
            self._docs[doc_id] = tf
            self._lengths[doc_id] = sum(tf.values())
            self._df.update(tf.keys())

    def remove(self, doc_id: str) -> None:
        with self._lock:
            tf = self._docs.pop(doc_id, None)
            if tf is None:
                return
            self._lengths.pop(doc_id, None)
            for term in tf:
                self._df[term] -= 1
                if self._df[term] <= 0:
                    del self._df[term]

    def search(self, query: str, k: int = 20) -> list[tuple[str, float]]:
        terms = tokenize(query)
        with self._lock:
            n = len(self._docs)
            if not n or not terms:
                return []
            avgdl = sum(self._lengths.values()) / n or 1.0
            scores: dict[str, float] = {}
            for term in set(terms):
                df = self._df.get(term, 0)
                if not df:
                    continue
                idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
                for doc_id, tf in self._docs.items():
                    f = tf.get(term, 0)
                    if not f:
                        continue
                    denom = f + self.k1 * (1 - self.b + self.b * self._lengths[doc_id] / avgdl)
                    scores[doc_id] = scores.get(doc_id, 0.0) + idf * f * (self.k1 + 1) / denom
        return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]


@dataclass
class IndexedItem:
    doc_id: str
    text: str
    timestamp: datetime | None = None
    valid_to: datetime | None = None
    entities: frozenset[str] = frozenset()
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SearchHit:
    doc_id: str
    score: float
    signals: dict[str, float]
    item: IndexedItem


class HybridRetriever:
    """Fuse lexical, semantic and entity signals; resolve time at query time.

    :param embedder: optional batch embedding function. Without it, retrieval is BM25 + entity only.
    :param rrf_k: reciprocal-rank-fusion constant (60 is the standard value).
    """

    def __init__(self, *, embedder: Embedder | None = None, rrf_k: int = 60) -> None:
        self.embedder = embedder
        self.rrf_k = rrf_k
        self._bm25 = BM25Index()
        self._items: dict[str, IndexedItem] = {}
        self._vectors: dict[str, list[float]] = {}
        self._lock = threading.RLock()

    def __len__(self) -> int:
        return len(self._items)

    def upsert(self, items: Iterable[IndexedItem]) -> None:
        batch = list(items)
        vectors = self.embedder([i.text for i in batch]) if (self.embedder and batch) else [None] * len(batch)
        with self._lock:
            for item, vec in zip(batch, vectors, strict=True):
                self._items[item.doc_id] = item
                self._bm25.add(item.doc_id, item.text)
                if vec is not None:
                    self._vectors[item.doc_id] = list(vec)

    def remove(self, doc_id: str) -> None:
        with self._lock:
            self._items.pop(doc_id, None)
            self._vectors.pop(doc_id, None)
            self._bm25.remove(doc_id)

    def clear(self) -> None:
        with self._lock:
            self._items.clear()
            self._vectors.clear()
            self._bm25 = BM25Index()

    def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        threshold: float = 0.1,
        as_of: datetime | None = None,
        filter_fn: Callable[[IndexedItem], bool] | None = None,
    ) -> list[SearchHit]:
        """Return up to ``top_k`` hits whose normalised fused score is >= ``threshold``.

        ``as_of`` restricts results to items valid at that instant (``timestamp <= as_of < valid_to``).
        Queries containing recency cues ("latest", "current", "now", ...) get a recency boost
        (time-aware query expansion, LongMemEval).
        """
        with self._lock:
            candidates = {d: it for d, it in self._items.items() if (filter_fn is None or filter_fn(it)) and _valid_at(it, as_of)}
            if not candidates:
                return []
            pool = max(top_k * 5, 50)
            rankings: dict[str, list[str]] = {}
            lexical = [(d, s) for d, s in self._bm25.search(query, k=len(self._items)) if d in candidates][:pool]
            rankings["bm25"] = [d for d, _ in lexical]
            if self.embedder and self._vectors:
                qv = self.embedder([query])[0]
                sem = sorted(
                    ((d, cosine(qv, self._vectors[d])) for d in candidates if d in self._vectors),
                    key=lambda kv: -kv[1],
                )
                rankings["semantic"] = [d for d, s in sem[:pool] if s > 0]
            q_tokens = set(tokenize(query))
            ent = sorted(
                ((d, len(q_tokens & {e.lower() for e in it.entities})) for d, it in candidates.items() if it.entities),
                key=lambda kv: -kv[1],
            )
            rankings["entity"] = [d for d, s in ent if s > 0][:pool]
            if _RECENCY_HINT.search(query):
                rec = sorted(
                    (d for d in candidates if candidates[d].timestamp),
                    key=lambda d: candidates[d].timestamp,  # type: ignore[arg-type,return-value]
                    reverse=True,
                )
                # Only boost documents that matched something else.
                matched = set().union(*map(set, rankings.values()))
                rankings["recency"] = [d for d in rec if d in matched][:pool]

            fused: dict[str, float] = {}
            signals: dict[str, dict[str, float]] = {}
            for name, ranked in rankings.items():
                for rank, d in enumerate(ranked):
                    contrib = 1.0 / (self.rrf_k + rank + 1)
                    fused[d] = fused.get(d, 0.0) + contrib
                    signals.setdefault(d, {})[name] = contrib
            if not fused:
                return []
            best = max(fused.values())
            hits = [SearchHit(d, s / best, signals[d], candidates[d]) for d, s in fused.items() if s / best >= threshold]
            hits.sort(key=lambda h: (-h.score, -(h.item.timestamp.timestamp() if h.item.timestamp else 0.0), h.doc_id))
            return hits[:top_k]


def _valid_at(item: IndexedItem, as_of: datetime | None) -> bool:
    if as_of is None:
        return True
    if item.timestamp and item.timestamp > as_of:
        return False
    if item.valid_to and item.valid_to <= as_of:
        return False
    return True


def grep(lines: Iterable[tuple[str, str]], pattern: str, *, ignore_case: bool = True, max_hits: int = 50) -> list[tuple[str, int, str]]:
    """Agentic-search fallback: regex over ``(path, text)`` pairs -> ``(path, line_no, line)``."""
    rx = re.compile(pattern, re.I if ignore_case else 0)
    hits = []
    for path, text in lines:
        for i, line in enumerate(text.splitlines(), start=1):
            if rx.search(line):
                hits.append((path, i, line))
                if len(hits) >= max_hits:
                    return hits
    return hits
