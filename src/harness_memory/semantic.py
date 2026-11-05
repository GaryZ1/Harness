"""Semantic memory: durable facts, preferences, decisions and references.

Function axis: *factual* memory. Loading policy: one line per memory in the always-loaded
``MEMORY.md`` index; topic files loaded on demand.

Types handled here: ``user`` (role, expertise, preferences), ``feedback`` (corrections and
confirmed approaches), ``project`` (decisions and deadlines **not derivable** from code/git) and
``reference`` (where to find external information). Experience types (``lesson``/``workflow``)
are also accepted because they share the same file format, but they require provenance.

Write semantics follow the 2026 evidence (Mem0 v3 ADD-only, Graphiti invalidate-not-delete):

* :meth:`SemanticMemory.remember` only ever **adds** a file;
* :meth:`SemanticMemory.supersede` adds the new version, marks the old one ``superseded`` with a
  ``valid_to`` timestamp and a pointer, and never deletes it in the same write;
* "current truth" is resolved at read time (:meth:`current`, :meth:`search` with ``as_of``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Iterable

from ._time import Clock, utcnow
from .errors import AlreadyExistsError, NotFoundError, ValidationError
from .frontmatter import (
    Epistemic,
    MemoryDocument,
    MemoryMeta,
    MemoryType,
    Provenance,
    Scope,
    Status,
)
from .index import IndexEntry, IndexStatus, MemoryIndex
from .retrieval import HybridRetriever, IndexedItem, SearchHit, tokenize
from .store import ABSENT, FileMemoryStore

SEMANTIC_TYPES = frozenset({MemoryType.USER, MemoryType.FEEDBACK, MemoryType.PROJECT, MemoryType.REFERENCE})
_SLUG = re.compile(r"[^a-z0-9]+")
_VERSION_SUFFIX = re.compile(r"^(?P<stem>.+?)(?:--v(?P<n>\d+))?\.md$")

#: Heuristics for content that is derivable from the repository and therefore should not be stored.
_DERIVABLE_PATTERNS = (
    re.compile(r"^\s*(?:[\w./-]+/)+[\w.-]+\s*$"),  # a bare file path
    re.compile(r"^\s*[0-9a-f]{7,40}\s*$"),  # a bare commit sha
    re.compile(r"^\s*(?:def|class|function|import)\s", re.M),  # pasted code
)


def slugify(text: str, max_len: int = 48) -> str:
    slug = _SLUG.sub("-", text.lower()).strip("-")
    return (slug[:max_len].rstrip("-")) or "memory"


def looks_derivable(body: str) -> bool:
    """Cheap guard against storing code/paths/shas that can be re-derived from the repo."""
    stripped = body.strip()
    return any(p.search(stripped) for p in _DERIVABLE_PATTERNS) and len(stripped.splitlines()) <= 3


@dataclass(frozen=True)
class WriteResult:
    path: str
    index_status: IndexStatus
    superseded: str | None = None

    @property
    def warning(self) -> str | None:
        return self.index_status.message


class SemanticMemory:
    """Typed topic files + bounded index + hybrid retrieval over one writable (or read-only) store."""

    def __init__(
        self,
        store: FileMemoryStore,
        *,
        index: MemoryIndex | None = None,
        retriever: HybridRetriever | None = None,
        clock: Clock | None = None,
        reject_if: Callable[[str], bool] | None = looks_derivable,
    ) -> None:
        self.store = store
        self.index = index or MemoryIndex(store)
        self.retriever = retriever or HybridRetriever()
        self.clock = clock or store.clock or utcnow
        self.reject_if = reject_if
        self._indexed = False

    # ------------------------------------------------------------------ writes
    def remember(
        self,
        type: MemoryType | str,
        description: str,
        body: str,
        *,
        slug: str | None = None,
        scope: Scope | None = None,
        sessions: Iterable[str] = (),
        evidence: Iterable[str] = (),
        epistemic: Epistemic = Epistemic.CLAIMED,
        entities: Iterable[str] = (),
    ) -> WriteResult:
        """ADD a new memory file ``<type>_<slug>.md`` and its index line.

        Raises :class:`AlreadyExistsError` if an *active* memory with the same slug exists - call
        :meth:`supersede` to change it.
        """
        mtype = MemoryType(type)
        if mtype is MemoryType.SKILL:
            raise ValidationError("skills belong in ProceduralMemory / SkillLibrary")
        description = " ".join(description.split())
        if not description:
            raise ValidationError("description is required (it becomes the index line)")
        if self.reject_if and self.reject_if(body):
            raise ValidationError(
                "this looks derivable from the codebase or git (path, sha or code); store the non-derivable insight instead"
            )
        name = f"{mtype.value}_{slugify(slug or description)}.md"
        if self.store.exists(name):
            existing = self.store.read_document(name)
            if existing.meta.status is Status.ACTIVE:
                raise AlreadyExistsError(f"{name} already exists; use supersede() to change it")
            name = self._next_version_name(name)
        meta = MemoryMeta(
            type=mtype,
            scope=scope or self.store.scope,
            description=description,
            source=Provenance(list(sessions), list(evidence), epistemic),
        )
        if entities:
            meta.extra["entities"] = sorted({e for e in entities})
        doc = MemoryDocument(meta, _ensure_title(body, description), path=name)
        with self.store.lock():
            self.store.write_document(doc, expected_sha256=ABSENT)
            status = self.index.upsert(IndexEntry(description, name, mtype.value))
        self._index_doc(doc)
        return WriteResult(name, status)

    def supersede(
        self,
        old_path: str,
        body: str,
        *,
        description: str | None = None,
        reason: str = "",
        sessions: Iterable[str] = (),
        evidence: Iterable[str] = (),
        epistemic: Epistemic = Epistemic.CLAIMED,
    ) -> WriteResult:
        """Replace ``old_path`` with a new version, keeping the old file as ``superseded`` history."""
        with self.store.lock():
            old_file = self.store.read(old_path)
            old = MemoryDocument.from_markdown(old_file.content, path=old_file.path)
            if old.meta.status is Status.SUPERSEDED:
                raise ValidationError(f"{old_path} is already superseded by {old.meta.superseded_by}")
            if self.reject_if and self.reject_if(body):
                raise ValidationError("new content looks derivable from the codebase or git")
            now = self.clock()
            new_path = self._next_version_name(old_path)
            desc = " ".join((description or old.meta.description).split())
            meta = MemoryMeta(
                type=old.meta.type,
                scope=old.meta.scope,
                description=desc,
                supersedes=old_path,
                source=Provenance(
                    list(dict.fromkeys([*old.meta.source.sessions, *sessions])),
                    list(dict.fromkeys([*evidence])) or list(old.meta.source.evidence),
                    epistemic,
                ),
                use_count=old.meta.use_count,
                last_used=old.meta.last_used,
            )
            meta.extra = {k: v for k, v in old.meta.extra.items() if k != "conflict_note"}
            new_doc = MemoryDocument(meta, _ensure_title(body, desc), path=new_path)
            self.store.write_document(new_doc, expected_sha256=ABSENT)
            old.meta.status = Status.SUPERSEDED
            old.meta.superseded_by = new_path
            old.meta.valid_to = now
            if reason:
                old.meta.extra["conflict_note"] = reason
            self.store.write_document(old, expected_sha256=old_file.sha256)
            self.index.remove(old_path)
            status = self.index.upsert(IndexEntry(desc, new_path, meta.type.value))
        self._index_doc(old)
        self._index_doc(new_doc)
        return WriteResult(new_path, status, superseded=old_path)

    def mark_stale(self, path: str, note: str) -> None:
        """Flag a memory whose evidence has drifted; it stays readable but drops out of default search."""
        with self.store.lock():
            f = self.store.read(path)
            doc = MemoryDocument.from_markdown(f.content, path=f.path)
            doc.meta.status = Status.STALE
            doc.meta.extra["stale_note"] = note
            self.store.write_document(doc, expected_sha256=f.sha256)
        self._index_doc(doc)

    def forget(self, path: str) -> None:
        """Explicit, user-requested deletion (file + index line). Git history keeps an audit trail."""
        with self.store.lock():
            if not self.store.exists(path):
                raise NotFoundError(path)
            self.store.delete(path)
            self.index.remove(path)
        self.retriever.remove(path)

    # ------------------------------------------------------------------ reads
    def get(self, path: str) -> MemoryDocument:
        return self.store.read_document(path)

    def all(self, *, include_inactive: bool = False) -> list[MemoryDocument]:
        docs = []
        for doc in self.store.iter_documents():
            if "/" in doc.path:  # topic files live at the root; subdirs are other memory types
                continue
            if doc.meta.type is MemoryType.SKILL:
                continue
            if include_inactive or doc.meta.status is Status.ACTIVE:
                docs.append(doc)
        return docs

    def current(self, path: str) -> MemoryDocument:
        """Follow ``superseded_by`` links to the live version of ``path``."""
        seen = set()
        doc = self.get(path)
        while doc.meta.superseded_by and doc.meta.superseded_by not in seen:
            seen.add(doc.path)
            doc = self.get(doc.meta.superseded_by)
        return doc

    def history(self, path: str) -> list[MemoryDocument]:
        """Oldest-to-newest chain of versions that ``path`` belongs to."""
        doc = self.current(path)
        chain = [doc]
        while doc.meta.supersedes:
            doc = self.get(doc.meta.supersedes)
            chain.append(doc)
        return list(reversed(chain))

    def search(
        self,
        query: str,
        *,
        top_k: int = 20,
        threshold: float = 0.1,
        types: Iterable[MemoryType | str] | None = None,
        as_of: datetime | None = None,
        include_stale: bool = False,
        include_tainted: bool = False,
    ) -> list[SearchHit]:
        """Hybrid search. By default only active, untainted memories are returned.

        With ``as_of`` the search runs over *all* versions valid at that instant (point-in-time recall).
        """
        self._ensure_indexed()
        wanted = {MemoryType(t) for t in types} if types else None

        def keep(item: IndexedItem) -> bool:
            meta: MemoryMeta = item.metadata["meta"]
            if wanted and meta.type not in wanted:
                return False
            if meta.tainted and not include_tainted:
                return False
            if as_of is not None:
                return meta.status is not Status.STALE or include_stale
            if meta.status is Status.SUPERSEDED:
                return False
            if meta.status is Status.STALE and not include_stale:
                return False
            return True

        return self.retriever.search(query, top_k=top_k, threshold=threshold, as_of=as_of, filter_fn=keep)

    def reindex(self) -> None:
        self.retriever.clear()
        self._indexed = True
        for doc in self.all(include_inactive=True):
            self._index_doc(doc)

    # ------------------------------------------------------------------ internals
    def _ensure_indexed(self) -> None:
        if not self._indexed:
            self.reindex()

    def _index_doc(self, doc: MemoryDocument) -> None:
        if not self._indexed and len(self.retriever) == 0:
            # Lazy: the first search will build the whole index.
            return
        entities = frozenset(doc.meta.extra.get("entities") or []) | frozenset(t for t in tokenize(doc.meta.description) if len(t) > 3)
        self.retriever.upsert(
            [
                IndexedItem(
                    doc_id=doc.path,
                    text=f"{doc.meta.description}\n{doc.body}",
                    timestamp=doc.meta.created,
                    valid_to=doc.meta.valid_to,
                    entities=entities,
                    metadata={"meta": doc.meta},
                )
            ]
        )

    def _next_version_name(self, path: str) -> str:
        m = _VERSION_SUFFIX.match(path)
        if not m:
            raise ValidationError(f"unexpected memory filename: {path}")
        stem = m.group("stem")
        n = int(m.group("n") or 1) + 1
        while self.store.exists(f"{stem}--v{n}.md"):
            n += 1
        return f"{stem}--v{n}.md"


def _ensure_title(body: str, description: str) -> str:
    body = body.strip("\n")
    if body.lstrip().startswith("#"):
        return body + "\n"
    return f"# {description}\n\n{body}\n"
