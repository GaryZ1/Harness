"""Procedural memory: skills, workflows and lessons - *how* to do things.

Two complementary structures:

* :class:`SkillLibrary` - Agent Skills directories (``skills/<name>/SKILL.md`` + resources) with
  **three-level progressive disclosure**: (1) name + description in the system prompt, (2) the
  SKILL.md body when relevant, (3) bundled files only as needed. Self-generated skills start as
  ``status: stale`` *candidates* and pass a **promotion gate** (verified evidence, human approval,
  or successful reuse in >= 2 distinct sessions) because unverified self-authored skills show ~0
  or negative benefit (SkillsBench, EvoClawBench).
* :class:`Playbook` - an ACE-style itemised playbook: bullets with ``helpful``/``harmful`` counters
  updated by **incremental delta operations** with deterministic de-duplication and pruning,
  never monolithic rewrites (avoids "brevity bias" and "context collapse").
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from ._time import to_iso
from .errors import AccessDeniedError, AlreadyExistsError, NotFoundError, ValidationError
from .frontmatter import Epistemic, join_frontmatter, split_frontmatter
from .retrieval import tokenize
from .store import ABSENT, FileMemoryStore

SKILL_NAME = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
MAX_DESCRIPTION_CHARS = 1024
DESCRIPTION_TOKEN_BUDGET = 100
MAX_BODY_LINES = 500
REUSE_PROMOTION_THRESHOLD = 2


def approx_tokens(text: str) -> int:
    """Cheap token estimate (~4 chars/token) used for budget warnings."""
    return max(1, (len(text) + 3) // 4)


@dataclass
class Skill:
    name: str
    description: str
    path: str
    status: str = "active"
    origin: str = "curated"  # curated | generated
    body: str = ""
    metadata: dict = field(default_factory=dict)

    @property
    def active(self) -> bool:
        return self.status == "active"

    def catalog_line(self) -> str:
        return f"- {self.name}: {self.description}"


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    reason: str


def validate_skill(name: str, description: str, body: str) -> list[str]:
    """Raise on hard errors; return soft warnings."""
    if not SKILL_NAME.match(name):
        raise ValidationError("skill name must be lowercase letters, digits and hyphens (max 64 chars)")
    if not description.strip():
        raise ValidationError("skill description is required")
    if len(description) > MAX_DESCRIPTION_CHARS:
        raise ValidationError(f"skill description exceeds {MAX_DESCRIPTION_CHARS} characters")
    warnings = []
    if approx_tokens(description) > DESCRIPTION_TOKEN_BUDGET:
        warnings.append(f"description is ~{approx_tokens(description)} tokens; keep it under ~{DESCRIPTION_TOKEN_BUDGET}")
    n_lines = len(body.splitlines())
    if n_lines > MAX_BODY_LINES:
        warnings.append(f"SKILL.md body is {n_lines} lines; keep it under {MAX_BODY_LINES} and move detail into linked files")
    return warnings


class SkillLibrary:
    """Skills stored under ``<prefix>/<name>/SKILL.md`` in a memory store (org libraries are read-only)."""

    def __init__(self, store: FileMemoryStore, prefix: str = "skills") -> None:
        self.store = store
        self.prefix = prefix.strip("/")

    def _skill_path(self, name: str) -> str:
        if not SKILL_NAME.match(name):
            raise ValidationError(f"invalid skill name: {name!r}")
        return f"{self.prefix}/{name}/SKILL.md"

    # ------------------------------------------------------------------ level 1: catalog
    def list(self, *, include_candidates: bool = False) -> list[Skill]:
        skills = []
        for rel in self.store.iter_files(self.prefix, suffix="SKILL.md"):
            parts = rel.split("/")
            if len(parts) != 3 or parts[-1] != "SKILL.md":
                continue
            try:
                skill = self._parse(rel)
            except ValidationError:
                continue
            if skill.active or include_candidates:
                skills.append(skill)
        return sorted(skills, key=lambda s: s.name)

    def catalog(self) -> str:
        """Level-1 disclosure: one line per *active* skill for the system prompt."""
        skills = self.list()
        if not skills:
            return ""
        return "Available skills (load one with its name when relevant):\n" + "\n".join(s.catalog_line() for s in skills)

    # ------------------------------------------------------------------ level 2 / 3
    def load(self, name: str) -> Skill:
        return self._parse(self._skill_path(name))

    def resource(self, name: str, rel: str) -> str:
        """Level-3 disclosure: read a file bundled with the skill (sandboxed to the skill dir)."""
        clean = self.store.sandbox.normalise(rel)
        if not clean or clean == "SKILL.md":
            raise ValidationError("use load() for SKILL.md")
        return self.store.read(f"{self.prefix}/{name}/{clean}").content

    def resources(self, name: str) -> list[str]:
        base = f"{self.prefix}/{name}"
        return [r[len(base) + 1 :] for r in self.store.iter_files(base) if not r.endswith("/SKILL.md")]

    def _parse(self, rel: str) -> Skill:
        data, body = split_frontmatter(self.store.read(rel).content)
        if not data or "name" not in data or "description" not in data:
            raise ValidationError(f"{rel} must have name and description frontmatter")
        meta = {k: v for k, v in data.items() if k not in ("name", "description", "status", "origin")}
        return Skill(
            name=str(data["name"]),
            description=str(data["description"]),
            path=rel,
            status=str(data.get("status", "active")),
            origin=str(data.get("origin", "curated")),
            body=body,
            metadata=meta,
        )

    # ------------------------------------------------------------------ writes
    def add_curated(self, name: str, description: str, body: str, *, resources: dict[str, str] | None = None) -> tuple[Skill, list[str]]:
        """Add a human-curated skill (active immediately)."""
        warnings = validate_skill(name, description, body)
        front = {"name": name, "description": description, "status": "active", "origin": "curated", "created": to_iso(self.store.clock())}
        self._write(name, front, body, resources)
        return self.load(name), warnings

    def propose(
        self,
        name: str,
        description: str,
        body: str,
        *,
        sessions: Sequence[str],
        evidence: Sequence[str] = (),
        epistemic: Epistemic = Epistemic.CLAIMED,
        tainted: bool = False,
        resources: dict[str, str] | None = None,
    ) -> tuple[Skill, list[str]]:
        """Record a self-generated skill as a *candidate* (``status: stale``) pending promotion."""
        if not sessions:
            raise ValidationError("generated skills require the originating session ids")
        warnings = validate_skill(name, description, body)
        front = {
            "name": name,
            "description": description,
            "status": "stale",
            "origin": "generated",
            "source": {"sessions": list(sessions), "evidence": list(evidence), "epistemic": epistemic.value},
            "reuse": {"successes": [], "failures": 0},
            "tainted": tainted,
            "created": to_iso(self.store.clock()),
        }
        self._write(name, front, body, resources)
        return self.load(name), warnings

    def _write(self, name: str, front: dict, body: str, resources: dict[str, str] | None) -> None:
        path = self._skill_path(name)
        if self.store.exists(path):
            raise AlreadyExistsError(f"skill {name} already exists")
        with self.store.lock():
            self.store.write(path, join_frontmatter(front, body), expected_sha256=ABSENT)
            for rel, content in (resources or {}).items():
                clean = self.store.sandbox.normalise(rel)
                if not clean or clean == "SKILL.md":
                    raise ValidationError(f"invalid resource path: {rel!r}")
                self.store.write(f"{self.prefix}/{name}/{clean}", content)

    def _update_front(self, name: str, mutate) -> Skill:
        path = self._skill_path(name)
        with self.store.lock():
            f = self.store.read(path)
            data, body = split_frontmatter(f.content)
            if data is None:
                raise ValidationError(f"{path} has no frontmatter")
            mutate(data)
            data["modified"] = to_iso(self.store.clock())
            self.store.write(path, join_frontmatter(data, body), expected_sha256=f.sha256)
        return self.load(name)

    def record_use(self, name: str, session_id: str, *, success: bool) -> Skill:
        """Record a reuse outcome (drives the reuse branch of the promotion gate)."""

        def mutate(data: dict) -> None:
            reuse = data.setdefault("reuse", {"successes": [], "failures": 0})
            if success:
                if session_id not in reuse["successes"]:
                    reuse["successes"].append(session_id)
            else:
                reuse["failures"] = int(reuse.get("failures", 0)) + 1
            data["use_count"] = int(data.get("use_count", 0)) + 1
            data["last_used"] = to_iso(self.store.clock())

        return self._update_front(name, mutate)

    def evaluate_promotion(self, name: str, *, approved_by: str | None = None) -> PromotionDecision:
        skill = self.load(name)
        if skill.active:
            return PromotionDecision(False, "already active")
        meta = skill.metadata
        tainted = bool(meta.get("tainted"))
        if approved_by:
            return PromotionDecision(True, f"approved by {approved_by}")
        if tainted:
            return PromotionDecision(False, "derived from a tainted session; requires human approval")
        source = meta.get("source") or {}
        if source.get("epistemic") == Epistemic.VERIFIED.value and source.get("evidence"):
            return PromotionDecision(True, "backed by verified evidence: " + ", ".join(source["evidence"]))
        reuse = meta.get("reuse") or {}
        successes = [s for s in reuse.get("successes", []) if s not in (source.get("sessions") or [])]
        failures = int(reuse.get("failures", 0))
        if len(successes) >= REUSE_PROMOTION_THRESHOLD and failures < len(successes):
            return PromotionDecision(True, f"reused successfully in {len(successes)} independent sessions")
        return PromotionDecision(False, "needs verified evidence, human approval, or successful reuse in >= 2 other sessions")

    def promote(self, name: str, *, approved_by: str | None = None) -> Skill:
        decision = self.evaluate_promotion(name, approved_by=approved_by)
        if not decision.promote:
            raise AccessDeniedError(f"cannot promote {name}: {decision.reason}")

        def mutate(data: dict) -> None:
            data["status"] = "active"
            data["promotion"] = {"reason": decision.reason, "at": to_iso(self.store.clock())}

        return self._update_front(name, mutate)

    def retire(self, name: str, reason: str) -> Skill:
        def mutate(data: dict) -> None:
            data["status"] = "superseded"
            data["retired_reason"] = reason

        return self._update_front(name, mutate)


# ---------------------------------------------------------------------------- ACE playbook

_BULLET = re.compile(r"^- \[(?P<id>[a-z]+-\d{5})\] (?P<text>.*?) \(helpful=(?P<h>\d+), harmful=(?P<x>\d+)\)\s*$")


@dataclass
class Bullet:
    id: str
    text: str
    helpful: int = 0
    harmful: int = 0

    @property
    def score(self) -> int:
        return self.helpful - self.harmful

    def render(self) -> str:
        return f"- [{self.id}] {self.text} (helpful={self.helpful}, harmful={self.harmful})"


@dataclass(frozen=True)
class DeltaResult:
    op: str
    bullet_id: str
    merged_into: str | None = None


def _similarity(a: str, b: str) -> float:
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


class Playbook:
    """Itemised, counter-bearing playbook file (default sections mirror the Codex MEMORY.md blocks)."""

    DEFAULT_SECTIONS = ("Strategies", "Reusable knowledge", "Failures and how to do differently", "User preferences")

    def __init__(self, store: FileMemoryStore, path: str = "playbook.md", *, dedup_threshold: float = 0.8, id_prefix: str = "pb") -> None:
        self.store = store
        self.path = path
        self.dedup_threshold = dedup_threshold
        self.id_prefix = id_prefix

    def load(self) -> dict[str, list[Bullet]]:
        sections: dict[str, list[Bullet]] = {s: [] for s in self.DEFAULT_SECTIONS}
        if not self.store.exists(self.path):
            return sections
        current = None
        for line in self.store.read(self.path).content.splitlines():
            if line.startswith("## "):
                current = line[3:].strip()
                sections.setdefault(current, [])
                continue
            m = _BULLET.match(line)
            if m and current is not None:
                sections[current].append(Bullet(m.group("id"), m.group("text"), int(m.group("h")), int(m.group("x"))))
        return sections

    def render(self, sections: dict[str, list[Bullet]]) -> str:
        out = ["# Playbook", "", "Itemised lessons. Edited only through delta operations.", ""]
        for name, bullets in sections.items():
            out.append(f"## {name}")
            out.extend(b.render() for b in bullets)
            out.append("")
        return "\n".join(out).rstrip() + "\n"

    def _save(self, sections: dict[str, list[Bullet]], expected: str | None) -> None:
        self.store.write(self.path, self.render(sections), expected_sha256=expected)

    def _next_id(self, sections: dict[str, list[Bullet]]) -> str:
        n = 0
        for bullets in sections.values():
            for b in bullets:
                n = max(n, int(b.id.split("-")[1]))
        return f"{self.id_prefix}-{n + 1:05d}"

    def _expected(self) -> str | None:
        return self.store.read(self.path).sha256 if self.store.exists(self.path) else ABSENT

    def apply(self, ops: Iterable[dict]) -> list[DeltaResult]:
        """Apply delta ops atomically.

        Supported ops::

            {"op": "add", "section": "...", "text": "..."}            # dedups into a near-identical bullet
            {"op": "tag", "id": "pb-00003", "label": "helpful"|"harmful"}
            {"op": "edit", "id": "pb-00003", "text": "..."}           # in-place wording fix, counters kept
            {"op": "remove", "id": "pb-00003"}
        """
        results: list[DeltaResult] = []
        with self.store.lock():
            expected = self._expected()
            sections = self.load()
            by_id = {b.id: b for bs in sections.values() for b in bs}
            for op in ops:
                kind = op.get("op")
                if kind == "add":
                    text = " ".join(str(op["text"]).split())
                    if not text:
                        raise ValidationError("bullet text is empty")
                    section = str(op.get("section") or self.DEFAULT_SECTIONS[0])
                    dup = max(
                        ((b, _similarity(b.text, text)) for b in by_id.values()),
                        key=lambda t: t[1],
                        default=(None, 0.0),
                    )
                    if dup[0] is not None and dup[1] >= self.dedup_threshold:
                        dup[0].helpful += 1
                        results.append(DeltaResult("merge", dup[0].id, merged_into=dup[0].id))
                        continue
                    bullet = Bullet(self._next_id(sections), text)
                    sections.setdefault(section, []).append(bullet)
                    by_id[bullet.id] = bullet
                    results.append(DeltaResult("add", bullet.id))
                elif kind == "tag":
                    b = by_id.get(op["id"])
                    if b is None:
                        raise NotFoundError(f"no bullet {op['id']}")
                    if op.get("label") == "helpful":
                        b.helpful += 1
                    elif op.get("label") == "harmful":
                        b.harmful += 1
                    else:
                        raise ValidationError("tag label must be 'helpful' or 'harmful'")
                    results.append(DeltaResult("tag", b.id))
                elif kind == "edit":
                    b = by_id.get(op["id"])
                    if b is None:
                        raise NotFoundError(f"no bullet {op['id']}")
                    b.text = " ".join(str(op["text"]).split())
                    results.append(DeltaResult("edit", b.id))
                elif kind == "remove":
                    if op["id"] not in by_id:
                        raise NotFoundError(f"no bullet {op['id']}")
                    for bs in sections.values():
                        bs[:] = [b for b in bs if b.id != op["id"]]
                    del by_id[op["id"]]
                    results.append(DeltaResult("remove", op["id"]))
                else:
                    raise ValidationError(f"unknown playbook op: {kind!r}")
            self._save(sections, expected)
        return results

    def prune(self, *, min_score: int = -2) -> list[str]:
        """Drop bullets whose ``helpful - harmful`` fell to ``min_score`` or below."""
        doomed = [b.id for bs in self.load().values() for b in bs if b.score <= min_score]
        if doomed:
            self.apply({"op": "remove", "id": i} for i in doomed)
        return doomed

    def top(self, k: int = 20) -> list[Bullet]:
        bullets = [b for bs in self.load().values() for b in bs]
        return sorted(bullets, key=lambda b: (-b.score, b.id))[:k]
