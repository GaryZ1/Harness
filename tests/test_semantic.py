import pytest

from harness_memory import AlreadyExistsError, MemoryType, SemanticMemory, Status, ValidationError
from harness_memory.frontmatter import Epistemic


def test_remember_adds_file_and_index(store):
    sm = SemanticMemory(store)
    res = sm.remember("feedback", "CI runs make test-fast first", "Run `make test-fast` before the full suite.", sessions=["s1"])
    assert res.path == "feedback_ci-runs-make-test-fast-first.md"
    doc = sm.get(res.path)
    assert doc.meta.type is MemoryType.FEEDBACK and doc.meta.source.sessions == ["s1"]
    assert sm.index.get(res.path).note == "feedback"
    with pytest.raises(AlreadyExistsError):
        sm.remember("feedback", "CI runs make test-fast first", "dup")


def test_rejects_derivable_and_requires_provenance(store):
    sm = SemanticMemory(store)
    with pytest.raises(ValidationError):
        sm.remember("project", "path", "src/app/main.py")
    with pytest.raises(ValidationError):
        sm.remember("lesson", "retry flaky", "Retry the flaky integration job once.")
    sm.remember("lesson", "retry flaky", "Retry the flaky integration job once.", evidence=["ci/log#12"])


def test_supersede_keeps_history_and_resolves_current(store, clock):
    sm = SemanticMemory(store)
    first = sm.remember("project", "Release freeze date", "Freeze starts 2026-11-20.", sessions=["s1"])
    clock.advance(days=2)
    res = sm.supersede(first.path, "Freeze moved to 2026-11-27.", reason="PM moved the date", sessions=["s2"], epistemic=Epistemic.VERIFIED)
    assert res.path.endswith("--v2.md")
    old = sm.get(first.path)
    assert old.meta.status is Status.SUPERSEDED and old.meta.superseded_by == res.path and old.meta.valid_to == clock.now
    assert old.meta.extra["conflict_note"] == "PM moved the date"
    assert sm.current(first.path).path == res.path
    assert [d.path for d in sm.history(res.path)] == [first.path, res.path]
    assert sm.index.get(first.path) is None and sm.index.get(res.path)
    assert set(sm.get(res.path).meta.source.sessions) == {"s1", "s2"}


def test_search_default_and_point_in_time(store, clock):
    sm = SemanticMemory(store)
    a = sm.remember("project", "Release freeze date", "Freeze starts 2026-11-20.")
    sm.remember("user", "Prefers terse answers", "User wants short replies without preamble.")
    t0 = clock.now
    clock.advance(days=1)
    b = sm.supersede(a.path, "Freeze moved to 2026-11-27.")
    hits = sm.search("release freeze")
    assert [h.doc_id for h in hits][0] == b.path
    assert a.path not in [h.doc_id for h in hits]
    past = sm.search("release freeze", as_of=t0)
    assert [h.doc_id for h in past] == [a.path]
    assert [h.doc_id for h in sm.search("terse replies", types=["user"])]


def test_stale_and_tainted_filtered(tmp_path, clock):
    from harness_memory import FileMemoryStore

    s = FileMemoryStore(tmp_path / "m", clock=clock, session_tainted=True)
    sm = SemanticMemory(s)
    sm.remember("reference", "Grafana dashboard", "Latency board lives in Grafana under SRE/latency.")
    assert sm.search("grafana latency") == []
    assert sm.search("grafana latency", include_tainted=True)
    s.session_tainted = False
    r2 = sm.remember("reference", "Runbook location", "Runbooks live in Confluence space OPS.")
    sm.mark_stale(r2.path, "space renamed")
    assert sm.search("runbooks confluence") == []
    assert sm.search("runbooks confluence", include_stale=True)


def test_forget(store):
    sm = SemanticMemory(store)
    r = sm.remember("user", "Name is Ana", "The user's name is Ana.")
    sm.forget(r.path)
    assert not store.exists(r.path) and sm.index.entries() == []
