from datetime import timedelta

from harness_memory import MemoryIndex, SemanticMemory, UsageLog, forget_derived, parse_citations, strip_citations
from harness_memory.forgetting import ForgettingPolicy, RawMemoryRef, expire_unused, select_raw_memories

REPLY = """The CI runs the fast target first.

<memory-citation>
feedback_ci.md:3-7|note=[CI order]
rollout_summaries/s1_fix-ci.md:12
<session_ids>s1, s2</session_ids>
</memory-citation>"""


def test_parse_and_strip():
    block = parse_citations(REPLY)
    assert [(c.path, c.start, c.end) for c in block.citations] == [("feedback_ci.md", 3, 7), ("rollout_summaries/s1_fix-ci.md", 12, 12)]
    assert block.citations[0].note == "CI order" and block.session_ids == ("s1", "s2")
    assert strip_citations(REPLY) == "The CI runs the fast target first."
    assert parse_citations("no block") is None


def test_usage_folds_into_frontmatter(store, clock):
    sm = SemanticMemory(store)
    r = sm.remember("feedback", "CI order", "Fast target first.")
    log = UsageLog(store, clock=clock)
    clock.advance(hours=1)
    log.record_reply(REPLY.replace("feedback_ci.md", r.path), session_id="s9")
    log.record_reply(REPLY.replace("feedback_ci.md", r.path), session_id="s10")
    stats = log.stats()
    assert stats[r.path].count == 2 and stats["rollout:s1"].count == 2
    modified = sm.get(r.path).meta.modified
    assert log.apply_to_store() == [r.path]
    doc = sm.get(r.path)
    assert doc.meta.use_count == 2 and doc.meta.last_used == clock.now and doc.meta.modified == modified


def test_select_raw_memories(clock):
    now = clock.now
    refs = [
        RawMemoryRef("old", now - timedelta(days=200), "a"),
        RawMemoryRef("used", now - timedelta(days=200), "b", usage_count=5, last_usage=now - timedelta(days=1)),
        RawMemoryRef("new", now - timedelta(days=1), "c"),
    ]
    picked = select_raw_memories(refs, ForgettingPolicy(max_unused_days=90, max_raw_memories=10), now=now)
    assert [r.session_id for r in picked] == ["used", "new"]
    assert len(select_raw_memories(refs, ForgettingPolicy(max_raw_memories=1), now=now)) == 1


def test_expire_unused_archives(store, clock):
    sm = SemanticMemory(store)
    r = sm.remember("reference", "Old dashboard", "Board X in Grafana.")
    clock.advance(days=200)
    rep = expire_unused(store, sm.index, ForgettingPolicy(expire_after_days=180), clock=clock)
    assert rep.archived == [r.path]
    assert store.exists(f"archive/{r.path}") and MemoryIndex(store).get(r.path) is None


def test_forget_derived_cascade(store):
    sm = SemanticMemory(store)
    solo = sm.remember("lesson", "Pin tz in tests", "Set TZ=UTC for date tests.", sessions=["s1"])
    shared = sm.remember("lesson", "Retry flaky job", "Retry once.", sessions=["s1", "s2"])
    rep = forget_derived(store, sm.index, ["s1"])
    assert rep.deleted == [solo.path] and rep.trimmed_provenance == [shared.path]
    assert not store.exists(solo.path)
    assert sm.get(shared.path).meta.source.sessions == ["s2"]
