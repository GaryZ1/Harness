from harness_memory import MemoryIndex, MemorySummary
from harness_memory.index import IndexEntry, IndexHealth, truncate_for_prompt


def test_upsert_remove_preserves_freeform(store):
    store.write("MEMORY.md", "# My index\n\nnotes here\n")
    idx = MemoryIndex(store)
    idx.upsert(IndexEntry("Alpha", "a.md", "user"))
    idx.upsert(IndexEntry("Alpha v2", "a.md", "user"))
    idx.upsert(IndexEntry("Beta", "b.md"))
    text = idx.text()
    assert "notes here" in text and text.count("(a.md)") == 1 and "Alpha v2" in text
    idx.remove("a.md")
    assert [e.target for e in idx.entries()] == ["b.md"]
    assert idx.dangling()[0].target == "b.md"


def test_status_and_truncation(store):
    idx = MemoryIndex(store, max_lines=10, max_bytes=10_000)
    for i in range(8):
        st = idx.upsert(IndexEntry(f"m{i}", f"m{i}.md"))
    assert st.health is IndexHealth.OVER_CAP or st.health is IndexHealth.NEAR_CAP
    assert len(truncate_for_prompt("a\n" * 300)) == 400
    assert truncate_for_prompt("x" * 30 + "\n", max_bytes=10) == ""


def test_summary_roundtrip_and_version(store):
    s = MemorySummary()
    s.sections["User Profile"] = "SRE on payments"
    s.save(store)
    loaded = MemorySummary.load(store)
    assert loaded.sections["User Profile"] == "SRE on payments" and loaded.sections["General Tips"] == ""
    assert store.read("memory_summary.md").content.startswith("v1\n")
    assert MemorySummary.parse("v0\n## User Profile\nx") is None
