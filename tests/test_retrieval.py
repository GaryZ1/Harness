from datetime import datetime, timedelta, timezone

from harness_memory.retrieval import BM25Index, HybridRetriever, IndexedItem, grep, tokenize


def test_tokenize_splits_identifiers():
    toks = tokenize("Run make test-fast on main.py")
    assert "test-fast" in toks and "fast" in toks and "main.py" in toks and "on" not in toks


def test_bm25_ranks_relevant_first():
    idx = BM25Index()
    idx.add("a", "postgres migration rollback procedure")
    idx.add("b", "frontend css color tokens")
    idx.add("c", "postgres connection pool size")
    res = idx.search("postgres rollback")
    assert res[0][0] == "a" and {d for d, _ in res} == {"a", "c"}
    idx.remove("a")
    assert idx.search("rollback") == []


def test_hybrid_with_embedder_and_recency():
    def emb(texts):
        return [[1.0 if "db" in t.lower() or "database" in t.lower() else 0.0, 1.0] for t in texts]

    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    r = HybridRetriever(embedder=emb)
    r.upsert(
        [
            IndexedItem("old", "database host is db1", timestamp=t0),
            IndexedItem("new", "database host is db2", timestamp=t0 + timedelta(days=30)),
            IndexedItem("other", "css theme", timestamp=t0),
        ]
    )
    hits = r.search("current database host")
    assert hits[0].doc_id == "new" and "recency" in hits[0].signals
    assert all(h.score >= 0.1 for h in hits)
    past = r.search("database host", as_of=t0 + timedelta(days=1))
    assert "new" not in [h.doc_id for h in past]


def test_validity_window():
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    r = HybridRetriever()
    r.upsert([IndexedItem("x", "fact alpha", timestamp=t0, valid_to=t0 + timedelta(days=1))])
    assert r.search("alpha", as_of=t0 + timedelta(hours=1))
    assert not r.search("alpha", as_of=t0 + timedelta(days=2))


def test_grep():
    hits = grep([("a.md", "one\nTwo\nthree"), ("b.md", "two again")], r"two")
    assert hits == [("a.md", 2, "Two"), ("b.md", 1, "two again")]
