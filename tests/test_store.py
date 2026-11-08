import pytest

from harness_memory import (
    Access,
    AccessDeniedError,
    ConcurrencyError,
    FileMemoryStore,
    LimitExceededError,
    MemoryDocument,
    MemoryMeta,
    MemoryType,
    PathSecurityError,
    Provenance,
    Scope,
    StoreLimits,
    ValidationError,
)
from harness_memory.store import ABSENT
from harness_memory.versioning import GitVersioner


def test_write_read_roundtrip(store):
    f = store.write("a.md", "hello\n")
    assert store.read("a.md").content == "hello\n"
    assert store.read("a.md").sha256 == f.sha256


def test_optimistic_concurrency(store):
    f = store.write("a.md", "v1")
    store.write("a.md", "v2", expected_sha256=f.sha256)
    with pytest.raises(ConcurrencyError):
        store.write("a.md", "v3", expected_sha256=f.sha256)
    with pytest.raises(ConcurrencyError):
        store.write("a.md", "x", expected_sha256=ABSENT)


def test_limits(tmp_path, clock):
    s = FileMemoryStore(tmp_path / "m", limits=StoreLimits(max_file_bytes=10, max_files=2), clock=clock)
    with pytest.raises(LimitExceededError):
        s.write("big.md", "x" * 11)
    s.write("1.md", "a")
    s.write("2.md", "b")
    with pytest.raises(LimitExceededError):
        s.write("3.md", "c")
    s.write("1.md", "replace ok")


def test_read_only(tmp_path, clock):
    (tmp_path / "ro").mkdir()
    s = FileMemoryStore(tmp_path / "ro", access=Access.READ_ONLY, scope=Scope.ORG, clock=clock)
    with pytest.raises(AccessDeniedError):
        s.write("x.md", "no")


def test_root_and_protected_paths(store):
    with pytest.raises(PathSecurityError):
        store.delete("")
    with pytest.raises(PathSecurityError):
        store.write(".git/config", "x")


def test_redaction_on_write(store):
    store.write("n.md", "token ghp_abcdefghijklmnopqrstuvwxyz0123456789")
    assert "ghp_" not in store.read("n.md").content


def test_document_stamping_and_validation(store, clock):
    doc = MemoryDocument(MemoryMeta(type=MemoryType.FEEDBACK, description="d"), "body\n", path="feedback_x.md")
    store.write_document(doc)
    loaded = store.read_document("feedback_x.md")
    assert loaded.meta.created == clock.now and loaded.meta.modified == clock.now
    with pytest.raises(ValidationError):
        MemoryDocument(MemoryMeta(type=MemoryType.LESSON), "b", path="lesson.md").to_markdown()
    ok = MemoryDocument(MemoryMeta(type=MemoryType.LESSON, source=Provenance(sessions=["s1"])), "b", path="lesson.md")
    assert "s1" in ok.to_markdown()


def test_tainted_session_stamp(tmp_path, clock):
    s = FileMemoryStore(tmp_path / "t", clock=clock, session_tainted=True)
    doc = MemoryDocument(MemoryMeta(type=MemoryType.USER, description="d"), "b", path="user_x.md")
    s.write_document(doc)
    assert s.read_document("user_x.md").meta.tainted is True


def test_iter_and_rename(store):
    store.write("a/b.md", "1")
    store.write("c.md", "2")
    assert list(store.iter_files()) == ["a/b.md", "c.md"]
    store.rename("c.md", "d/c.md")
    assert store.exists("d/c.md") and not store.exists("c.md")


def test_git_versioner(tmp_path, clock):
    v = GitVersioner(tmp_path / "g")
    v.ensure_repo()
    s = FileMemoryStore(tmp_path / "g", clock=clock, versioner=v)
    s.write("a.md", "1")
    sha = s.commit("feat(memory): add a")
    assert sha
    assert s.commit("chore(memory): nothing") is None
    v.set_baseline()
    s.write("b.md", "2")
    assert "b.md" in v.diff_since_baseline()
