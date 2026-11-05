from harness_memory import MemoryToolHandler
from harness_memory.index import MAX_INDEX_LINES


def h(store, **kw):
    return MemoryToolHandler(store, **kw)


def test_create_view_and_listing(store):
    t = h(store)
    r = t.handle({"command": "create", "path": "/memories/notes.md", "file_text": "line1\nline2\n"})
    assert not r.is_error, r.content
    v = t.handle({"command": "view", "path": "/memories/notes.md"})
    assert "     1\tline1" in v.content and "     2\tline2" in v.content
    part = t.handle({"command": "view", "path": "/memories/notes.md", "view_range": [2, -1]})
    assert "line1" not in part.content and "     2\tline2" in part.content
    d = t.handle({"command": "view", "path": "/memories"})
    assert "/memories/notes.md" in d.content and "excluding hidden items" in d.content


def test_create_refuses_overwrite_by_default(store):
    t = h(store)
    t.handle({"command": "create", "path": "/memories/a.md", "file_text": "x"})
    assert t.handle({"command": "create", "path": "/memories/a.md", "file_text": "y"}).is_error
    assert not h(store, create_overwrites=True).handle({"command": "create", "path": "/memories/a.md", "file_text": "y"}).is_error


def test_str_replace_and_insert(store):
    t = h(store)
    t.handle({"command": "create", "path": "/memories/a.md", "file_text": "alpha\nbeta\nbeta\n"})
    assert t.handle({"command": "str_replace", "path": "/memories/a.md", "old_str": "beta", "new_str": "x"}).is_error
    assert not t.handle({"command": "str_replace", "path": "/memories/a.md", "old_str": "alpha", "new_str": "ALPHA"}).is_error
    assert t.handle({"command": "str_replace", "path": "/memories/a.md", "old_str": "nope", "new_str": "x"}).is_error
    assert not t.handle({"command": "insert", "path": "/memories/a.md", "insert_line": 0, "insert_text": "top"}).is_error
    assert store.read("a.md").content == "top\nALPHA\nbeta\nbeta\n"
    assert t.handle({"command": "insert", "path": "/memories/a.md", "insert_line": 99, "insert_text": "x"}).is_error


def test_delete_rename_root_protection(store):
    t = h(store)
    t.handle({"command": "create", "path": "/memories/a.md", "file_text": "x"})
    assert t.handle({"command": "delete", "path": "/memories"}).is_error
    assert t.handle({"command": "rename", "old_path": "/memories", "new_path": "/memories/x"}).is_error
    assert not t.handle({"command": "rename", "old_path": "/memories/a.md", "new_path": "/memories/b.md"}).is_error
    assert not t.handle({"command": "delete", "path": "/memories/b.md"}).is_error


def test_path_attacks(store):
    t = h(store)
    for p in ["/memories/../x.md", "/etc/passwd", "/memories/%2e%2e%2fx.md", "/memories/..\\x.md"]:
        assert t.handle({"command": "create", "path": p, "file_text": "x"}).is_error, p


def test_view_truncation(store):
    t = h(store)
    store.write("big.md", "y" * 50 + "\n" + ("z" * 99 + "\n") * 400)
    out = t.handle({"command": "view", "path": "/memories/big.md"}).content
    assert len(out) <= 16_000 and "truncated" in out


def test_index_cap_warning(store):
    t = h(store)
    near = "\n".join(f"- [m{i}](m{i}.md)" for i in range(int(MAX_INDEX_LINES * 0.95)))
    r = t.handle({"command": "create", "path": "/memories/MEMORY.md", "file_text": near})
    assert not r.is_error and "close to its limit" in r.content
    over = "\n".join(f"- [m{i}](m{i}.md)" for i in range(MAX_INDEX_LINES + 5))
    r = h(store, create_overwrites=True).handle({"command": "create", "path": "/memories/MEMORY.md", "file_text": over})
    assert r.is_error and "Rewrite the index" in r.content
    assert store.read("MEMORY.md").content.count("\n") >= MAX_INDEX_LINES  # write accepted


def test_restricted_mode(store):
    t = h(store, restricted=True)
    assert t.handle({"command": "create", "path": "/memories/a.md", "file_text": "x"}).is_error
    assert not t.handle({"command": "create", "path": "/memories/inbox/20261103-pref.md", "file_text": "x"}).is_error
    assert t.handle({"command": "delete", "path": "/memories/inbox/20261103-pref.md"}).is_error


def test_errors_never_raise(store):
    t = h(store)
    assert t.handle({"command": "bogus"}).is_error
    assert t.handle({"command": "create", "path": "/memories/a.md"}).is_error
    assert t.handle({"command": "view", "path": "/memories/missing.md"}).is_error
    block = t.handle({"command": "view", "path": "/memories/missing.md"}).to_block("tu_1")
    assert block["is_error"] is True and block["tool_use_id"] == "tu_1"


def test_frontmatter_modified_stamp(store, clock):
    t = h(store)
    t.handle({"command": "create", "path": "/memories/user_role.md", "file_text": "---\ntype: user\n---\nSenior SRE\n"})
    assert "modified: '2026-11-03T10:00:00Z'" in store.read("user_role.md").content
