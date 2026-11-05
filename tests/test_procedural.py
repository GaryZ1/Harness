import pytest

from harness_memory import AccessDeniedError, PathSecurityError, Playbook, SkillLibrary, ValidationError
from harness_memory.frontmatter import Epistemic


def test_curated_skill_progressive_disclosure(store):
    lib = SkillLibrary(store)
    skill, warnings = lib.add_curated(
        "db-migration",
        "Run and verify a Postgres schema migration.",
        "# Steps\n1. Run `scripts/migrate.sh`\n",
        resources={"scripts/migrate.sh": "#!/bin/sh\necho migrate\n"},
    )
    assert skill.active and warnings == []
    assert "- db-migration: Run and verify" in lib.catalog()
    assert "Steps" in lib.load("db-migration").body
    assert lib.resources("db-migration") == ["scripts/migrate.sh"]
    assert "echo migrate" in lib.resource("db-migration", "scripts/migrate.sh")
    with pytest.raises(PathSecurityError):
        lib.resource("db-migration", "../../MEMORY.md")


def test_skill_validation(store):
    lib = SkillLibrary(store)
    with pytest.raises(ValidationError):
        lib.add_curated("Bad Name", "x", "y")
    _, warnings = lib.add_curated("long", "word " * 200, "line\n" * 600)
    assert len(warnings) == 2


def test_generated_skill_promotion_gate(store):
    lib = SkillLibrary(store)
    lib.propose("retry-flaky", "Retry flaky CI jobs once.", "body", sessions=["s1"])
    assert lib.catalog() == ""  # candidates are not advertised
    assert not lib.evaluate_promotion("retry-flaky").promote
    with pytest.raises(AccessDeniedError):
        lib.promote("retry-flaky")
    lib.record_use("retry-flaky", "s1", success=True)  # originating session does not count
    lib.record_use("retry-flaky", "s2", success=True)
    assert not lib.evaluate_promotion("retry-flaky").promote
    lib.record_use("retry-flaky", "s3", success=True)
    assert lib.promote("retry-flaky").active
    assert "retry-flaky" in lib.catalog()


def test_verified_evidence_and_taint(store):
    lib = SkillLibrary(store)
    lib.propose("a-skill", "d", "b", sessions=["s1"], evidence=["tests/test_a.py::test_ok"], epistemic=Epistemic.VERIFIED)
    assert lib.evaluate_promotion("a-skill").promote
    lib.propose("b-skill", "d", "b", sessions=["s1"], evidence=["x"], epistemic=Epistemic.VERIFIED, tainted=True)
    assert not lib.evaluate_promotion("b-skill").promote
    assert lib.promote("b-skill", approved_by="reviewer").active


def test_playbook_deltas(store):
    pb = Playbook(store)
    res = pb.apply(
        [
            {"op": "add", "section": "Strategies", "text": "Run the fast test target before the full suite"},
            {"op": "add", "section": "Failures and how to do differently", "text": "Do not edit generated CI files"},
        ]
    )
    assert [r.op for r in res] == ["add", "add"]
    dup = pb.apply([{"op": "add", "text": "Run the fast test target before the full suite!"}])
    assert dup[0].op == "merge"
    first = res[0].bullet_id
    pb.apply([{"op": "tag", "id": first, "label": "helpful"}, {"op": "tag", "id": res[1].bullet_id, "label": "harmful"}])
    bullets = {b.id: b for b in pb.top()}
    assert bullets[first].helpful == 2
    for _ in range(2):
        pb.apply([{"op": "tag", "id": res[1].bullet_id, "label": "harmful"}])
    assert pb.prune(min_score=-3) == [res[1].bullet_id]
    assert pb.top()[0].id == first
    with pytest.raises(ValidationError):
        pb.apply([{"op": "rewrite_everything"}])
