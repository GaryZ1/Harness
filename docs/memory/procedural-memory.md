# Procedural memory: `harness_memory.procedural`

**Function:** *how* to do things: skills, workflows and tactical lessons.
**Loading policy:** the skill **catalog** (name plus description) is always loaded. A `SKILL.md` body loads when relevant, and bundled files load only as needed. Together these are the three levels of progressive disclosure in Agent Skills.

## Skills (`SkillLibrary`)

```text
skills/<name>/SKILL.md        # frontmatter: name, description, status, origin, source, reuse, ...
skills/<name>/scripts/...     # level-3 resources, sandboxed to the skill directory
```

| Method | Level / effect |
|---|---|
| `catalog()` | Level 1. One line per **active** skill for the system prompt. Candidates are never advertised. |
| `load(name)` | Level 2. Returns the `SKILL.md` body. |
| `resource(name, rel)` / `resources(name)` | Level 3. Bundled files. Path traversal is rejected. |
| `add_curated(...)` | Human-authored skill, active immediately. |
| `propose(..., sessions, evidence, epistemic, tainted)` | Self-generated skill, stored as `status: stale` (a **candidate**). |
| `record_use(name, session_id, success)` | Reuse telemetry: successes per distinct session, failure count, `use_count`, `last_used`. |
| `evaluate_promotion(name, approved_by=None)` / `promote(...)` | The promotion gate (below). |
| `retire(name, reason)` | Marks a skill superseded. |

Validation: names match `^[a-z0-9][a-z0-9-]{0,63}$` and descriptions are ≤ 1024 characters. Warnings are raised when a description exceeds about 100 tokens or a body exceeds 500 lines. Put critical instructions at the top, because compaction truncates from the end.

### Promotion gate

Unverified self-authored skills average about 0 or negative benefit (SkillsBench −1.3 pp, versus +16 pp for curated skills), and they turn "over-specific" (EvoClawBench). A candidate becomes `active` only when one of these holds:

1. `source.epistemic == verified` **and** `source.evidence` cites a passing test or verified run;
2. a human approves it (`approved_by="…"`);
3. it was reused successfully in **≥ 2 sessions other than the ones that produced it**, with fewer failures than successes.

A candidate from a **tainted** session can only be promoted by a human. MemoryGraft attacks target exactly this kind of "successful experience" store.

## Playbook (`Playbook`, ACE-style)

`playbook.md` holds itemised bullets with counters, edited **only through delta operations**:

```markdown
## Strategies
- [pb-00001] Run the fast test target before the full suite (helpful=3, harmful=0)
## Failures and how to do differently
- [pb-00002] Do not edit generated CI files (helpful=1, harmful=0)
```

```python
pb = Playbook(store)
pb.apply([
    {"op": "add", "section": "Strategies", "text": "Check the release calendar before deploys"},
    {"op": "tag", "id": "pb-00001", "label": "helpful"},
    {"op": "edit", "id": "pb-00002", "text": "Never edit ci/generated/*; regenerate instead"},
])
pb.prune(min_score=-2)     # drop bullets whose helpful - harmful fell to -2
pb.top(20)                 # best bullets for prompt injection
```

* An `add` whose text is ≥ 80% token-Jaccard-similar to an existing bullet **merges** into it (`helpful += 1`) instead of duplicating.
* Every `apply` is atomic, with an optimistic-concurrency check on the file hash.
* Whole-file rewrites are impossible by construction. This is the fix for ACE's "context collapse".
