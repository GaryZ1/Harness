"""``MemorySystem`` - one facade wiring every memory type into a harness session lifecycle.

Session lifecycle::

    ms = MemorySystem(base="~/.harness", user_id="u_1", repo_root=".")
    session = ms.start_session("s_123")                 # mounts frozen, transcript opened
    system_prompt += ms.session_start_context(session)  # instructions + summary/index + skills + task state
    ... tool loop: session.memory_tool.handle(input); session.record_tool_output(name) ...
    reply = session.finish_reply(model_reply)           # citations -> usage log, trailer stripped
    after compaction: ms.post_compaction_context(session)
    ms.end_session(session)                             # taint flags persisted
    ms.run_background(llm)                              # Phase 1 + Phase 2 (idle-gated)

Four persistence layers stay separate: **instructions** (human-written, always loaded),
**learned memory** (semantic + procedural + episodic summaries; index always loaded, rest on
demand), **task state** (per repo, under git) and **evidence** (immutable transcripts).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ._time import Clock, utcnow
from .episodic import RolloutSummaryStore, SessionMeta, SessionRegistry, TranscriptLog
from .errors import NotFoundError
from .forgetting import ForgettingPolicy
from .frontmatter import Scope
from .index import MemoryIndex, MemorySummary, truncate_for_prompt
from .instructions import InstructionLoader
from .llm import LLM
from .pipeline.consolidation import ConsolidationReport, Consolidator
from .pipeline.extraction import ExtractionConfig, ExtractionResult, Extractor, JobStore
from .procedural import Playbook, SkillLibrary
from .scopes import MemoryLayout, Mount, MountSet, open_store, repo_id_for
from .security import SecretRedactor, TaintTracker
from .semantic import SemanticMemory
from .store import Access, FileMemoryStore
from .task_state import TaskState
from .tool import ORGANISATION_GUIDANCE, MemoryToolHandler
from .usage import CITATION_INSTRUCTIONS, UsageLog, strip_citations
from .working import ContextBudget, WorkingMemory

READ_PROTOCOL = """<memory-protocol>
You have persistent memory mounted at /memories (the project store). Memory is guidance; the current
environment wins.
- Unless the request is clearly self-contained, do a quick memory pass (ideally <= 4-6 steps): skim the
  summary/index below for keywords -> grep or view the index -> open 1-2 topic files, skills or rollout
  summaries -> search transcripts only for exact evidence.
- Verify cheap, drift-prone facts (paths, versions, commands) before relying on them; otherwise say the fact
  comes from memory and may be stale. Never present unverified memory as confirmed-current.
- ASSUME INTERRUPTION: your context may be reset at any moment. Record progress on long tasks in the task
  progress log, and save durable learnings (preferences, corrections, gotchas) to memory as you go.
- Save only durable, non-derivable information. Never store secrets.
</memory-protocol>"""


@dataclass
class Session:
    meta: SessionMeta
    transcript: TranscriptLog
    memory_tool: MemoryToolHandler
    working: WorkingMemory
    taint: TaintTracker
    usage: UsageLog
    mounts: MountSet
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def session_id(self) -> str:
        return self.meta.session_id

    def record(self, kind: str, payload: dict[str, Any] | str) -> None:
        self.transcript.append(kind, payload)

    def record_tool_output(self, tool_name: str, output: str, *, untrusted: bool | None = None) -> None:
        """Log a tool result and update session taint (untrusted/open-world output pollutes the session)."""
        polluted_before = self.taint.polluted
        self.taint.record_tool_output(tool_name, untrusted=untrusted)
        self.transcript.append("tool_result", {"tool": tool_name, "content": output})
        if self.taint.polluted and not polluted_before:
            # Writes made from now on are stamped `tainted: true`.
            self.memory_tool.store.session_tainted = True

    def finish_reply(self, reply: str) -> str:
        """Record citation usage, log the reply, and return it with the citation trailer removed."""
        self.usage.record_reply(reply, session_id=self.session_id)
        self.transcript.append("assistant", reply)
        return strip_citations(reply)


class MemorySystem:
    def __init__(
        self,
        *,
        base: str | Path = "~/.harness",
        user_id: str,
        repo_root: str | Path,
        repo_id: str | None = None,
        git: bool = True,
        clock: Clock = utcnow,
        redactor: SecretRedactor | None = None,
        restricted_writes: bool = False,
        forgetting: ForgettingPolicy | None = None,
        extraction: ExtractionConfig | None = None,
        context_window_tokens: int | None = None,
    ) -> None:
        self.layout = MemoryLayout(Path(base))
        self.user_id = user_id
        self.repo_root = Path(repo_root).resolve()
        self.repo_id = repo_id or repo_id_for(self.repo_root)
        self.git = git
        self.clock = clock
        self.redactor = redactor or SecretRedactor()
        self.restricted_writes = restricted_writes
        self.forgetting = forgetting or ForgettingPolicy()
        self.extraction = extraction or ExtractionConfig()
        self.context_window_tokens = context_window_tokens

        self.org_store = open_store(self.layout.org_dir, Scope.ORG, access=Access.READ_ONLY, git=False, clock=clock, redactor=self.redactor)
        self.user_store = open_store(self.layout.user_dir(user_id), Scope.USER, git=git, clock=clock, redactor=self.redactor)
        self.project_store = open_store(self.layout.project_dir(self.repo_id), Scope.PROJECT, git=git, clock=clock, redactor=self.redactor)

        self.semantic = SemanticMemory(self.project_store)
        self.user_semantic = SemanticMemory(self.user_store)
        self.playbook = Playbook(self.project_store)
        self.skills = SkillLibrary(self.project_store)
        self.org_skills = SkillLibrary(self.org_store)
        self.rollouts = RolloutSummaryStore(self.project_store)
        self.task = TaskState(self.repo_root, clock=clock, git=git)
        self.instructions = InstructionLoader(self.repo_root, user_file=self.layout.base / "AGENTS.md")
        self.registry = SessionRegistry(self.layout.sessions_base, clock=clock, redactor=self.redactor)
        self.jobs = JobStore(self.layout.base / "state" / "memory_jobs.sqlite")

    # ------------------------------------------------------------------ session lifecycle
    def start_session(
        self,
        session_id: str,
        *,
        ephemeral: bool = False,
        subagent: bool = False,
        parent_session: str | None = None,
        memory_enabled: bool = True,
    ) -> Session:
        meta = self.registry.start(
            session_id,
            ephemeral=ephemeral,
            subagent=subagent,
            parent_session=parent_session,
            memory_enabled=memory_enabled,
            user_id=self.user_id,
            project_id=self.repo_id,
        )
        mounts = MountSet()
        mounts.attach(Mount("org", Scope.ORG, self.org_store, "Organisation standards and shared skills."))
        mounts.attach(Mount("user", Scope.USER, self.user_store, "Your preferences and corrections across projects."))
        project_store = self.project_store
        if subagent:
            # Sub-agents read memory but never write it (Memory(generate=None) semantics).
            project_store = FileMemoryStore(self.project_store.root, scope=Scope.PROJECT, access=Access.READ_ONLY, clock=self.clock)
        mounts.attach(Mount("project", Scope.PROJECT, project_store, "Learned memory for this repository."))
        mounts.freeze()
        working = WorkingMemory(clock=self.clock, budget=ContextBudget(self.context_window_tokens) if self.context_window_tokens else None)
        tool_store = FileMemoryStore(
            project_store.root,
            scope=Scope.PROJECT,
            access=project_store.access,
            clock=self.clock,
            redactor=self.redactor,
            versioner=None,
        )
        return Session(
            meta=meta,
            transcript=self.registry.transcript(session_id),
            memory_tool=MemoryToolHandler(tool_store, restricted=self.restricted_writes),
            working=working,
            taint=TaintTracker(),
            usage=UsageLog(self.project_store, clock=self.clock),
            mounts=mounts,
        )

    def end_session(self, session: Session, *, progress_summary: str | None = None) -> SessionMeta:
        self.registry.apply_taint(session.session_id, session.taint)
        if progress_summary and self.task.initialized:
            self.task.end_session(session.session_id, progress_summary)
        if not session.meta.subagent and self.project_store.writable:
            self.project_store.commit(f"chore(memory): inline writes from session {session.session_id}")
        return self.registry.end(session.session_id)

    # ------------------------------------------------------------------ read path
    def _learned_memory_block(self) -> str:
        summary = MemorySummary.load(self.project_store)
        if summary is not None:
            body = summary.render()
            source = "memory_summary.md"
        else:
            body = MemoryIndex(self.project_store).for_prompt()
            source = "MEMORY.md"
        user_index = truncate_for_prompt(MemoryIndex(self.user_store).text(), max_lines=60, max_bytes=8_000)
        parts = []
        if body.strip():
            parts.append(f'<project-memory source="{source}">\n{body.strip()}\n</project-memory>')
        if user_index.strip():
            parts.append(f'<user-memory source="MEMORY.md">\n{user_index.strip()}\n</user-memory>')
        return "\n".join(parts)

    def session_start_context(self, session: Session, *, include_task_state: bool = True, run_init: bool = False) -> str:
        """Stable, cache-friendly context for the start of a session (put it in the system prompt)."""
        blocks = []
        instr = self.instructions.load()
        if instr.files:
            blocks.append(instr.render())
        blocks.append(READ_PROTOCOL)
        blocks.append(CITATION_INSTRUCTIONS)
        if session.memory_tool.store.writable:
            blocks.append(ORGANISATION_GUIDANCE)
        blocks.append(session.mounts.describe())
        learned = self._learned_memory_block()
        if learned:
            blocks.append(learned)
        catalog = "\n".join(c for c in (self.org_skills.catalog(), self.skills.catalog()) if c)
        if catalog:
            blocks.append(f"<skills>\n{catalog}\n</skills>")
        if include_task_state and self.task.initialized:
            blocks.append(self.task.orient(run_init=run_init).render())
        return "\n\n".join(b for b in blocks if b)

    def post_compaction_context(self, session: Session) -> str:
        """Re-inject instructions, memory summary/index, task state and the plan after compaction."""
        session.working.after_compaction()
        return self.session_start_context(session) + "\n\n" + session.working.recitation()

    def search(self, query: str, **kwargs: Any):
        return self.semantic.search(query, **kwargs)

    # ------------------------------------------------------------------ background write path
    def run_background(
        self, llm: LLM, *, rate_limit_remaining_percent: float = 100.0
    ) -> tuple[list[ExtractionResult], ConsolidationReport]:
        """Phase 1 over idle sessions, then Phase 2 consolidation of the project store."""
        extractor = Extractor(
            llm, self.registry, self.project_store, self.jobs, config=self.extraction, clock=self.clock, redactor=self.redactor
        )
        results = extractor.run(rate_limit_remaining_percent=rate_limit_remaining_percent)
        report = Consolidator(
            llm,
            self.project_store,
            semantic=self.semantic,
            playbook=self.playbook,
            skills=self.skills,
            policy=self.forgetting,
            clock=self.clock,
        ).run()
        return results, report

    def forget_session(self, session_id: str) -> list[str]:
        """User-requested "forget this session": drop its raw memory + rollout summary; Phase 2 cascades."""
        removed = self.rollouts.delete_for_session(session_id)
        raw = f"raw_memories/{session_id}.json"
        try:
            self.project_store.delete(raw)
            removed.append(raw)
        except NotFoundError:
            pass
        return removed
