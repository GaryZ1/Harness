"""Training-ready trajectory schema and export (JSONL), built from eval trajectories or event streams.

Design (reference §11.7 steps 3, 5, 13; §11.8):

* **Native tool roles only.** Tool results are ``tool_result`` blocks (Anthropic shape) in the
  canonical record and ``role: "tool"`` messages in the OpenAI chat export. Tool results smuggled
  into plain user text break native tool-role training on every provider (DeepSeek drops its
  reasoning on user turns), so :meth:`TrainingRecord.validate` rejects unpaired tool calls and
  ``to_openai_chat`` never emits a tool result as a user message.
* **One record serves evals and RL.** A record carries task id, split, seed, rollout id (unique per
  rollout, as on every agent-RFT tool call), profile + model + harness version, the full message
  log, flattened tool calls, the reward with its components, hack signals and the eval outcome.
* **Token-level data passes through** when the loop recorded it (``rollout_details``: token ids /
  logprobs, Harbor ``collect_rollout_details``), so a black-box trainer can attach later.
* **Secrets never leave the box.** Every string is passed through
  :class:`harness_memory.SecretRedactor` on export, except opaque provider payloads that must
  round-trip byte-exact (``signature``, ``encrypted_content``, ``data`` of redacted thinking).
* Exports are atomic (temp file + ``os.replace``) or locked appends.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence

from harness_core.types import Event, Message, Role, ToolResultBlock, ToolUseBlock, utcnow
from harness_evals import Trajectory
from harness_memory import SecretRedactor
from harness_memory.locking import atomic_write_text, file_lock

from .errors import TrajectorySchemaError

SCHEMA_VERSION = "harness_rl.trajectory/v1"
#: Keys whose values are opaque provider payloads and must not be altered by redaction.
OPAQUE_KEYS = frozenset({"signature", "encrypted_content", "data"})


@dataclass
class TrainingRecord:
    rollout_id: str
    task_id: str
    messages: list[dict[str, Any]]
    system: str = ""
    tools: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    reward: float | None = None
    reward_components: dict[str, float] = field(default_factory=dict)
    hack_signals: list[dict[str, str]] = field(default_factory=list)
    outcome: dict[str, Any] = field(default_factory=dict)
    split: str = "train"
    seed: int | None = None
    episode_index: int = 0
    model: str = ""
    profile: str | None = None
    profile_hash: str | None = None
    harness_version: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    turns: int = 0
    rollout_details: dict[str, Any] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: utcnow().isoformat())
    schema: str = SCHEMA_VERSION

    # ------------------------------------------------------------------ invariants
    def validate(self) -> "TrainingRecord":
        if not self.rollout_id or not self.task_id:
            raise TrajectorySchemaError("rollout_id and task_id are required")
        if self.schema != SCHEMA_VERSION:
            raise TrajectorySchemaError(f"unsupported schema {self.schema!r}")
        open_calls: dict[str, int] = {}
        seen_ids: set[str] = set()
        for i, m in enumerate(self.messages):
            role = m.get("role")
            if role not in ("user", "assistant", "system"):
                raise TrajectorySchemaError(f"message {i}: invalid role {role!r}")
            content = m.get("content")
            blocks = content if isinstance(content, list) else []
            for b in blocks:
                t = b.get("type")
                if t == "tool_use":
                    if role != "assistant":
                        raise TrajectorySchemaError(f"message {i}: tool_use outside an assistant message")
                    if b["id"] in seen_ids:
                        raise TrajectorySchemaError(f"message {i}: duplicate tool_use id {b['id']}")
                    seen_ids.add(b["id"])
                    open_calls[b["id"]] = i
                elif t == "tool_result":
                    if role != "user":
                        raise TrajectorySchemaError(f"message {i}: tool_result must be in the native user/tool position")
                    if b.get("tool_use_id") not in open_calls:
                        raise TrajectorySchemaError(f"message {i}: tool_result for unknown or already answered id {b.get('tool_use_id')}")
                    del open_calls[b["tool_use_id"]]
        if open_calls and not self.metadata.get("truncated"):
            raise TrajectorySchemaError(f"tool_use without tool_result: {sorted(open_calls)} (mark metadata.truncated if the rollout was cut off)")
        if self.reward is not None and not isinstance(self.reward, (int, float)):
            raise TrajectorySchemaError("reward must be numeric")
        return self

    # ------------------------------------------------------------------ (de)serialisation
    def to_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "TrainingRecord":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_openai_chat(self) -> list[dict[str, Any]]:
        """Chat-completions shape: assistant ``tool_calls`` and ``role: "tool"`` results (never user text)."""
        out: list[dict[str, Any]] = [{"role": "system", "content": self.system}] if self.system else []
        for m in self.messages:
            content = m.get("content")
            blocks = [{"type": "text", "text": content}] if isinstance(content, str) else list(content or [])
            if m["role"] == "assistant":
                text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
                calls = [{"id": b["id"], "type": "function", "function": {"name": b["name"], "arguments": json.dumps(b.get("input") or {}, sort_keys=True)}}
                         for b in blocks if b.get("type") == "tool_use"]
                msg: dict[str, Any] = {"role": "assistant", "content": text or None}
                if calls:
                    msg["tool_calls"] = calls
                reasoning = [b for b in blocks if b.get("type") in ("thinking", "redacted_thinking")]
                if reasoning:
                    msg["reasoning_blocks"] = reasoning   # kept verbatim for providers that replay reasoning
                out.append(msg)
            else:
                for b in blocks:
                    if b.get("type") == "tool_result":
                        c = b.get("content")
                        text = c if isinstance(c, str) else "".join(str(x.get("text", "")) for x in c or [] if isinstance(x, dict))
                        out.append({"role": "tool", "tool_call_id": b["tool_use_id"], "content": text})
                texts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
                if texts:
                    out.append({"role": m["role"], "content": "".join(texts)})
        return out


# ----------------------------------------------------------------------------- builders


def _messages_to_api(messages: Sequence[Message]) -> list[dict[str, Any]]:
    return [m.to_api() for m in messages]


def from_trajectory(
    traj: Trajectory, *, rollout_id: str, task_id: str, tools: Sequence[Mapping[str, Any]] = (), **fields: Any
) -> TrainingRecord:
    """Build a record from a :class:`harness_evals.Trajectory` (messages, usage, tool calls, rollout details)."""
    rec = TrainingRecord(
        rollout_id=rollout_id, task_id=task_id, messages=_messages_to_api(traj.messages), system=traj.system, tools=[dict(t) for t in tools],
        tool_calls=[c.to_dict() for c in traj.tool_calls], model=traj.model, turns=int(traj.turns or 0),
        usage={"input_tokens": traj.usage.input_tokens, "output_tokens": traj.usage.output_tokens,
               "cache_read_input_tokens": traj.usage.cache_read_input_tokens, "cache_creation_input_tokens": traj.usage.cache_creation_input_tokens},
        rollout_details=traj.metadata.get("rollout_details"),
    )
    for k, v in fields.items():
        if k not in TrainingRecord.__dataclass_fields__:
            raise TypeError(f"unknown TrainingRecord field {k!r}")
        setattr(rec, k, v)
    return rec


def from_events(events: Iterable[Event], *, rollout_id: str, task_id: str, **fields: Any) -> TrainingRecord:
    """Build a record from a session :class:`~harness_core.types.Event` stream (see ``Trajectory.from_events``)."""
    return from_trajectory(Trajectory.from_events(events), rollout_id=rollout_id, task_id=task_id, **fields)


def from_messages(messages: Sequence[Message], *, rollout_id: str, task_id: str, **fields: Any) -> TrainingRecord:
    return from_trajectory(Trajectory.from_messages(messages), rollout_id=rollout_id, task_id=task_id, **fields)


def tool_results_in_user_text(messages: Sequence[Message]) -> list[int]:
    """Indices of user messages that echo a tool's output as plain text instead of a tool_result block.

    Heuristic regression check for loops: a user text block that starts with a tool marker and
    follows an assistant tool_use whose result is missing.
    """
    bad = []
    answered: set[str] = set()
    pending: set[str] = set()
    for i, m in enumerate(messages):
        if m.role is Role.ASSISTANT:
            pending |= {b.id for b in m.content if isinstance(b, ToolUseBlock)}
        elif m.role is Role.USER:
            answered |= {b.tool_use_id for b in m.content if isinstance(b, ToolResultBlock)}
            if pending - answered and m.text.strip():
                bad.append(i)
            pending -= answered
    return bad


# ----------------------------------------------------------------------------- export


def _redact(obj: Any, red: SecretRedactor, key: str | None = None) -> Any:
    if isinstance(obj, str):
        return obj if key in OPAQUE_KEYS else red.redact(obj)
    if isinstance(obj, list):
        return [_redact(x, red) for x in obj]
    if isinstance(obj, dict):
        return {k: _redact(v, red, k) for k, v in obj.items()}
    return obj


class TrajectoryExporter:
    """Writes validated, redacted :class:`TrainingRecord` JSONL."""

    def __init__(self, path: str | Path, *, redactor: SecretRedactor | None = None, fmt: str = "native") -> None:
        if fmt not in ("native", "openai_chat"):
            raise ValueError("fmt must be native|openai_chat")
        self.path = Path(path)
        self.redactor = redactor or SecretRedactor()
        self.fmt = fmt

    def _line(self, rec: TrainingRecord) -> str:
        rec.validate()
        d = rec.to_dict()
        if self.fmt == "openai_chat":
            d["messages"] = rec.to_openai_chat()
            d.pop("system", None)
        return json.dumps(_redact(d, self.redactor), ensure_ascii=False, sort_keys=True, default=str)

    def write(self, records: Iterable[TrainingRecord]) -> int:
        """Atomically replace the file with ``records`` (all validated before anything is written)."""
        lines = [self._line(r) for r in records]
        atomic_write_text(self.path, "".join(line + "\n" for line in lines))
        return len(lines)

    def append(self, rec: TrainingRecord) -> None:
        line = self._line(rec)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with file_lock(self.path.with_suffix(self.path.suffix + ".lock")):
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(line + "\n")


def load_jsonl(path: str | Path) -> Iterator[TrainingRecord]:
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            yield TrainingRecord.from_dict(json.loads(line))


__all__ = [
    "OPAQUE_KEYS",
    "SCHEMA_VERSION",
    "TrainingRecord",
    "TrajectoryExporter",
    "from_events",
    "from_messages",
    "from_trajectory",
    "load_jsonl",
    "tool_results_in_user_text",
]
