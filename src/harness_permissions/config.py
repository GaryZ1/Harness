"""``harness-policy.toml`` loader, validator and settings-layer resolution.

The schema is the one proposed in the research reference (§7.7, "Example of our own harness policy
file") - ``[defaults]``, ``[denials]``, ``[rules]``, ``[classifier]``, ``[taint]``, ``[mcp]``,
``[hooks]`` - extended with the sections this package needs to be fully configurable:
``[workspace]``, ``[network]``, ``[exec_policy]``, ``[subagents]``, ``[approvals]``, ``[fatigue]``
and ``[high_assurance]``. Every key is declared once in :data:`SCHEMA` (type, default, merge
strategy, description); the docs page ``docs/permissions/policy-file.md`` is checked against it by
the test-suite.

The same document can be written as TOML, YAML or JSON. TOML uses :mod:`tomllib` (3.11+) or the
optional ``tomli`` package; without either, loading a ``.toml`` file raises
:class:`~harness_permissions.errors.PolicyLoadError` telling you to install ``tomli`` or use YAML/JSON.

**Layers and precedence** (checklist §7.7 step 1): managed/admin > session flags > project > user >
plugins. "A deny at any layer wins, and lower layers can only narrow":

* ``rules.deny`` / ``rules.ask`` and other restriction lists are the **union** of all layers;
* ``rules.allow`` is the union of all layers *except plugins* (plugins can never allow), and only
  the managed layer's allows count when managed sets ``allow_managed_rules_only = true``;
* numeric limits take the **minimum**, enumerations the **strictest** value, modes the **least
  permissive** value across every layer that sets them;
* widening keys (``allow_bypass``, ``allow_managed_rules_only``) are honoured **only** from the
  managed layer; plugin documents may only add restrictions (see :data:`PLUGIN_KEYS`).
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import yaml

from .boundary import Crossing
from .errors import PolicyLoadError, PolicyValidationError
from .execpolicy import ExecPolicy, host_executable, prefix_rule
from .hooks import DEFAULT_TIMEOUTS_S
from .modes import PermissionMode, least_permissive
from .reviewer import DEFAULT_BLOCK_RULES
from .rules import Rule, RuleEffect, lint_rule
from .taint import DEFAULT_SOURCES, DEFAULT_STRICT_ACTIONS, TaintSource


class Layer(str, Enum):
    MANAGED = "managed"
    SESSION = "session"
    PROJECT = "project"
    USER = "user"
    PLUGIN = "plugin"

    @property
    def authority(self) -> int:
        """Lower number = more authority."""
        return LAYER_ORDER.index(self)


LAYER_ORDER = [Layer.MANAGED, Layer.SESSION, Layer.PROJECT, Layer.USER, Layer.PLUGIN]

# ----------------------------------------------------------------------------- type checkers

Checker = Callable[[str, Any], Any]


def _bool(key: str, v: Any) -> bool:
    if not isinstance(v, bool):
        raise PolicyValidationError(f"{key}: expected true/false, got {v!r}")
    return v


def _posint(key: str, v: Any) -> int:
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        raise PolicyValidationError(f"{key}: expected a positive integer, got {v!r}")
    return v


def _posnum(key: str, v: Any) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or v <= 0:
        raise PolicyValidationError(f"{key}: expected a positive number, got {v!r}")
    return float(v)


def _fraction(key: str, v: Any) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 1:
        raise PolicyValidationError(f"{key}: expected a number between 0 and 1, got {v!r}")
    return float(v)


def _str_or_none(key: str, v: Any) -> str | None:
    if v is None or v == "":
        return None
    if not isinstance(v, str):
        raise PolicyValidationError(f"{key}: expected a string, got {v!r}")
    return v


def _strlist(key: str, v: Any) -> list[str]:
    if not isinstance(v, list) or not all(isinstance(x, str) and x.strip() for x in v):
        raise PolicyValidationError(f"{key}: expected a list of non-empty strings, got {v!r}")
    return list(v)


def _enum(*allowed: str) -> Checker:
    def check(key: str, v: Any) -> str:
        if v not in allowed:
            raise PolicyValidationError(f"{key}: expected one of {list(allowed)}, got {v!r}")
        return v
    check.allowed = allowed  # type: ignore[attr-defined]
    return check


def _enumlist(*allowed: str) -> Checker:
    def check(key: str, v: Any) -> list[str]:
        items = _strlist(key, v)
        bad = [x for x in items if x not in allowed]
        if bad:
            raise PolicyValidationError(f"{key}: unknown value(s) {bad}; allowed: {list(allowed)}")
        return items
    check.allowed = allowed  # type: ignore[attr-defined]
    return check


def _mode(key: str, v: Any) -> str:
    try:
        return PermissionMode.parse(v).value
    except Exception as exc:  # noqa: BLE001
        raise PolicyValidationError(f"{key}: {exc}") from exc


def _must_be_true(key: str, v: Any) -> bool:
    if v is not True:
        raise PolicyValidationError(f"{key}: must be true - deny rules always win and cannot be disabled")
    return True


def _inputs(key: str, v: Any) -> list[str]:
    items = _strlist(key, v)
    forbidden = [x for x in items if x in {"tool_results", "tool_outputs", "assistant_text", "assistant_messages"}]
    if forbidden:
        raise PolicyValidationError(f"{key}: the reviewer must never see {forbidden} (reasoning-blind by design)")
    bad = [x for x in items if x not in {"user_messages", "tool_calls"}]
    if bad:
        raise PolicyValidationError(f"{key}: unknown input(s) {bad}; allowed: ['user_messages', 'tool_calls']")
    return items


def _stages(key: str, v: Any) -> list[str]:
    items = _enumlist("single_token_block_biased", "cot_on_flag")(key, v)
    if not items or items[0] != "single_token_block_biased":
        raise PolicyValidationError(f"{key}: the first stage must be 'single_token_block_biased'")
    return items


def _timeouts(key: str, v: Any) -> dict[str, float]:
    if not isinstance(v, dict):
        raise PolicyValidationError(f"{key}: expected a table of hook kind -> seconds")
    out = {}
    for k, s in v.items():
        if k not in DEFAULT_TIMEOUTS_S:
            raise PolicyValidationError(f"{key}.{k}: unknown hook kind; allowed: {sorted(DEFAULT_TIMEOUTS_S)}")
        out[k] = _posnum(f"{key}.{k}", s)
    return out


def _tables(required: Mapping[str, Checker], optional: Mapping[str, Checker]) -> Checker:
    def check(key: str, v: Any) -> list[dict[str, Any]]:
        if not isinstance(v, list):
            raise PolicyValidationError(f"{key}: expected an array of tables")
        out = []
        for i, item in enumerate(v):
            where = f"{key}[{i}]"
            if not isinstance(item, dict):
                raise PolicyValidationError(f"{where}: expected a table")
            unknown = set(item) - set(required) - set(optional)
            if unknown:
                raise PolicyValidationError(f"{where}: unknown key(s) {sorted(unknown)}")
            missing = set(required) - set(item)
            if missing:
                raise PolicyValidationError(f"{where}: missing key(s) {sorted(missing)}")
            out.append({k: (required.get(k) or optional[k])(f"{where}.{k}", val) for k, val in item.items()})
        return out
    return check


def _any(key: str, v: Any) -> Any:
    return v


def _pattern(key: str, v: Any) -> list[Any]:
    if not isinstance(v, list) or not v:
        raise PolicyValidationError(f"{key}: expected a non-empty list of tokens (strings or lists of alternatives)")
    return v


def _examples(key: str, v: Any) -> list[Any]:
    if not isinstance(v, list):
        raise PolicyValidationError(f"{key}: expected a list of example commands")
    return v


_HOOK_TABLE = _tables({"name": _str_or_none, "command": _strlist}, {"matcher": _str_or_none, "timeout_s": _posnum, "events": _strlist})
_PREFIX_TABLE = _tables(
    {"pattern": _pattern, "decision": _enum("allow", "prompt", "forbidden"), "justification": _str_or_none},
    {"match": _examples, "not_match": _examples},
)
_HOST_TABLE = _tables({"name": _str_or_none, "paths": _strlist}, {})

_CONFIGURABLE_SOURCES = tuple(s.value for s in TaintSource if s.value in {"web_fetch", "browser", "mcp:openWorld", "email", "out_of_repo_files", "subagent_report", "memory_read"})
_CROSSINGS = tuple(c.value for c in Crossing)
_STRICTNESS = {
    "denials.headless_on_limit": ["deny", "terminate"],
    "taint.on_strict": ["strict_review", "ask"],
    "mcp.on_description_change": ["warn", "block_and_ask", "block"],
    "hooks.on_failure": ["ignore", "block"],
    "subagents.on_ask": ["escalate", "deny"],
}


@dataclass(frozen=True)
class Key:
    name: str
    check: Checker
    default: Any
    strategy: str      # highest | min | union | concat | strictest | or | least_permissive_mode | managed_only | rules
    doc: str
    vendor: str = "ours"   # "schema" (research schema default), "vendor" (vendor default), "ours"


SCHEMA: dict[str, Key] = {k.name: k for k in [
    Key("version", _posint, 1, "highest", "Schema version. Only 1 exists.", "ours"),
    Key("defaults.interactive_mode", _mode, "auto", "least_permissive_mode", "Mode for interactive sessions: plan | dont-ask | default | accept-edits | auto | bypass.", "schema"),
    Key("defaults.headless_mode", _mode, "dont-ask", "least_permissive_mode", "Mode for headless runs (no human attached).", "schema"),
    Key("defaults.sandbox", _enum("none", "read-only", "workspace-write", "external"), "workspace-write", "highest", "OS sandbox the host runs tools in. Mutating shell commands are boundary-auto-allowed only under workspace-write.", "schema"),
    Key("defaults.deny_wins", _must_be_true, True, "highest", "Deny > ask > allow. Must be true.", "schema"),
    Key("defaults.allow_bypass", _bool, False, "managed_only", "Admin consent for bypass mode (honoured only from the managed layer).", "ours"),
    Key("defaults.allow_managed_rules_only", _bool, False, "managed_only", "Ignore allow rules from every layer except managed.", "ours"),
    Key("denials.escalate_consecutive", _posint, 3, "min", "Consecutive denials before escalating to the human (auto mode: 3).", "vendor"),
    Key("denials.escalate_total", _posint, 20, "min", "Total denials before escalating to the human (auto mode: 20).", "vendor"),
    Key("denials.headless_on_limit", _enum("terminate", "deny"), "terminate", "strictest", "Headless behaviour at the denial limit: terminate the run (claude -p) or keep denying.", "schema"),
    Key("rules.allow", _strlist, [], "rules", "Allow rules (Tool(pattern)). Blanket/interpreter/package-runner wildcards are rejected.", "schema"),
    Key("rules.ask", _strlist, [], "rules", "Ask rules: always require a human, even in bypass mode.", "ours"),
    Key("rules.deny", _strlist, [], "rules", "Deny rules. Union of every layer; matched fuzzily.", "schema"),
    Key("classifier.enabled", _bool, True, "highest", "Use the auto-reviewer in auto mode when one is configured.", "ours"),
    Key("classifier.stages", _stages, ["single_token_block_biased", "cot_on_flag"], "highest", "Reviewer stages. Stage 2 runs only on a stage-1 flag.", "schema"),
    Key("classifier.inputs", _inputs, ["user_messages", "tool_calls"], "highest", "Reviewer inputs. tool_results / assistant text are rejected.", "schema"),
    Key("classifier.environment", _strlist, ["current_git_repo"], "highest", "Environment slot: the trusted boundary described to the reviewer.", "schema"),
    Key("classifier.block_groups", _enumlist(*DEFAULT_BLOCK_RULES), list(DEFAULT_BLOCK_RULES), "union", "Default block-rule groups enabled.", "schema"),
    Key("classifier.block_rules", _strlist, [], "union", "Extra block rules (group 'custom').", "ours"),
    Key("classifier.allow_exceptions", _strlist, [], "highest", "Allow-exceptions slot (widening, so only the highest layer that sets it counts).", "vendor"),
    Key("taint.sources", _enumlist(*_CONFIGURABLE_SOURCES), list(DEFAULT_SOURCES), "union", "Tool categories whose output taints the session.", "schema"),
    Key("taint.strict_actions", _enumlist(*_CROSSINGS), list(DEFAULT_STRICT_ACTIONS), "union", "Crossings that need a human (or strict review) while tainted.", "schema"),
    Key("taint.on_strict", _enum("ask", "strict_review"), "ask", "strictest", "Tainted + strict action: always ask, or allow the stage-1-only reviewer to approve.", "ours"),
    Key("mcp.pin_versions", _bool, True, "or", "Include the MCP server version in the pinned description hash.", "schema"),
    Key("mcp.hash_tool_descriptions", _bool, True, "or", "Hash tool descriptions and detect changes (rug pulls).", "schema"),
    Key("mcp.on_description_change", _enum("block_and_ask", "block", "warn"), "block_and_ask", "strictest", "What a changed description does: ask a human, deny, or only warn.", "schema"),
    Key("mcp.trust_annotations_from", _enumlist("vetted", "none"), ["vetted"], "highest", "Whose MCP annotations are trusted: vetted servers, or none.", "schema"),
    Key("mcp.vetted_servers", _strlist, [], "highest", "MCP server names whose annotations are trusted.", "ours"),
    Key("hooks.on_failure", _enum("block", "ignore"), "block", "strictest", "Hook failure/timeout handling. block = fail closed.", "schema"),
    Key("hooks.timeouts_s", _timeouts, dict(DEFAULT_TIMEOUTS_S), "highest", "Per-kind hook timeouts in seconds.", "vendor"),
    Key("hooks.pre_tool_use", _HOOK_TABLE, [], "concat", "Command hooks: [[hooks.pre_tool_use]] name, command (argv list), matcher (regex), timeout_s, events.", "ours"),
    Key("workspace.roots", _strlist, [], "highest", "Writable roots (empty = enclosing git repo of the engine's cwd). Relative paths resolve against the policy file's directory.", "ours"),
    Key("workspace.read_roots", _strlist, [], "highest", "Extra read-only roots inside the boundary.", "ours"),
    Key("workspace.temp_roots", _strlist, ["/tmp"], "highest", "Temp dirs treated as inside the boundary (Codex workspace-write includes /tmp).", "vendor"),
    Key("workspace.protected_paths", _strlist, [], "union", "Extra protected write paths (added to the built-ins).", "ours"),
    Key("workspace.credential_paths", _strlist, [], "union", "Extra credential paths (read-denied, added to the built-ins).", "ours"),
    Key("workspace.session_branch", _str_or_none, None, "highest", "Branch the session may push to without the git_push_non_session_branch crossing.", "ours"),
    Key("network.allowed_domains", _strlist, [], "highest", "Egress allowlist (example.com exact, *.example.com subdomains).", "ours"),
    Key("network.block_private", _bool, True, "or", "Hard-deny loopback/private/link-local/metadata destinations.", "vendor"),
    Key("exec_policy.prefix_rules", _PREFIX_TABLE, [], "concat", "Codex-style [[exec_policy.prefix_rules]]: pattern, decision, justification, match, not_match.", "vendor"),
    Key("exec_policy.host_executables", _HOST_TABLE, [], "concat", "[[exec_policy.host_executables]]: name, paths (pinned absolute binaries).", "vendor"),
    Key("subagents.max_depth", _posint, 3, "min", "Maximum sub-agent nesting depth (CLAUDE_CODE_MAX_SUBAGENT_SPAWN_DEPTH).", "vendor"),
    Key("subagents.max_concurrent", _posint, 20, "min", "Maximum concurrently running sub-agents (CLAUDE_CODE_MAX_CONCURRENT_SUBAGENTS).", "vendor"),
    Key("subagents.on_ask", _enum("escalate", "deny"), "escalate", "strictest", "A sub-agent call that needs a human: escalate to the broker or deny.", "ours"),
    Key("approvals.timeout_s", _posnum, 300, "min", "Seconds a pending approval waits before it is denied.", "ours"),
    Key("approvals.on_timeout", _enum("deny"), "deny", "highest", "Timeout outcome. Only deny is allowed.", "ours"),
    Key("fatigue.window", _posint, 50, "highest", "Decisions in the ask-rate window.", "ours"),
    Key("fatigue.max_ask_rate", _fraction, 0.25, "highest", "Ask rate above which fatigue is reported.", "ours"),
    Key("fatigue.rubber_stamp_rate", _fraction, 0.93, "highest", "Human approval rate treated as rubber-stamping (93% observed by Anthropic).", "ours"),
    Key("fatigue.min_human_decisions", _posint, 20, "highest", "Human decisions needed before approval-rate/latency checks apply.", "ours"),
    Key("fatigue.fast_approval_s", _posnum, 2.0, "highest", "Median approval latency below which approvals look reflexive.", "ours"),
    Key("fatigue.suggest_after", _posint, 3, "highest", "Approvals of the same pattern before a narrow allow rule is suggested.", "ours"),
    Key("high_assurance.plan_then_execute", _bool, False, "or", "Once tainted, deny calls outside the locked plan (deny all mutations if none is locked).", "ours"),
]}

#: Keys a plugin-layer document may set (restrictions only).
PLUGIN_KEYS = frozenset({
    "version", "rules.deny", "rules.ask", "exec_policy.prefix_rules", "workspace.protected_paths", "workspace.credential_paths",
    "taint.sources", "taint.strict_actions", "classifier.block_rules", "classifier.block_groups",
})


# ----------------------------------------------------------------------------- documents


@dataclass
class PolicyDocument:
    """One validated policy source. ``values`` holds only the keys the source set explicitly."""

    layer: Layer
    values: dict[str, Any]
    source: str = "<inline>"
    rules: list[Rule] = field(default_factory=list)

    @property
    def path(self) -> Path | None:
        return None if self.source.startswith("<") else Path(self.source)


def _flatten(data: Mapping[str, Any], problems: list[str]) -> dict[str, Any]:
    flat: dict[str, Any] = {}
    sections = {k.split(".", 1)[0] for k in SCHEMA if "." in k}
    for top, val in data.items():
        if top == "version":
            flat["version"] = val
            continue
        if top not in sections:
            problems.append(f"{top}: unknown section; known: {sorted(sections)}")
            continue
        if not isinstance(val, Mapping):
            problems.append(f"{top}: expected a table")
            continue
        for k, v in val.items():
            key = f"{top}.{k}"
            if key not in SCHEMA:
                known = sorted(s.split(".", 1)[1] for s in SCHEMA if s.startswith(top + "."))
                problems.append(f"{key}: unknown key; known in [{top}]: {known}")
                continue
            flat[key] = v
    return flat


def parse_policy(data: Mapping[str, Any] | None, layer: Layer | str = Layer.PROJECT, source: str = "<inline>") -> PolicyDocument:
    """Validate a policy mapping. Collects every problem before raising."""
    layer = Layer(layer)
    if data is None:
        data = {}
    if not isinstance(data, Mapping):
        raise PolicyValidationError(f"{source}: policy document must be a table/object")
    problems: list[str] = []
    flat = _flatten(data, problems)
    values: dict[str, Any] = {}
    for key, raw in flat.items():
        try:
            values[key] = SCHEMA[key].check(key, raw)
        except PolicyValidationError as exc:
            problems.extend(exc.problems)
    if layer is Layer.PLUGIN:
        for key in sorted(set(values) - PLUGIN_KEYS):
            problems.append(f"{key}: plugin policy may only add restrictions (allowed keys: {sorted(PLUGIN_KEYS)})")
        for i, r in enumerate(values.get("exec_policy.prefix_rules", [])):
            if r.get("decision") == "allow":
                problems.append(f"exec_policy.prefix_rules[{i}]: plugin policy cannot add allow exec rules")
    if layer not in {Layer.MANAGED}:
        for key in ("defaults.allow_bypass", "defaults.allow_managed_rules_only"):
            if values.get(key) is True:
                problems.append(f"{key}: only the managed (admin) layer may set this to true")
    rules: list[Rule] = []
    for effect in RuleEffect:
        for i, text in enumerate(values.get(f"rules.{effect.value}", [])):
            try:
                rule = Rule.parse(text, effect, layer=layer.value, origin=source)
            except PolicyValidationError as exc:
                problems.extend(f"rules.{effect.value}[{i}]: {p}" for p in exc.problems)
                continue
            lint = lint_rule(rule)
            problems.extend(f"rules.{effect.value}[{i}]: {p}" for p in lint)
            if not lint:
                rules.append(rule)
    for i, r in enumerate(values.get("exec_policy.prefix_rules", [])):
        try:
            prefix_rule(r["pattern"], r["decision"], r.get("justification") or "", r.get("match", []), r.get("not_match", []))
        except PolicyValidationError as exc:
            problems.extend(f"exec_policy.prefix_rules[{i}]: {p}" for p in exc.problems)
    for i, h in enumerate(values.get("exec_policy.host_executables", [])):
        try:
            host_executable(h["name"] or "", h["paths"])
        except PolicyValidationError as exc:
            problems.extend(f"exec_policy.host_executables[{i}]: {p}" for p in exc.problems)
    for i, h in enumerate(values.get("hooks.pre_tool_use", [])):
        if not h.get("name"):
            problems.append(f"hooks.pre_tool_use[{i}].name: required")
    if problems:
        raise PolicyValidationError([f"{source}: {p}" for p in problems])
    return PolicyDocument(layer=layer, values=values, source=source, rules=rules)


def _load_toml(text: str, path: Path) -> Any:
    try:
        import tomllib as toml_mod  # type: ignore[import-not-found]
    except ModuleNotFoundError:  # pragma: no cover - Python 3.10
        try:
            import tomli as toml_mod  # type: ignore[import-not-found,no-redef]
        except ModuleNotFoundError as exc:
            raise PolicyLoadError(
                f"{path}: reading TOML needs Python 3.11+ (tomllib) or the 'tomli' package (pip install tomli); "
                "alternatively write the same schema as .yaml or .json"
            ) from exc
    try:
        return toml_mod.loads(text)
    except Exception as exc:  # noqa: BLE001 - TOMLDecodeError differs between tomllib/tomli
        raise PolicyLoadError(f"{path}: invalid TOML: {exc}") from exc


def load_policy_file(path: str | Path, layer: Layer | str = Layer.PROJECT) -> PolicyDocument:
    """Load ``.toml``, ``.yaml``/``.yml`` or ``.json``; relative ``workspace.*roots`` resolve against the file's directory."""
    p = Path(path).expanduser()
    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        raise PolicyLoadError(f"{p}: cannot read policy file: {exc}") from exc
    suffix = p.suffix.lower()
    if suffix == ".toml":
        data = _load_toml(text, p)
    elif suffix in {".yaml", ".yml"}:
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise PolicyLoadError(f"{p}: invalid YAML: {exc}") from exc
    elif suffix == ".json":
        try:
            data = json.loads(text)
        except json.JSONDecodeError as exc:
            raise PolicyLoadError(f"{p}: invalid JSON: {exc}") from exc
    else:
        raise PolicyLoadError(f"{p}: unsupported policy format {suffix!r} (use .toml, .yaml, .yml or .json)")
    doc = parse_policy(data or {}, layer, str(p.resolve()))
    for key in ("workspace.roots", "workspace.read_roots"):
        if key in doc.values:
            doc.values[key] = [str((p.parent / r).resolve()) if not os.path.isabs(os.path.expanduser(r)) else os.path.expanduser(r) for r in doc.values[key]]
    return doc


def discover_policy_files(project_root: str | Path) -> dict[Layer, list[Path]]:
    """Conventional locations: ``$HARNESS_MANAGED_POLICY`` or ``/etc/harness/``, ``~/.harness/``, project root and ``.harness/``."""
    root = Path(project_root)
    names = ["harness-policy.toml", "harness-policy.yaml", "harness-policy.yml", "harness-policy.json"]
    managed_env = os.environ.get("HARNESS_MANAGED_POLICY")
    candidates = {
        Layer.MANAGED: [Path(managed_env)] if managed_env else [Path("/etc/harness") / n for n in names],
        Layer.USER: [Path.home() / ".harness" / n for n in names],
        Layer.PROJECT: [root / n for n in names] + [root / ".harness" / n for n in names],
    }
    return {layer: [c for c in paths if c.is_file()] for layer, paths in candidates.items()}


# ----------------------------------------------------------------------------- resolution


@dataclass
class ResolvedPolicy:
    """The effective policy after merging every layer."""

    values: dict[str, Any]
    rules: list[Rule]
    exec_policy: ExecPolicy
    hook_specs: list[dict[str, Any]]
    sources: list[str]
    layers_present: set[Layer]

    def __getitem__(self, key: str) -> Any:
        return self.values[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.values.get(key, default)

    @property
    def interactive_mode(self) -> PermissionMode:
        return PermissionMode.parse(self.values["defaults.interactive_mode"])

    @property
    def headless_mode(self) -> PermissionMode:
        return PermissionMode.parse(self.values["defaults.headless_mode"])

    @property
    def allow_bypass(self) -> bool:
        return bool(self.values["defaults.allow_bypass"])

    def rules_with(self, effect: RuleEffect) -> list[Rule]:
        return [r for r in self.rules if r.effect is effect]


def _strictest(key: str, values: list[Any]) -> Any:
    order = _STRICTNESS[key]
    return max(values, key=order.index)


def resolve(documents: Iterable[PolicyDocument]) -> ResolvedPolicy:
    """Merge documents according to each key's strategy (see module docstring)."""
    docs = sorted(documents, key=lambda d: d.layer.authority)
    present = {d.layer for d in docs}
    out: dict[str, Any] = {}
    for name, key in SCHEMA.items():
        setters = [d for d in docs if name in d.values]
        vals = [d.values[name] for d in setters]
        if key.strategy == "rules":
            out[name] = key.default
            continue
        if not setters and key.strategy != "union":
            out[name] = key.default if not isinstance(key.default, (list, dict)) else type(key.default)(key.default)
            continue
        if key.strategy == "highest":
            out[name] = vals[0]
        elif key.strategy == "min":
            out[name] = min(vals)
        elif key.strategy == "union":
            # base = managed's list if managed set it (only admin may drop a default), else the default
            managed_vals = [d.values[name] for d in setters if d.layer is Layer.MANAGED]
            merged: list[Any] = []
            for v in [managed_vals[0] if managed_vals else key.default] + [d.values[name] for d in setters if d.layer is not Layer.MANAGED]:
                merged.extend(x for x in v if x not in merged)
            out[name] = merged
        elif key.strategy == "concat":
            out[name] = [item for v in vals for item in v]
        elif key.strategy == "strictest":
            out[name] = _strictest(name, vals)
        elif key.strategy == "or":
            # true is the strict value: lower layers may turn it on; only managed may turn a default-on flag off
            managed_vals = [d.values[name] for d in setters if d.layer is Layer.MANAGED]
            others = [d.values[name] for d in setters if d.layer is not Layer.MANAGED]
            out[name] = (managed_vals[0] or any(others)) if managed_vals else (any(others) or key.default)
        elif key.strategy == "least_permissive_mode":
            out[name] = least_permissive(*(PermissionMode.parse(v) for v in vals)).value
        elif key.strategy == "managed_only":
            out[name] = next((d.values[name] for d in setters if d.layer is Layer.MANAGED), key.default)
        else:  # pragma: no cover - schema typo
            raise AssertionError(f"unknown strategy {key.strategy}")
    managed_only_allows = out["defaults.allow_managed_rules_only"]
    rules: list[Rule] = []
    for d in docs:
        for r in d.rules:
            if r.effect is RuleEffect.ALLOW and (d.layer is Layer.PLUGIN or (managed_only_allows and d.layer is not Layer.MANAGED)):
                continue
            rules.append(r)
    ep = ExecPolicy()
    for d in docs:
        for r in d.values.get("exec_policy.prefix_rules", []):
            if d.layer is Layer.PLUGIN and r["decision"] == "allow":
                continue
            ep.add(prefix_rule(r["pattern"], r["decision"], r.get("justification") or "", r.get("match", []), r.get("not_match", [])))
    # host executables: the most authoritative layer's pin for a name wins
    for d in reversed(docs):
        for h in d.values.get("exec_policy.host_executables", []):
            ep.pin(host_executable(h["name"], h["paths"]))
    hook_specs = [h for d in docs if d.layer is not Layer.PLUGIN for h in d.values.get("hooks.pre_tool_use", [])]
    return ResolvedPolicy(values=out, rules=rules, exec_policy=ep, hook_specs=hook_specs, sources=[d.source for d in docs], layers_present=present)


def load_layers(
    *,
    managed: Iterable[str | Path] = (),
    project: Iterable[str | Path] = (),
    user: Iterable[str | Path] = (),
    plugins: Iterable[str | Path] = (),
    session: Mapping[str, Any] | None = None,
) -> ResolvedPolicy:
    """Load and merge policy files from every layer."""
    docs: list[PolicyDocument] = []
    for layer, paths in ((Layer.MANAGED, managed), (Layer.PROJECT, project), (Layer.USER, user), (Layer.PLUGIN, plugins)):
        docs.extend(load_policy_file(p, layer) for p in paths)
    if session:
        docs.append(parse_policy(session, Layer.SESSION, "<session flags>"))
    return resolve(docs)


def default_policy() -> ResolvedPolicy:
    """Built-in defaults only (no files)."""
    return resolve([])
