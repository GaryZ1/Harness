"""Per-model feature flags for scaffolds, frozen per session.

Research basis (build plan "Co-design": "the harness should load a per-model profile rather than
one universal configuration"; reference §10.3 rainbow deploys and checklist items 13 and 17):

* Scaffold choices depend on the model (edit-tool format, todo tools removed on the newest Claude
  models, effort defaults...). Flags are therefore resolved per model with glob patterns from the
  caller's config - **no model id is hard-coded here**.
* Model, effort and tool set are part of the prompt-cache key, and a running session's prompt or
  tools must never be hot-patched: :meth:`FeatureFlags.snapshot` resolves every flag once at session
  start, and the frozen :class:`FlagSnapshot` is what the session uses (store it in the
  :class:`~harness_ops.deploy.SessionPin` snapshot).
* Percentage rollouts use the same deterministic hash as traffic splitting.

Config (YAML/JSON)::

    flags:
      todo_tool:
        default: true
        models: {"test-model-new*": false}      # first matching pattern wins, exact ids beat patterns
      parallel_tools:
        default: false
        rollout_percent: 25                     # deterministic per routing key
        versions: ["v2", "v3"]                  # optional: only on these harness versions
"""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping

import yaml

from ._util import stable_bucket
from .errors import ConfigError


@dataclass(frozen=True)
class FlagRule:
    name: str
    default: Any = False
    models: tuple[tuple[str, Any], ...] = ()
    rollout_percent: float | None = None
    versions: tuple[str, ...] | None = None

    def resolve(self, model: str, *, routing_key: str = "", version: str | None = None) -> Any:
        if self.versions is not None and version is not None and version not in self.versions:
            return _off(self.default)
        for pattern, value in self.models:  # exact id first
            if pattern == model:
                return value
        for pattern, value in self.models:
            if any(ch in pattern for ch in "*?[") and fnmatch.fnmatchcase(model, pattern):
                return value
        if self.rollout_percent is not None:
            bucket = stable_bucket(routing_key, f"flag:{self.name}", 10_000)
            return self.default if bucket < self.rollout_percent * 100 else _off(self.default)
        return self.default


def _off(default: Any) -> Any:
    return (not default) if isinstance(default, bool) else None


@dataclass(frozen=True)
class FlagSnapshot:
    model: str
    version: str | None
    values: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "values", MappingProxyType(dict(self.values)))

    def enabled(self, name: str) -> bool:
        return bool(self.values.get(name, False))

    def get(self, name: str, default: Any = None) -> Any:
        return self.values.get(name, default)

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model, "version": self.version, "values": dict(self.values)}


class FeatureFlags:
    def __init__(self, rules: Mapping[str, FlagRule]) -> None:
        self.rules = MappingProxyType(dict(rules))

    @classmethod
    def from_dict(cls, doc: Mapping[str, Any]) -> "FeatureFlags":
        if not isinstance(doc, Mapping) or not isinstance(doc.get("flags"), Mapping):
            raise ConfigError("expected a mapping with a 'flags' mapping")
        rules: dict[str, FlagRule] = {}
        for name, spec in doc["flags"].items():
            path = f"flags.{name}"
            if not isinstance(spec, Mapping):
                raise ConfigError("must be a mapping", path)
            unknown = set(spec) - {"default", "models", "rollout_percent", "versions"}
            if unknown:
                raise ConfigError(f"unknown keys {sorted(unknown)}", path)
            models = spec.get("models") or {}
            if not isinstance(models, Mapping):
                raise ConfigError("must be a mapping of model id/pattern -> value", f"{path}.models")
            pct = spec.get("rollout_percent")
            if pct is not None and (isinstance(pct, bool) or not isinstance(pct, (int, float)) or not 0 <= pct <= 100):
                raise ConfigError("must be a number within [0, 100]", f"{path}.rollout_percent")
            versions = spec.get("versions")
            if versions is not None and (not isinstance(versions, list) or not all(isinstance(v, str) for v in versions)):
                raise ConfigError("must be a list of version ids", f"{path}.versions")
            rules[str(name)] = FlagRule(
                str(name), spec.get("default", False), tuple((str(k), v) for k, v in models.items()),
                float(pct) if pct is not None else None, tuple(versions) if versions is not None else None,
            )
        return cls(rules)

    @classmethod
    def load(cls, path: str | Path) -> "FeatureFlags":
        try:
            doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        except yaml.YAMLError as exc:
            raise ConfigError(f"invalid YAML/JSON: {exc}") from exc
        return cls.from_dict(doc)

    def value(self, name: str, model: str, *, routing_key: str = "", version: str | None = None) -> Any:
        rule = self.rules.get(name)
        if rule is None:
            raise KeyError(f"unknown flag {name!r}")
        return rule.resolve(model, routing_key=routing_key, version=version)

    def snapshot(self, model: str, *, routing_key: str = "", version: str | None = None) -> FlagSnapshot:
        """Resolve every flag once, at session start. The session never re-reads live flags."""
        return FlagSnapshot(model, version, {n: r.resolve(model, routing_key=routing_key, version=version) for n, r in self.rules.items()})
