"""JSON Schema for tools: a stdlib validator, a schema-from-signature builder and a per-provider compiler.

Research basis (``reports/SOTA agentic harness reference.md`` §5.7 steps 6-7, §5.8 "Provider-specific"):

* **Validate before running.** Tool inputs are parsed as JSON objects and checked against the
  tool's schema *before* the handler runs; failures become ``is_error`` tool results listing every
  problem with a JSON-pointer path and a corrected example (MCP SEP-1303: validation errors are
  tool-execution errors "to enable model self-correction").
* **One canonical schema, compiled per provider.** Author one JSON Schema 2020-12 document per
  tool (MCP's default dialect since 2025-11-25, SEP-1613) and compile it: Anthropic
  ``input_schema`` (``strict`` needs ``additionalProperties: false``), OpenAI strict (every
  property ``required``, optional ones as nullable unions, ``additionalProperties: false``) and
  DeepSeek strict (the OpenAI rules plus no ``minLength``/``maxLength``/``minItems``/``maxItems``).
  :func:`lint_schema` flags constructs a target rejects so CI can catch them.

The validator implements the subset of JSON Schema 2020-12 that tool schemas use in practice:
``type`` (incl. type lists), ``enum``, ``const``, ``properties``, ``required``,
``additionalProperties``, ``patternProperties``, ``minProperties``/``maxProperties``,
``dependentRequired``, ``items``, ``prefixItems``, ``minItems``/``maxItems``, ``uniqueItems``,
``minLength``/``maxLength``, ``pattern``, ``minimum``/``maximum``,
``exclusiveMinimum``/``exclusiveMaximum``, ``multipleOf``, ``allOf``/``anyOf``/``oneOf``/``not``,
``if``/``then``/``else`` and local ``$ref`` (``#/$defs/...``, ``#/definitions/...``). ``format``
and other annotation keywords are accepted and ignored (as the spec allows). Unknown keywords
are ignored, so third-party MCP schemas never fail registration; :func:`lint_schema` reports them.
"""

from __future__ import annotations

import copy
import enum
import inspect
import json
import math
import re
import types
import typing
from collections import abc
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Callable, Iterable, Literal, Mapping

from .errors import SchemaError

JSON_TYPES = frozenset({"string", "number", "integer", "boolean", "object", "array", "null"})

#: Keywords :func:`validate` enforces. Everything else is treated as an annotation.
VALIDATED_KEYWORDS = frozenset(
    {
        "type", "enum", "const", "properties", "required", "additionalProperties", "patternProperties",
        "minProperties", "maxProperties", "dependentRequired", "items", "prefixItems", "minItems",
        "maxItems", "uniqueItems", "minLength", "maxLength", "pattern", "minimum", "maximum",
        "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "allOf", "anyOf", "oneOf", "not", "if",
        "then", "else", "$ref",
    }
)
ANNOTATION_KEYWORDS = frozenset(
    {
        "$schema", "$id", "$defs", "definitions", "$comment", "title", "description", "default", "examples",
        "format", "deprecated", "readOnly", "writeOnly", "contentMediaType", "contentEncoding",
    }
)

#: Keywords DeepSeek strict mode rejects (search-summarised DeepSeek tool-calls guide).
DEEPSEEK_STRICT_UNSUPPORTED = frozenset({"minLength", "maxLength", "minItems", "maxItems"})

Target = Literal["anthropic", "anthropic_strict", "openai_strict", "deepseek_strict"]

_MAX_ISSUES = 25
_MAX_REF_DEPTH = 64


# ----------------------------------------------------------------------------- issues


@dataclass(frozen=True)
class ValidationIssue:
    """One validation failure. ``path`` is a JSON pointer into the instance ('' = the root)."""

    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path or '(root)'}: {self.message}"


def format_issues(issues: Iterable[ValidationIssue]) -> str:
    return "\n".join(f"- {i}" for i in issues)


def _ptr(path: str, token: str | int) -> str:
    t = str(token).replace("~", "~0").replace("/", "~1")
    return f"{path}/{t}"


# ----------------------------------------------------------------------------- schema checks


def check_schema(schema: Any, *, _root: Any = None, _path: str = "#") -> None:
    """Raise :class:`SchemaError` if ``schema`` is structurally malformed.

    Only structural keywords are checked (``type`` values, ``required`` lists, sub-schema shapes,
    regex compilability, resolvable local ``$ref``). Unknown keywords are allowed.
    """
    root = schema if _root is None else _root
    if isinstance(schema, bool):
        return
    if not isinstance(schema, dict):
        raise SchemaError(f"{_path}: schema must be an object or boolean, got {type(schema).__name__}")
    t = schema.get("type")
    if t is not None:
        names = t if isinstance(t, list) else [t]
        bad = [n for n in names if n not in JSON_TYPES]
        if bad or not names:
            raise SchemaError(f"{_path}/type: unknown JSON type(s) {bad or names!r}")
    req = schema.get("required")
    if req is not None and not (isinstance(req, list) and all(isinstance(r, str) for r in req)):
        raise SchemaError(f"{_path}/required: must be a list of property names")
    if "enum" in schema and not isinstance(schema["enum"], list):
        raise SchemaError(f"{_path}/enum: must be a list")
    if "pattern" in schema:
        try:
            _compile(schema["pattern"])
        except (re.error, TypeError) as exc:
            raise SchemaError(f"{_path}/pattern: invalid regular expression: {exc}") from exc
    for kw in ("properties", "patternProperties", "$defs", "definitions"):
        sub = schema.get(kw)
        if sub is None:
            continue
        if not isinstance(sub, dict):
            raise SchemaError(f"{_path}/{kw}: must be an object")
        for name, s in sub.items():
            if kw == "patternProperties":
                try:
                    _compile(name)
                except re.error as exc:
                    raise SchemaError(f"{_path}/{kw}: invalid regex {name!r}: {exc}") from exc
            check_schema(s, _root=root, _path=f"{_path}/{kw}/{name}")
    for kw in ("additionalProperties", "items", "not", "if", "then", "else"):
        if kw in schema:
            check_schema(schema[kw], _root=root, _path=f"{_path}/{kw}")
    for kw in ("allOf", "anyOf", "oneOf", "prefixItems"):
        if kw in schema:
            subs = schema[kw]
            if not isinstance(subs, list) or (kw != "prefixItems" and not subs):
                raise SchemaError(f"{_path}/{kw}: must be a non-empty list of schemas")
            for i, s in enumerate(subs):
                check_schema(s, _root=root, _path=f"{_path}/{kw}/{i}")
    for kw in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"):
        if kw in schema and not _is_number(schema[kw]):
            raise SchemaError(f"{_path}/{kw}: must be a number")
    if "multipleOf" in schema and schema["multipleOf"] <= 0:
        raise SchemaError(f"{_path}/multipleOf: must be > 0")
    for kw in ("minLength", "maxLength", "minItems", "maxItems", "minProperties", "maxProperties"):
        if kw in schema and not (isinstance(schema[kw], int) and not isinstance(schema[kw], bool) and schema[kw] >= 0):
            raise SchemaError(f"{_path}/{kw}: must be a non-negative integer")
    if "$ref" in schema:
        _resolve_ref(root, schema["$ref"])


def _resolve_ref(root: Any, ref: Any) -> Any:
    if not isinstance(ref, str) or not ref.startswith("#"):
        raise SchemaError(f"only local $ref values ('#/...') are supported, got {ref!r}")
    node = root
    for raw in ref[1:].split("/")[1:] if ref != "#" else []:
        token = raw.replace("~1", "/").replace("~0", "~")
        if isinstance(node, dict) and token in node:
            node = node[token]
        elif isinstance(node, list) and token.isdigit() and int(token) < len(node):
            node = node[int(token)]
        else:
            raise SchemaError(f"unresolvable $ref {ref!r}")
    return node


@lru_cache(maxsize=512)
def _compile(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern)


# ----------------------------------------------------------------------------- validation


def _is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _type_ok(v: Any, t: str) -> bool:
    if t == "string":
        return isinstance(v, str)
    if t == "integer":
        return (isinstance(v, int) and not isinstance(v, bool)) or (isinstance(v, float) and v.is_integer())
    if t == "number":
        return _is_number(v) and not (isinstance(v, float) and (math.isnan(v) or math.isinf(v)))
    if t == "boolean":
        return isinstance(v, bool)
    if t == "object":
        return isinstance(v, dict)
    if t == "array":
        return isinstance(v, list)
    if t == "null":
        return v is None
    return False


def _json_type(v: Any) -> str:
    for t in ("null", "boolean", "integer", "number", "string", "array", "object"):
        if _type_ok(v, t):
            return t
    return type(v).__name__


def _canon(v: Any) -> str:
    """Canonical JSON so ``1 != True`` and ``{"a":1} == {"a":1}`` regardless of key order."""
    return json.dumps(v, sort_keys=True, separators=(",", ":"), default=repr)


def validate(instance: Any, schema: Any, *, root: Any = None) -> list[ValidationIssue]:
    """Validate ``instance`` against ``schema`` and return every issue found (empty = valid)."""
    issues: list[ValidationIssue] = []
    _validate(instance, schema, root if root is not None else schema, "", issues, 0)
    return issues[:_MAX_ISSUES]


def is_valid(instance: Any, schema: Any) -> bool:
    return not validate(instance, schema)


def _validate(v: Any, s: Any, root: Any, path: str, out: list[ValidationIssue], depth: int) -> None:
    if len(out) >= _MAX_ISSUES:
        return
    if s is True or s == {}:
        return
    if s is False:
        out.append(ValidationIssue(path, "no value is allowed here"))
        return
    if not isinstance(s, dict):
        return
    if "$ref" in s:
        if depth > _MAX_REF_DEPTH:
            out.append(ValidationIssue(path, "schema $ref recursion too deep"))
            return
        _validate(v, _resolve_ref(root, s["$ref"]), root, path, out, depth + 1)

    t = s.get("type")
    if t is not None:
        names = t if isinstance(t, list) else [t]
        if not any(_type_ok(v, n) for n in names):
            out.append(ValidationIssue(path, f"expected {' or '.join(names)}, got {_json_type(v)}"))
            return  # further checks would only produce noise

    if "enum" in s and _canon(v) not in {_canon(e) for e in s["enum"]}:
        out.append(ValidationIssue(path, f"must be one of {json.dumps(s['enum'])}, got {json.dumps(v, default=repr)}"))
    if "const" in s and _canon(v) != _canon(s["const"]):
        out.append(ValidationIssue(path, f"must equal {json.dumps(s['const'])}"))

    if isinstance(v, str):
        if "minLength" in s and len(v) < s["minLength"]:
            out.append(ValidationIssue(path, f"must be at least {s['minLength']} characters"))
        if "maxLength" in s and len(v) > s["maxLength"]:
            out.append(ValidationIssue(path, f"must be at most {s['maxLength']} characters (got {len(v)})"))
        if "pattern" in s and not _compile(s["pattern"]).search(v):
            out.append(ValidationIssue(path, f"must match pattern {s['pattern']!r}"))

    if _is_number(v):
        if "minimum" in s and v < s["minimum"]:
            out.append(ValidationIssue(path, f"must be >= {s['minimum']}"))
        if "maximum" in s and v > s["maximum"]:
            out.append(ValidationIssue(path, f"must be <= {s['maximum']}"))
        if "exclusiveMinimum" in s and v <= s["exclusiveMinimum"]:
            out.append(ValidationIssue(path, f"must be > {s['exclusiveMinimum']}"))
        if "exclusiveMaximum" in s and v >= s["exclusiveMaximum"]:
            out.append(ValidationIssue(path, f"must be < {s['exclusiveMaximum']}"))
        if "multipleOf" in s:
            q = v / s["multipleOf"]
            if not math.isclose(q, round(q), rel_tol=0, abs_tol=1e-9):
                out.append(ValidationIssue(path, f"must be a multiple of {s['multipleOf']}"))

    if isinstance(v, dict):
        _validate_object(v, s, root, path, out, depth)
    if isinstance(v, list):
        _validate_array(v, s, root, path, out, depth)

    for sub in s.get("allOf", ()):
        _validate(v, sub, root, path, out, depth + 1)
    if "anyOf" in s and not any(not validate(v, sub, root=root) for sub in s["anyOf"]):
        out.append(ValidationIssue(path, "must match at least one of the allowed forms (anyOf)"))
    if "oneOf" in s:
        n = sum(1 for sub in s["oneOf"] if not validate(v, sub, root=root))
        if n != 1:
            out.append(ValidationIssue(path, f"must match exactly one allowed form (oneOf); matched {n}"))
    if "not" in s and not validate(v, s["not"], root=root):
        out.append(ValidationIssue(path, "matches a form that is explicitly disallowed (not)"))
    if "if" in s:
        branch = s.get("then") if not validate(v, s["if"], root=root) else s.get("else")
        if branch is not None:
            _validate(v, branch, root, path, out, depth + 1)


def _validate_object(v: dict, s: dict, root: Any, path: str, out: list[ValidationIssue], depth: int) -> None:
    props: dict = s.get("properties", {}) or {}
    for name in s.get("required", ()):
        if name not in v:
            out.append(ValidationIssue(_ptr(path, name), "is a required property and is missing"))
    if "minProperties" in s and len(v) < s["minProperties"]:
        out.append(ValidationIssue(path, f"must have at least {s['minProperties']} properties"))
    if "maxProperties" in s and len(v) > s["maxProperties"]:
        out.append(ValidationIssue(path, f"must have at most {s['maxProperties']} properties"))
    for name, deps in (s.get("dependentRequired") or {}).items():
        if name in v:
            for d in deps:
                if d not in v:
                    out.append(ValidationIssue(_ptr(path, d), f"is required when {name!r} is present"))
    pattern_props: dict = s.get("patternProperties", {}) or {}
    additional = s.get("additionalProperties", True)
    for key, val in v.items():
        matched = False
        if key in props:
            matched = True
            _validate(val, props[key], root, _ptr(path, key), out, depth + 1)
        for pat, sub in pattern_props.items():
            if _compile(pat).search(key):
                matched = True
                _validate(val, sub, root, _ptr(path, key), out, depth + 1)
        if not matched:
            if additional is False:
                allowed = ", ".join(sorted(props)) or "(none)"
                out.append(ValidationIssue(_ptr(path, key), f"unexpected property; allowed properties: {allowed}"))
            elif isinstance(additional, dict):
                _validate(val, additional, root, _ptr(path, key), out, depth + 1)


def _validate_array(v: list, s: dict, root: Any, path: str, out: list[ValidationIssue], depth: int) -> None:
    if "minItems" in s and len(v) < s["minItems"]:
        out.append(ValidationIssue(path, f"must have at least {s['minItems']} items"))
    if "maxItems" in s and len(v) > s["maxItems"]:
        out.append(ValidationIssue(path, f"must have at most {s['maxItems']} items (got {len(v)})"))
    if s.get("uniqueItems"):
        seen: set[str] = set()
        for i, item in enumerate(v):
            c = _canon(item)
            if c in seen:
                out.append(ValidationIssue(_ptr(path, i), "duplicate item (uniqueItems)"))
            seen.add(c)
    prefix = s.get("prefixItems") or []
    for i, sub in enumerate(prefix[: len(v)]):
        _validate(v[i], sub, root, _ptr(path, i), out, depth + 1)
    items = s.get("items")
    if items is not None:
        for i in range(len(prefix), len(v)):
            _validate(v[i], items, root, _ptr(path, i), out, depth + 1)


def apply_defaults(instance: dict[str, Any], schema: Mapping[str, Any]) -> dict[str, Any]:
    """Return a copy of ``instance`` with missing top-level properties filled from ``default``."""
    out = dict(instance)
    for name, sub in (schema.get("properties") or {}).items():
        if name not in out and isinstance(sub, dict) and "default" in sub:
            out[name] = copy.deepcopy(sub["default"])
    return out


# ----------------------------------------------------------------------------- examples


def example_from_schema(schema: Any, *, root: Any = None, depth: int = 0, required_only: bool = True) -> Any:
    """Synthesise a minimal valid-looking example (used in "corrected input" error messages)."""
    root = schema if root is None else root
    if not isinstance(schema, dict) or depth > 8:
        return None
    if "$ref" in schema:
        try:
            return example_from_schema(_resolve_ref(root, schema["$ref"]), root=root, depth=depth + 1, required_only=required_only)
        except SchemaError:
            return None
    for key in ("examples",):
        if isinstance(schema.get(key), list) and schema[key]:
            return copy.deepcopy(schema[key][0])
    if "default" in schema:
        return copy.deepcopy(schema["default"])
    if "const" in schema:
        return copy.deepcopy(schema["const"])
    if isinstance(schema.get("enum"), list) and schema["enum"]:
        return copy.deepcopy(schema["enum"][0])
    for kw in ("anyOf", "oneOf"):
        if isinstance(schema.get(kw), list) and schema[kw]:
            non_null = [s for s in schema[kw] if not (isinstance(s, dict) and s.get("type") == "null")]
            return example_from_schema((non_null or schema[kw])[0], root=root, depth=depth + 1, required_only=required_only)
    t = schema.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), t[0] if t else None)
    if t == "object" or (t is None and "properties" in schema):
        props = schema.get("properties") or {}
        req = set(schema.get("required") or ())
        names = [n for n in props if n in req] if required_only else list(props)
        return {n: example_from_schema(props[n], root=root, depth=depth + 1, required_only=required_only) for n in names}
    if t == "array":
        item = example_from_schema(schema.get("items", {}), root=root, depth=depth + 1, required_only=required_only)
        n = max(int(schema.get("minItems", 1) or 0), 1)
        return [item] * n if item is not None else []
    if t == "string":
        fmt = schema.get("format")
        sample = {"date": "2026-01-31", "date-time": "2026-01-31T12:00:00Z", "uri": "https://example.com", "email": "user@example.com"}.get(
            fmt or "", "..."
        )
        return sample.ljust(int(schema.get("minLength", 0) or 0), ".")
    if t == "integer":
        if "minimum" in schema:
            return int(math.ceil(schema["minimum"]))
        if "exclusiveMinimum" in schema:
            return int(math.floor(schema["exclusiveMinimum"])) + 1
        return 0
    if t == "number":
        return float(schema.get("minimum", 0))
    if t == "boolean":
        return False
    if t == "null":
        return None
    return None


# ----------------------------------------------------------------------------- schema from signature


_DOC_PARAM_SPHINX = re.compile(r"^\s*:param\s+(?:[^:]*\s)?(\w+):\s*(.*)$")
_DOC_SECTION = re.compile(r"^\s*(Args|Arguments|Parameters|Returns|Raises|Yields|Examples?):\s*$")
_DOC_GOOGLE_ARG = re.compile(r"^\s+(\*{0,2}\w+)\s*(?:\([^)]*\))?\s*:\s*(.*)$")


def parse_docstring(doc: str | None) -> tuple[str, dict[str, str]]:
    """Split a docstring into (description, {param: description}). Supports Sphinx and Google styles."""
    if not doc:
        return "", {}
    lines = inspect.cleandoc(doc).splitlines()
    desc_lines: list[str] = []
    params: dict[str, str] = {}
    section: str | None = None
    current: str | None = None
    for line in lines:
        m_sec = _DOC_SECTION.match(line)
        if m_sec:
            section, current = m_sec.group(1), None
            continue
        m_sp = _DOC_PARAM_SPHINX.match(line)
        if m_sp:
            current = m_sp.group(1)
            params[current] = m_sp.group(2).strip()
            section = "sphinx"
            continue
        if line.strip().startswith(":") and section in (None, "sphinx"):  # :returns:, :raises: ...
            section, current = "sphinx-other", None
            continue
        if section in ("Args", "Arguments", "Parameters"):
            m_g = _DOC_GOOGLE_ARG.match(line)
            if m_g:
                current = m_g.group(1).lstrip("*")
                params[current] = m_g.group(2).strip()
                continue
            if current and line.strip():
                params[current] = (params[current] + " " + line.strip()).strip()
            continue
        if section == "sphinx" and current and line.startswith((" ", "\t")) and line.strip():
            params[current] = (params[current] + " " + line.strip()).strip()
            continue
        if section is None:
            desc_lines.append(line)
    return "\n".join(desc_lines).strip(), params


def _is_run_context(annotation: Any) -> bool:
    return getattr(annotation, "__name__", None) == "RunContext"


def type_to_schema(tp: Any) -> dict[str, Any]:
    """Map a Python annotation to JSON Schema (best effort; unknown types map to ``{}``)."""
    if tp is inspect.Parameter.empty or tp is Any:
        return {}
    origin = typing.get_origin(tp)
    args = typing.get_args(tp)
    if origin is typing.Annotated:
        base = type_to_schema(args[0])
        for extra in args[1:]:
            if isinstance(extra, str):
                base["description"] = extra
            elif isinstance(extra, dict):
                base.update(extra)
        return base
    if origin is Literal:
        vals = list(args)
        out: dict[str, Any] = {"enum": vals}
        kinds = {_json_type(v) for v in vals}
        if len(kinds) == 1:
            out["type"] = kinds.pop()
        return out
    if origin in (typing.Union, getattr(types, "UnionType", object())):
        non_null = [a for a in args if a is not type(None)]
        nullable = len(non_null) != len(args)
        if len(non_null) == 1:
            inner = type_to_schema(non_null[0])
            return _make_nullable(inner) if nullable and inner else inner
        alts = [type_to_schema(a) for a in non_null] + ([{"type": "null"}] if nullable else [])
        return {"anyOf": alts}
    if isinstance(tp, type) and issubclass(tp, enum.Enum):
        vals = [m.value for m in tp]
        out = {"enum": vals}
        kinds = {_json_type(v) for v in vals}
        if len(kinds) == 1:
            out["type"] = kinds.pop()
        return out
    if tp is bool:
        return {"type": "boolean"}
    if tp is int:
        return {"type": "integer"}
    if tp is float:
        return {"type": "number"}
    if tp is str:
        return {"type": "string"}
    if tp is type(None):
        return {"type": "null"}
    if tp in (list, tuple, set, frozenset) or origin in (list, tuple, set, frozenset, abc.Sequence, abc.Iterable):
        out = {"type": "array"}
        if args and not (origin is tuple and len(args) == 2 and args[1] is Ellipsis):
            if origin is tuple and len(args) > 1:
                out["prefixItems"] = [type_to_schema(a) for a in args]
                out["minItems"] = out["maxItems"] = len(args)
            else:
                out["items"] = type_to_schema(args[0])
        elif args:
            out["items"] = type_to_schema(args[0])
        if origin in (set, frozenset) or tp in (set, frozenset):
            out["uniqueItems"] = True
        return out
    if tp is dict or origin in (dict, abc.Mapping):
        out = {"type": "object"}
        if len(args) == 2 and args[1] is not Any:
            out["additionalProperties"] = type_to_schema(args[1])
        return out
    if isinstance(tp, type) and typing.is_typeddict(tp):
        hints = typing.get_type_hints(tp)
        required = sorted(getattr(tp, "__required_keys__", hints.keys()))
        return {
            "type": "object",
            "properties": {k: type_to_schema(v) for k, v in hints.items()},
            "required": required,
            "additionalProperties": False,
        }
    return {}


def schema_from_function(fn: Callable[..., Any]) -> tuple[str, dict[str, Any], bool]:
    """Derive ``(description, input_schema, wants_ctx)`` from a function signature and docstring.

    * Parameters without defaults are ``required``; defaults that are JSON values become ``default``.
    * A parameter named ``ctx`` (or annotated ``RunContext``) receives the :class:`RunContext` and
      is excluded from the schema.
    * Generated schemas set ``additionalProperties: false`` so they are strict-compatible.
    """
    sig = inspect.signature(fn)
    try:
        hints = typing.get_type_hints(fn, include_extras=True)
    except Exception:  # unresolvable forward refs: fall back to raw annotations
        hints = {}
    description, param_docs = parse_docstring(inspect.getdoc(fn))
    props: dict[str, Any] = {}
    required: list[str] = []
    wants_ctx = False
    for name, p in sig.parameters.items():
        ann = hints.get(name, p.annotation)
        if name == "ctx" or _is_run_context(ann):
            wants_ctx = True
            continue
        if name == "self":
            continue
        if p.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise SchemaError(f"{fn.__name__}: *args/**kwargs cannot be described as a tool schema; declare parameters explicitly")
        if p.kind is inspect.Parameter.POSITIONAL_ONLY:
            raise SchemaError(f"{fn.__name__}: positional-only parameter {name!r} cannot receive a JSON object field")
        prop = type_to_schema(ann)
        if name in param_docs and "description" not in prop:
            prop["description"] = param_docs[name]
        if p.default is inspect.Parameter.empty:
            required.append(name)
        elif p.default is not None:
            try:
                json.dumps(p.default)
                prop["default"] = p.default.value if isinstance(p.default, enum.Enum) else p.default
            except (TypeError, ValueError):
                pass
        props[name] = prop
    schema: dict[str, Any] = {"type": "object", "properties": props, "additionalProperties": False}
    if required:
        schema["required"] = required
    return description, schema, wants_ctx


# ----------------------------------------------------------------------------- per-provider compiler


def _walk_schemas(schema: Any, fn: Callable[[dict], None]) -> None:
    """Call ``fn`` on every sub-schema dict (never on ``properties`` maps themselves)."""
    if isinstance(schema, list):
        for s in schema:
            _walk_schemas(s, fn)
        return
    if not isinstance(schema, dict):
        return
    fn(schema)
    for kw in ("properties", "patternProperties", "$defs", "definitions"):
        for sub in (schema.get(kw) or {}).values():
            _walk_schemas(sub, fn)
    for kw in ("items", "additionalProperties", "not", "if", "then", "else"):
        if isinstance(schema.get(kw), dict):
            _walk_schemas(schema[kw], fn)
    for kw in ("allOf", "anyOf", "oneOf", "prefixItems"):
        if isinstance(schema.get(kw), list):
            _walk_schemas(schema[kw], fn)


def _is_object_schema(s: dict) -> bool:
    t = s.get("type")
    return t == "object" or (isinstance(t, list) and "object" in t) or (t is None and "properties" in s)


def _make_nullable(s: Any) -> Any:
    if not isinstance(s, dict):
        return s
    s = dict(s)
    t = s.get("type")
    if isinstance(t, str):
        if t != "null":
            s["type"] = [t, "null"]
    elif isinstance(t, list):
        if "null" not in t:
            s["type"] = [*t, "null"]
    else:
        return {"anyOf": [s, {"type": "null"}]}
    if isinstance(s.get("enum"), list) and None not in s["enum"]:
        s["enum"] = [*s["enum"], None]
    return s


def compile_schema(schema: Mapping[str, Any], target: Target = "anthropic") -> dict[str, Any]:
    """Compile a canonical schema for one provider. Never mutates the input.

    * ``anthropic``: as authored (``$schema`` dropped).
    * ``anthropic_strict``: every object gets ``additionalProperties: false``; ``required`` kept.
    * ``openai_strict``: additionally every property becomes ``required`` and formerly-optional
      properties become nullable (``["string", "null"]`` / ``anyOf [..., null]``); ``default``
      values are dropped because the model must send the field anyway.
    * ``deepseek_strict``: ``openai_strict`` minus ``minLength``/``maxLength``/``minItems``/``maxItems``.
    """
    out = copy.deepcopy(dict(schema))
    out.pop("$schema", None)
    if target == "anthropic":
        return out

    def strictify(s: dict) -> None:
        if not _is_object_schema(s):
            return
        s["additionalProperties"] = False
        if target in ("openai_strict", "deepseek_strict"):
            props = s.get("properties") or {}
            req = set(s.get("required") or ())
            for name in list(props):
                if name not in req:
                    props[name] = _make_nullable(props[name])
                if isinstance(props[name], dict):
                    props[name].pop("default", None)
            s["properties"] = props
            s["required"] = list(props)

    _walk_schemas(out, strictify)
    if target == "deepseek_strict":

        def strip(s: dict) -> None:
            for kw in DEEPSEEK_STRICT_UNSUPPORTED:
                s.pop(kw, None)

        _walk_schemas(out, strip)
    return out


def lint_schema(schema: Mapping[str, Any], target: Target = "anthropic") -> list[str]:
    """Return human-readable problems ``schema`` has for ``target`` (empty list = clean)."""
    problems: list[str] = []
    try:
        check_schema(dict(schema))
    except SchemaError as exc:
        return [str(exc)]
    if _is_object_schema(dict(schema)) is False:
        problems.append("tool input_schema must be an object schema ({'type': 'object', ...})")

    def visit(s: Any, path: str) -> None:
        if isinstance(s, list):
            for i, x in enumerate(s):
                visit(x, f"{path}/{i}")
            return
        if not isinstance(s, dict):
            return
        unknown = set(s) - VALIDATED_KEYWORDS - ANNOTATION_KEYWORDS
        if unknown:
            problems.append(f"{path}: keyword(s) {sorted(unknown)} are not validated client-side")
        if target != "anthropic" and _is_object_schema(s):
            if s.get("additionalProperties") is not False:
                problems.append(f"{path}: strict mode requires additionalProperties: false")
            if target in ("openai_strict", "deepseek_strict"):
                missing = sorted(set(s.get("properties") or {}) - set(s.get("required") or ()))
                if missing:
                    problems.append(f"{path}: strict mode requires every property in 'required' (missing {missing})")
        if target == "deepseek_strict":
            bad = sorted(set(s) & DEEPSEEK_STRICT_UNSUPPORTED)
            if bad:
                problems.append(f"{path}: DeepSeek strict mode does not support {bad}")
        for kw, sub in s.items():
            if kw in ("properties", "patternProperties", "$defs", "definitions") and isinstance(sub, dict):
                for name, x in sub.items():
                    visit(x, f"{path}/{kw}/{name}")
            elif kw in ("items", "additionalProperties", "not", "if", "then", "else") and isinstance(sub, dict):
                visit(sub, f"{path}/{kw}")
            elif kw in ("allOf", "anyOf", "oneOf", "prefixItems"):
                visit(sub, f"{path}/{kw}")

    visit(dict(schema), "#")
    return problems


__all__ = [
    "ANNOTATION_KEYWORDS",
    "DEEPSEEK_STRICT_UNSUPPORTED",
    "JSON_TYPES",
    "Target",
    "VALIDATED_KEYWORDS",
    "ValidationIssue",
    "apply_defaults",
    "check_schema",
    "compile_schema",
    "example_from_schema",
    "format_issues",
    "is_valid",
    "lint_schema",
    "parse_docstring",
    "schema_from_function",
    "type_to_schema",
    "validate",
]
