from __future__ import annotations

import enum
from typing import Literal, Optional

import pytest

from harness_core.types import RunContext
from harness_tools.errors import SchemaError
from harness_tools.schema import (
    check_schema,
    compile_schema,
    example_from_schema,
    lint_schema,
    schema_from_function,
    validate,
)

SCHEMA = {
    "type": "object",
    "properties": {
        "user_id": {"type": "string", "pattern": "^USR-[0-9]+$"},
        "limit": {"type": "integer", "minimum": 1, "maximum": 50},
        "mode": {"type": "string", "enum": ["concise", "detailed"]},
        "tags": {"type": "array", "items": {"type": "string", "minLength": 1}, "maxItems": 3, "uniqueItems": True},
        "when": {"$ref": "#/$defs/date"},
    },
    "required": ["user_id"],
    "additionalProperties": False,
    "$defs": {"date": {"type": "string", "pattern": r"^\d{4}-\d{2}-\d{2}$"}},
}


def paths(issues):
    return {i.path for i in issues}


def test_valid_instance_has_no_issues():
    assert validate({"user_id": "USR-1", "limit": 5, "mode": "concise", "tags": ["a"], "when": "2026-01-02"}, SCHEMA) == []


def test_reports_every_issue_with_json_pointer_paths():
    issues = validate({"limit": 0, "mode": "loud", "tags": ["a", "a", "b", ""], "when": "Jan 2", "extra": 1}, SCHEMA)
    assert {"/user_id", "/limit", "/mode", "/tags", "/tags/1", "/tags/3", "/when", "/extra"} <= paths(issues)
    assert any("unexpected property" in i.message and "user_id" in i.message for i in issues)


def test_booleans_are_not_integers_and_integral_floats_are():
    s = {"type": "integer"}
    assert validate(True, s) and not validate(3, s) and not validate(3.0, s) and validate(3.5, s)
    assert validate(1, {"enum": [True]})  # 1 must not equal True


def test_composition_keywords():
    s = {"oneOf": [{"type": "string"}, {"type": "integer", "minimum": 0}], "not": {"const": "forbidden"}}
    assert not validate("x", s) and not validate(3, s)
    assert validate(-1, s) and validate("forbidden", s)
    cond = {"type": "object", "if": {"properties": {"kind": {"const": "file"}}}, "then": {"required": ["path"]}}
    assert validate({"kind": "file"}, cond) and not validate({"kind": "dir"}, cond)


@pytest.mark.parametrize(
    "bad",
    [{"type": "strng"}, {"required": "a"}, {"pattern": "("}, {"properties": []}, {"anyOf": []}, {"$ref": "#/$defs/missing"}, {"minLength": -1}],
)
def test_check_schema_rejects_malformed(bad):
    with pytest.raises(SchemaError):
        check_schema(bad)


def test_check_schema_accepts_unknown_keywords_from_third_parties():
    check_schema({"type": "object", "x-vendor": True, "unevaluatedProperties": False})


class Color(enum.Enum):
    RED = "red"
    BLUE = "blue"


def lookup(user_id: str, limit: int = 10, mode: Literal["concise", "detailed"] = "concise", color: Optional[Color] = None,
           tags: list[str] | None = None, ctx: RunContext | None = None) -> str:
    """Look up a user.

    Args:
        user_id: The user's ID, like USR-123.
        limit: Max rows
            to return.
    """
    return ""


def test_schema_from_function_maps_types_defaults_and_docs():
    desc, schema, wants_ctx = schema_from_function(lookup)
    assert desc == "Look up a user." and wants_ctx
    props = schema["properties"]
    assert schema["required"] == ["user_id"] and "ctx" not in props
    assert props["user_id"] == {"type": "string", "description": "The user's ID, like USR-123."}
    assert props["limit"] == {"type": "integer", "description": "Max rows to return.", "default": 10}
    assert props["mode"]["enum"] == ["concise", "detailed"] and props["mode"]["default"] == "concise"
    assert props["color"] == {"enum": ["red", "blue", None], "type": ["string", "null"]}
    assert validate({"user_id": "u", "color": None}, schema) == [] and validate({"user_id": "u", "color": "green"}, schema)
    assert props["tags"]["type"] == ["array", "null"] and props["tags"]["items"] == {"type": "string"}
    assert schema["additionalProperties"] is False


def test_schema_from_function_rejects_var_kwargs():
    def bad(**kwargs):
        """x"""

    with pytest.raises(SchemaError):
        schema_from_function(bad)


def test_openai_strict_makes_optionals_required_and_nullable_without_mutating():
    s = {"type": "object", "properties": {"q": {"type": "string"}, "n": {"type": "integer", "default": 3}}, "required": ["q"]}
    out = compile_schema(s, "openai_strict")
    assert out["required"] == ["q", "n"] and out["additionalProperties"] is False
    assert out["properties"]["n"] == {"type": ["integer", "null"]}
    assert "additionalProperties" not in s and s["required"] == ["q"]
    assert lint_schema(out, "openai_strict") == []
    assert any("required" in p for p in lint_schema(s, "openai_strict"))


def test_deepseek_strict_strips_length_and_item_bounds_but_not_property_names():
    s = {"type": "object", "properties": {"maxItems": {"type": "array", "items": {"type": "string", "maxLength": 5}, "minItems": 1}}}
    out = compile_schema(s, "deepseek_strict")
    assert "maxItems" in out["properties"]  # a *property* called maxItems survives
    prop = out["properties"]["maxItems"]
    assert "minItems" not in prop and "maxLength" not in prop["items"]
    assert any("DeepSeek" in p for p in lint_schema(s, "deepseek_strict"))


def test_example_from_schema_produces_valid_examples():
    ex = example_from_schema({"type": "object", "properties": {"n": {"type": "integer", "minimum": 2}, "m": {"enum": ["a"]}}, "required": ["n", "m"]})
    assert ex == {"n": 2, "m": "a"}
