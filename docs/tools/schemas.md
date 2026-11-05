# JSON Schema: `harness_tools.schema`

**Purpose:** validate every tool input before its handler runs, derive schemas from Python signatures, and compile one canonical schema for each provider's strict mode.

## Research basis

* "Validate inputs against schema before running" and "validation failures belong in tool results (`isError`) to enable model self-correction" (MCP SEP-1303).
* JSON Schema 2020-12 is MCP's default dialect (2025-11-25, SEP-1613), and 2026-07-28 allows any 2020-12 keyword with `$ref` (SEP-2106).
* **One schema for all providers is a pitfall** (§5.8). OpenAI strict requires every property in `required`, optional fields as nullable unions, and `additionalProperties: false`. DeepSeek strict rejects `minLength`/`maxLength`/`minItems`/`maxItems`. Anthropic `strict` needs `additionalProperties: false`. Checklist step 7 calls for one canonical schema plus a per-provider compiler and a CI lint.

## API

```python
from harness_tools.schema import validate, check_schema, schema_from_function, compile_schema, lint_schema, example_from_schema

issues = validate({"limit": "5"}, schema)          # [ValidationIssue(path="/user_id", message="is a required property..."), ...]
check_schema(schema)                               # raises SchemaError on malformed schemas (done at registration)

desc, schema, wants_ctx = schema_from_function(fn) # signature + docstring -> schema (additionalProperties: false)

compile_schema(schema, "openai_strict")            # all required, optionals -> ["T", "null"], defaults dropped
compile_schema(schema, "deepseek_strict")          # openai_strict minus length/item bounds
compile_schema(schema, "anthropic_strict")         # additionalProperties: false everywhere, required kept
lint_schema(schema, "deepseek_strict")             # ["#/properties/tags: DeepSeek strict mode does not support ['maxItems']", ...]
example_from_schema(schema)                        # minimal example for "Example of a valid input: ..."
```

### Supported keywords

The validator implements `type` (including type lists), `enum`, `const`, `properties`, `required`, `additionalProperties`, `patternProperties`, `minProperties`/`maxProperties`, `dependentRequired`, `items`, `prefixItems`, `minItems`/`maxItems`, `uniqueItems`, `minLength`/`maxLength`, `pattern`, `minimum`/`maximum`, `exclusiveMinimum`/`exclusiveMaximum`, `multipleOf`, `allOf`/`anyOf`/`oneOf`/`not`, `if`/`then`/`else` and local `$ref` (`#/$defs/...`). It also accepts `format` and the other annotation keywords but does not enforce them, which the spec allows. Unknown keywords are ignored, so third-party MCP schemas always register; `lint_schema` reports them as "not validated client-side".

JSON semantics are respected: `true` is not an integer, `3.0` is, and `1` does not equal `true` in `enum`/`const`.

### Signature mapping

| Python | JSON Schema |
|---|---|
| `str`, `int`, `float`, `bool` | `string`, `integer`, `number`, `boolean` |
| `list[T]`, `tuple[T, ...]`, `set[T]` | `array` with `items` (`set` adds `uniqueItems`) |
| `tuple[A, B]` | `prefixItems` with fixed length |
| `dict[str, T]` | `object` with `additionalProperties: T` |
| `Literal[...]`, `Enum` subclass | `enum` (plus `type` when homogeneous) |
| `T \| None`, `Optional[T]` | `["T", "null"]` (`null` is added to `enum` too) |
| `TypedDict` | closed `object` with `required` keys |
| `Annotated[T, "desc"]` | adds `description` |
| parameter named `ctx` / annotated `RunContext` | excluded; receives the `RunContext` |

Parameter descriptions come from Google-style (`Args:`) or Sphinx-style (`:param x:`) docstrings. Parameters without defaults are `required`. JSON-serialisable defaults become `default`, and the registry fills them in before calling the handler.

## Defaults

| Setting | Default | Anchor |
|---|---|---|
| Max issues reported | 25 | our choice (enough to fix a call, bounded output) |
| `$ref` depth | 64 | our choice (recursion guard) |
| Generated schemas | `additionalProperties: false` | strict-mode compatible on every provider |
| DeepSeek unsupported keywords | `minLength`, `maxLength`, `minItems`, `maxItems` | DeepSeek tool-calls guide (search-summarised) |

## Invariants

* `compile_schema` never mutates its input, and only touches schema positions. A *property* named `maxItems` survives the DeepSeek compile.
* `check_schema` runs at registration, so a malformed schema can never reach a session.
* Validation issues carry JSON-pointer paths (`/tags/1`), so error messages point at the exact field.

## Failure modes

* Remote `$ref` (`https://...`) raises `SchemaError`. Only local references are supported, which matches MCP's composition bounds.
* `format` is not enforced. Use `pattern` when a format really matters (e.g. `^USR-[0-9]+$`), and add `input_examples`.
* Signatures with `*args`, `**kwargs` or positional-only parameters cannot be described as a JSON object and raise `SchemaError`.
