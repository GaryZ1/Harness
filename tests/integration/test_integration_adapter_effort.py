"""harness_core.anthropic_adapter: per-request ``extra["effort"]`` maps to ``output_config.effort``."""

from __future__ import annotations

import pytest

from harness_core.anthropic_adapter import AnthropicModelClient
from harness_core.errors import ConfigError
from harness_core.types import Message, ModelRequest


class _NoSDK:
    """Stand-in SDK client (build_params never calls it)."""


def _params(extra: dict, **client_kw):
    client = AnthropicModelClient("test-model", client=_NoSDK(), **client_kw)
    params, betas = client.build_params(ModelRequest(system="s", messages=[Message.user("hi")], extra=extra))
    return params


def test_request_effort_is_mapped_into_output_config_and_not_sent_top_level():
    params = _params({"model": "test-model-worker", "effort": "low"})
    assert params["model"] == "test-model-worker"
    assert params["output_config"] == {"effort": "low"}
    assert "effort" not in params


def test_request_effort_overrides_client_effort_and_merges_with_output_config():
    params = _params({"effort": "max"}, effort="low")
    assert params["output_config"] == {"effort": "max"}
    params = _params({"effort": "high", "output_config": {"format": {"type": "json_schema"}}})
    assert params["output_config"] == {"effort": "high", "format": {"type": "json_schema"}}
    assert "output_config" not in _params({})  # no effort anywhere -> the model default


def test_invalid_request_effort_is_refused():
    with pytest.raises(ConfigError, match="effort"):
        _params({"effort": "extreme"})
