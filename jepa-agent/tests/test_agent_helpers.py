"""Tests for agent helper functions — _parse_tool_result, SERVER_DEFS."""

import json

import pytest

from agent import _parse_tool_result, SERVER_DEFS


# ── _parse_tool_result ──


class MockTextContent:
    """Minimal mock matching mcp.types.TextContent interface."""

    def __init__(self, text: str):
        self.text = text
        self.type = "text"


class MockCallToolResult:
    """Minimal mock matching mcp.types.CallToolResult interface."""

    def __init__(self, content: list | None = None, is_error: bool = False):
        self.content = content or []
        self.isError = is_error
        self.structuredContent = None


class TestParseToolResult:
    def test_parses_json(self):
        data = {"key": "value", "nested": [1, 2, 3]}
        result = MockCallToolResult(content=[MockTextContent(json.dumps(data))])
        parsed = _parse_tool_result(result)
        assert parsed == data

    def test_returns_raw_text_on_invalid_json(self):
        result = MockCallToolResult(content=[MockTextContent("hello world")])
        parsed = _parse_tool_result(result)
        assert parsed == "hello world"

    def test_returns_empty_dict_on_no_content(self):
        result = MockCallToolResult(content=[])
        parsed = _parse_tool_result(result)
        assert parsed == {}

    def test_merges_multiple_content_blocks(self):
        result = MockCallToolResult(content=[
            MockTextContent('{"a": 1}'),
            MockTextContent('{"b": 2}'),
        ])
        # Multiple blocks joined = not valid JSON, falls back to raw text
        parsed = _parse_tool_result(result)
        assert isinstance(parsed, str)
        assert '{"a": 1}' in parsed
        assert '{"b": 2}' in parsed

    def test_error_result(self):
        result = MockCallToolResult(
            content=[MockTextContent("Something went wrong")],
            is_error=True,
        )
        parsed = _parse_tool_result(result)
        assert parsed == {"_error": "Something went wrong"}

    def test_error_without_content(self):
        result = MockCallToolResult(content=[], is_error=True)
        parsed = _parse_tool_result(result)
        assert parsed == {"_error": "Unknown error"}


# ── SERVER_DEFS ──


class TestServerDefs:
    def test_all_6_servers_present(self):
        expected = {
            "local_router",
            "code_understanding",
            "semantic_search",
            "obsidian_brain",
            "cloud_execution",
            "validators",
        }
        assert set(SERVER_DEFS.keys()) == expected

    def test_each_server_has_required_keys(self):
        for name, cfg in SERVER_DEFS.items():
            assert "module" in cfg, f"{name} missing 'module'"
            assert "args" in cfg, f"{name} missing 'args'"
            assert "env" in cfg, f"{name} missing 'env'"
            assert isinstance(cfg["module"], str)
            assert cfg["module"].startswith("mcp_servers.")

    def test_each_server_has_pythonpath_in_env(self):
        for name, cfg in SERVER_DEFS.items():
            assert "PYTHONPATH" in cfg["env"], f"{name} missing PYTHONPATH in env"
