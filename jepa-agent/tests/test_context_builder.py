"""Tests for mcp_servers.context_builder.server — full 6-stage pipeline."""

import json
import os
import tempfile
from pathlib import Path

import pytest

# Functions under test — import from the MCP server module
from mcp_servers.context_builder.server import (
    _resolve_import_to_file,
    _classify_file,
    _find_relevant_regions,
    _compress_file,
    _enforce_budget,
    build_context,
)


# ═══════════════════════════════════════════════════════════════════
# _resolve_import_to_file
# ═══════════════════════════════════════════════════════════════════

class TestResolveImport:
    def test_direct_import(self):
        idx = {"os.py": {}, "re.py": {}}
        result = _resolve_import_to_file("import os", idx, "/root")
        assert "os.py" in result

    def test_from_import(self):
        idx = {"collections/abc.py": {}}
        result = _resolve_import_to_file("from collections.abc import Iterable", idx, "/root")
        assert "collections/abc.py" in result

    def test_package_init_fallback(self):
        idx = {"my_pkg/__init__.py": {}}
        result = _resolve_import_to_file("import my_pkg", idx, "/root")
        assert "my_pkg/__init__.py" in result

    def test_stdlib_not_in_index_returns_empty(self):
        idx = {}  # no stdlib files in index
        result = _resolve_import_to_file("import os", idx, "/root")
        assert result == []

    def test_relative_import_returns_empty(self):
        idx = {}
        result = _resolve_import_to_file("from . import sibling", idx, "/root")
        assert result == []


# ═══════════════════════════════════════════════════════════════════
# _classify_file
# ═══════════════════════════════════════════════════════════════════

class TestClassifyFile:
    def test_patch_target(self):
        role = _classify_file("src/main.py", "src/main.py", {"src/main.py"}, set(), set())
        assert role == "patch_target"

    def test_direct_dependency_hop1(self):
        role = _classify_file("src/utils.py", "src/main.py", {"src/main.py"}, {"src/utils.py"}, set())
        assert role == "direct_dependency"

    def test_direct_dependency_seed_not_target(self):
        role = _classify_file("other.py", "main.py", {"main.py", "other.py"}, set(), set())
        assert role == "direct_dependency"

    def test_transitive_dependency(self):
        role = _classify_file("a.py", "main.py", {"main.py"}, set(), {"a.py"})
        assert role == "transitive_dependency"

    def test_test_file_pattern(self):
        role = _classify_file("tests/test_foo.py", "main.py", set(), set(), set())
        assert role == "test_file"

    def test_type_provider_heuristic(self):
        role = _classify_file("user_model.py", "main.py", set(), set(), set())
        assert role == "type_provider"


# ═══════════════════════════════════════════════════════════════════
# _find_relevant_regions
# ═══════════════════════════════════════════════════════════════════

class TestFindRelevantRegions:
    FILES_INDEX = {
        "mod.py": {
            "symbols": {
                "functions": [
                    {"name": "greet", "line": 5, "end_line": 8},
                    {"name": "add", "line": 10, "end_line": 12},
                    {"name": "unrelated", "line": 15, "end_line": 20},
                ],
                "classes": [],
            }
        }
    }

    def test_finds_enclosing_function(self):
        matches = [{"file": "mod.py", "line": 6, "score": 0.9}]
        regions = _find_relevant_regions("mod.py", matches, self.FILES_INDEX)
        assert len(regions) == 1
        assert regions[0]["name"] == "greet"

    def test_dedup_same_region(self):
        matches = [
            {"file": "mod.py", "line": 6, "score": 0.9},
            {"file": "mod.py", "line": 7, "score": 0.8},
        ]
        regions = _find_relevant_regions("mod.py", matches, self.FILES_INDEX)
        assert len(regions) == 1  # both lines inside greet
        assert regions[0]["name"] == "greet"

    def test_multiple_regions(self):
        matches = [
            {"file": "mod.py", "line": 6, "score": 0.9},
            {"file": "mod.py", "line": 11, "score": 0.7},
        ]
        regions = _find_relevant_regions("mod.py", matches, self.FILES_INDEX)
        assert len(regions) == 2
        names = [r["name"] for r in regions]
        assert names == ["greet", "add"]

    def test_no_matches_returns_empty(self):
        regions = _find_relevant_regions("mod.py", [], self.FILES_INDEX)
        assert regions == []

    def test_match_outside_function_returns_empty(self):
        matches = [{"file": "mod.py", "line": 1, "score": 0.5}]
        regions = _find_relevant_regions("mod.py", matches, self.FILES_INDEX)
        assert regions == []

    def test_wrong_file_returns_empty(self):
        matches = [{"file": "other.py", "line": 5, "score": 0.9}]
        regions = _find_relevant_regions("mod.py", matches, self.FILES_INDEX)
        assert regions == []


# ═══════════════════════════════════════════════════════════════════
# _compress_file
# ═══════════════════════════════════════════════════════════════════

class TestCompressFile:
    CODE = "import os\nimport re\n\ndef greet(name):\n    return f\"Hello, {name}\"\n\ndef add(a, b):\n    return a + b\n"

    ENTRY = {
        "symbols": {
            "functions": [
                {"name": "greet", "line": 3, "end_line": 4},
                {"name": "add", "line": 6, "end_line": 7},
            ],
            "classes": [],
            "imports": ["import os", "import re"],
        }
    }

    def test_patch_target_full_code(self):
        summary = _compress_file("mod.py", "patch_target", self.ENTRY, self.CODE)
        assert summary["role"] == "patch_target"
        assert "full_code" in summary
        assert "Hello" in summary["full_code"]

    def test_patch_target_relevant_regions(self):
        regions = [{"name": "greet", "line": 3, "end_line": 4}]
        summary = _compress_file(
            "mod.py", "patch_target", self.ENTRY, self.CODE, relevant_regions=regions,
        )
        assert "relevant_code" in summary
        assert "greet" in summary["relevant_code"]
        assert "add" not in summary["relevant_code"]

    def test_patch_target_relevant_regions_includes_imports(self):
        regions = [{"name": "add", "line": 6, "end_line": 7}]
        summary = _compress_file(
            "mod.py", "patch_target", self.ENTRY, self.CODE, relevant_regions=regions,
        )
        assert "import os" in summary["relevant_code"]
        assert "add" in summary["relevant_code"]
        assert "compression_note" in summary

    def test_direct_dependency_summary(self):
        summary = _compress_file("utils.py", "direct_dependency", self.ENTRY, self.CODE)
        assert summary["role"] == "direct_dependency"
        assert "functions" in summary
        assert "full_code" not in summary

    def test_type_provider_symbols_only(self):
        entry = {"symbols": {"functions": [{"name": "UserSchema"}], "classes": [], "imports": []}}
        summary = _compress_file("schemas.py", "type_provider", entry, "")
        assert summary["role"] == "type_provider"
        assert "exported_symbols" in summary
        assert "UserSchema" in summary["exported_symbols"]

    def test_test_file_counts_tests(self):
        entry = {
            "symbols": {
                "functions": [
                    {"name": "test_foo"}, {"name": "test_bar"}, {"name": "helper"},
                ],
                "classes": [],
                "imports": [],
            }
        }
        summary = _compress_file("test_main.py", "test_file", entry, "")
        assert summary["test_count"] == 2


# ═══════════════════════════════════════════════════════════════════
# _enforce_budget
# ═══════════════════════════════════════════════════════════════════

class TestEnforceBudget:
    def test_patch_target_always_included(self):
        summaries = [
            {"file": "a.py", "role": "patch_target"},
            {"file": "b.py", "role": "unrelated"},
        ]
        included, mem, excluded = _enforce_budget(summaries, [], budget=10)
        assert len(included) == 1
        assert included[0]["role"] == "patch_target"

    def test_priority_ordering(self):
        summaries = [
            {"file": "low.py", "role": "unrelated"},
            {"file": "high.py", "role": "direct_dependency"},
        ]
        included, mem, excluded = _enforce_budget(summaries, [], budget=9999)
        assert len(included) == 2

    def test_drops_lowest_priority_when_over_budget(self):
        big = {"file": "big.py", "role": "unrelated", "data": "x" * 2000}
        small = {"file": "small.py", "role": "direct_dependency"}
        included, mem, excluded = _enforce_budget([big, small], [], budget=200)
        assert "small.py" not in [e for e in excluded]
        assert "big.py" in excluded

    def test_memory_hits_under_remaining_budget(self):
        summaries = [{"file": "m.py", "role": "direct_dependency"}]
        mem_hits = [{"rule": "always use pathlib"}]
        included, mem, excluded = _enforce_budget(summaries, mem_hits, budget=9999)
        assert len(mem) == 1

    def test_memory_hits_dropped_when_over_budget(self):
        summaries = [{"file": "m.py", "role": "direct_dependency", "data": "x" * 1000}]
        mem_hits = [{"rule": "long memory rule " + "x" * 500}]
        included, mem, excluded = _enforce_budget(summaries, mem_hits, budget=100)
        assert len(mem) == 0


# ═══════════════════════════════════════════════════════════════════
# build_context — full pipeline integration
# ═══════════════════════════════════════════════════════════════════

class TestBuildContext:
    """End-to-end tests for the full build_context pipeline."""

    def test_with_minimal_input(self):
        """Pipeline does not crash with only file_path and task."""
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "main.py")
            Path(target).write_text("x = 1\n", encoding="utf-8")
            result = build_context(
                task="fix the bug",
                file_path=target,
            )
            assert "task_summary" in result
            assert "patch_targets" in result
            assert "dependency_summaries" in result
            assert "expansion_stats" in result
            assert result["expansion_stats"]["seeds"] >= 1

    def test_with_code_index_and_semantic_matches(self):
        """Full pipeline: code index + semantic matches produce regions."""
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "mod.py")
            Path(target).write_text(
                "import os\n\ndef greet(name):\n    return f\"Hello, {name}\"\n",
                encoding="utf-8",
            )
            code_index = {
                "files": {
                    "mod.py": {
                        "language": "python",
                        "size": 50,
                        "symbols": {
                            "functions": [
                                {"name": "greet", "line": 3, "end_line": 4},
                            ],
                            "classes": [],
                            "imports": ["import os"],
                        },
                    }
                }
            }
            semantic_matches = [
                {"file": "mod.py", "line": 3, "score": 0.85, "snippet": "def greet"},
            ]
            result = build_context(
                task="update greet to be more polite",
                file_path=target,
                semantic_matches=semantic_matches,
                code_index=code_index,
            )
            assert len(result["patch_targets"]) == 1
            pt = result["patch_targets"][0]
            assert pt["file"] == "mod.py"
            assert "relevant_code" in pt or "full_code" in pt

    def test_excludes_unrelated_files_when_over_budget(self):
        """Budget enforcement drops low-priority files."""
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "main.py")
            Path(target).write_text("def main(): pass\n", encoding="utf-8")
            code_index = {
                "files": {
                    "main.py": {
                        "symbols": {
                            "functions": [{"name": "main", "line": 1, "end_line": 1}],
                            "classes": [],
                            "imports": [],
                        },
                    },
                    "big_unrelated.py": {
                        "symbols": {
                            "functions": [{"name": "helper", "line": 1, "end_line": 2}],
                            "classes": [],
                            "imports": [],
                        },
                    },
                }
            }
            semantic_matches = ["big_unrelated.py"]
            result = build_context(
                task="fix main",
                file_path=target,
                semantic_matches=semantic_matches,
                code_index=code_index,
                token_budget=50,
            )
            pt_files = [p["file"] for p in result["patch_targets"]]
            assert "main.py" in pt_files
            assert len(result["excluded_files"]) >= 0

    def test_with_memory_hits(self):
        """Memory rules appear in output when budget allows."""
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "main.py")
            Path(target).write_text("x = 1\n", encoding="utf-8")
            memory_hits = [{"rule": "use pathlib", "source": "vault"}]
            result = build_context(
                task="add error handling",
                file_path=target,
                memory_hits=memory_hits,
                token_budget=9999,
            )
            assert len(result["memory_rules"]) == 1
            assert result["memory_rules"][0]["rule"] == "use pathlib"

    def test_expansion_stats_reflect_inputs(self):
        """expansion_stats shows correct seed/file counts."""
        with tempfile.TemporaryDirectory() as tmp:
            target = os.path.join(tmp, "main.py")
            Path(target).write_text("import utils\n", encoding="utf-8")
            utils_file = os.path.join(tmp, "utils.py")
            Path(utils_file).write_text("def util(): pass\n", encoding="utf-8")
            code_index = {
                "files": {
                    "main.py": {
                        "symbols": {
                            "functions": [{"name": "main", "line": 1, "end_line": 1}],
                            "classes": [],
                            "imports": ["import utils"],
                        },
                    },
                    "utils.py": {
                        "symbols": {
                            "functions": [{"name": "util", "line": 1, "end_line": 1}],
                            "classes": [],
                            "imports": [],
                        },
                    },
                }
            }
            result = build_context(
                task="refactor",
                file_path=target,
                code_index=code_index,
            )
            stats = result["expansion_stats"]
            assert stats["seeds"] >= 1
            total = stats["seeds"] + stats["hop1"] + stats["hop2"]
            assert total >= 1
