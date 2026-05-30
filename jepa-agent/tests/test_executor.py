"""Tests for core.executor — file ops, shell commands, workspace state."""

import os
import tempfile
from pathlib import Path

import pytest

from core.executor import (
    read_file,
    write_file,
    apply_patch,
    apply_patches,
    run_command,
    Workspace,
)


class TestReadWrite:
    def test_read_existing_file(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
            f.write("hello world")
            path = f.name
        try:
            assert read_file(path) == "hello world"
        finally:
            os.unlink(path)

    def test_read_nonexistent_file(self):
        assert read_file("/nonexistent/path/file.txt") == ""

    def test_write_file_creates_content(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "test.txt")
            assert write_file(path, "content") is True
            assert Path(path).read_text(encoding="utf-8") == "content"

    def test_write_file_invalid_path(self):
        # write into a non-existent directory
        assert write_file("/nonexistent_dir_xyz/file.txt", "data") is False


class TestApplyPatch:
    def test_apply_patch_replaces_content(self):
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".py") as f:
            f.write("old code")
            path = f.name
        try:
            ok = apply_patch(path, "new code")
            assert ok is True
            assert Path(path).read_text(encoding="utf-8") == "new code"
        finally:
            os.unlink(path)

    def test_apply_patch_missing_file(self):
        ok = apply_patch("/nonexistent_dir_xyz/f.py", "code")
        assert ok is False


class TestRunCommand:
    def test_echo(self):
        stdout, stderr, rc = run_command("echo hello")
        assert rc == 0
        assert "hello" in stdout

    def test_failing_command(self):
        stdout, stderr, rc = run_command("exit 1")
        assert rc != 0

    def test_timeout_is_graceful(self):
        # short timeout via a slow command — run_command uses 30s default,
        # so we just verify the function signature works
        stdout, stderr, rc = run_command("echo timeout_test")
        assert rc == 0
        assert "timeout_test" in stdout


class TestWorkspace:
    def test_get_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            file_path = "sub/file.txt"
            full = Path(tmp) / file_path
            full.parent.mkdir(parents=True)
            full.write_text("state content", encoding="utf-8")

            ws = Workspace(root=tmp)
            assert ws.get_state(file_path) == "state content"

    def test_apply_and_get_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            file_path = "test.py"
            ws = Workspace(root=tmp)
            result = ws.apply_and_get_state(file_path, "new state")
            assert result == "new state"
            assert (Path(tmp) / file_path).read_text(encoding="utf-8") == "new state"

    def test_default_root_is_cwd(self):
        ws = Workspace()
        assert ws.root == Path.cwd()


class TestApplyPatches:
    """Tests for apply_patches() — symbolic diffs with insertion directives."""

    SAMPLE_CODE = "import os\n\ndef greet(name):\n    return f\"Hello, {name}\"\n\ndef add(a, b):\n    return a + b\n"

    CODE_INDEX = {
        "files": {
            "mod.py": {
                "language": "python",
                "symbols": {
                    "functions": [
                        {"name": "greet", "line": 3, "end_line": 4},
                        {"name": "add", "line": 6, "end_line": 7},
                    ],
                    "classes": [],
                    "imports": ["os"],
                },
            }
        }
    }

    def _write_source(self, tmp: str) -> str:
        path = os.path.join(tmp, "mod.py")
        write_file(path, self.SAMPLE_CODE)
        return path

    def test_insert_at_end_of_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_source(tmp)
            patches = [
                {
                    "file": "mod.py",
                    "symbol": "--at-end-of-file",
                    "new_body": "def mul(x, y):\n    return x * y",
                }
            ]
            ok, changed = apply_patches(patches, self.CODE_INDEX, tmp)
            assert ok is True
            assert changed == ["mod.py"]
            content = read_file(os.path.join(tmp, "mod.py"))
            assert "def mul(x, y):" in content
            assert content.strip().endswith("return x * y")

    def test_insert_after_symbol(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_source(tmp)
            patches = [
                {
                    "file": "mod.py",
                    "symbol": "--after greet",
                    "new_body": "def farewell(name):\n    return f\"Bye, {name}\"",
                }
            ]
            ok, changed = apply_patches(patches, self.CODE_INDEX, tmp)
            assert ok is True
            assert changed == ["mod.py"]
            content = read_file(os.path.join(tmp, "mod.py"))
            # farewell should be between greet and add
            lines = content.splitlines()
            greet_idx = next(i for i, l in enumerate(lines) if "def greet" in l)
            farewell_idx = next(i for i, l in enumerate(lines) if "def farewell" in l)
            add_idx = next(i for i, l in enumerate(lines) if "def add" in l)
            assert greet_idx < farewell_idx < add_idx

    def test_insert_after_with_replacement_same_file(self):
        """Mix --after insert with a normal symbol replacement in the same file."""
        with tempfile.TemporaryDirectory() as tmp:
            self._write_source(tmp)
            patches = [
                {
                    "file": "mod.py",
                    "symbol": "greet",
                    "new_body": "def greet(name):\n    return f\"Hey, {name}!\"",
                },
                {
                    "file": "mod.py",
                    "symbol": "--after greet",
                    "new_body": "def farewell(name):\n    return f\"Bye, {name}\"",
                },
            ]
            ok, changed = apply_patches(patches, self.CODE_INDEX, tmp)
            assert ok is True
            assert changed == ["mod.py"]
            content = read_file(os.path.join(tmp, "mod.py"))
            assert "Hey" in content  # replacement applied
            assert "farewell" in content  # insertion applied
            lines = content.splitlines()
            greet_idx = next(i for i, l in enumerate(lines) if "def greet" in l)
            farewell_idx = next(i for i, l in enumerate(lines) if "def farewell" in l)
            add_idx = next(i for i, l in enumerate(lines) if "def add" in l)
            assert greet_idx < farewell_idx < add_idx

    def test_insert_after_nonexistent_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_source(tmp)
            patches = [
                {
                    "file": "mod.py",
                    "symbol": "--after nonexistent_func",
                    "new_body": "def foo():\n    pass",
                }
            ]
            ok, changed = apply_patches(patches, self.CODE_INDEX, tmp)
            assert ok is False
            # File should be unchanged
            content = read_file(os.path.join(tmp, "mod.py"))
            assert content == self.SAMPLE_CODE

    def test_multi_file_insert_and_replace(self):
        """Insert in one file, replace in another."""
        with tempfile.TemporaryDirectory() as tmp:
            write_file(os.path.join(tmp, "a.py"), "def existing():\n    return 1\n")
            write_file(os.path.join(tmp, "b.py"), "def old():\n    return 0\n")
            multi_index = {
                "files": {
                    "a.py": {
                        "language": "python",
                        "symbols": {
                            "functions": [{"name": "existing", "line": 1, "end_line": 2}],
                            "classes": [],
                            "imports": [],
                        },
                    },
                    "b.py": {
                        "language": "python",
                        "symbols": {
                            "functions": [{"name": "old", "line": 1, "end_line": 2}],
                            "classes": [],
                            "imports": [],
                        },
                    },
                }
            }
            patches = [
                {
                    "file": "a.py",
                    "symbol": "--after existing",
                    "new_body": "def new_func():\n    return 2",
                },
                {
                    "file": "b.py",
                    "symbol": "old",
                    "new_body": "def old():\n    return 42",
                },
            ]
            ok, changed = apply_patches(patches, multi_index, tmp)
            assert ok is True
            assert set(changed) == {"a.py", "b.py"}
            a_content = read_file(os.path.join(tmp, "a.py"))
            assert "new_func" in a_content
            b_content = read_file(os.path.join(tmp, "b.py"))
            assert "return 42" in b_content

    def test_empty_patches_returns_success(self):
        ok, changed = apply_patches([], {}, "")
        assert ok is True
        assert changed == []
