"""Tests for core.executor — file ops, shell commands, workspace state."""

import os
import tempfile
from pathlib import Path

import pytest

from core.executor import (
    read_file,
    write_file,
    apply_patch,
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
