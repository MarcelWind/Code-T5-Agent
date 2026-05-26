"""Tool execution — file ops, shell commands, state tracking."""

import subprocess
import tempfile
from pathlib import Path


def read_file(path: str) -> str:
    """Read file content. Returns empty string on error."""
    p = Path(path)
    if not p.exists():
        return ""
    return p.read_text(encoding="utf-8")


def write_file(path: str, content: str) -> bool:
    """Write content to file. Returns True on success."""
    try:
        Path(path).write_text(content, encoding="utf-8")
        return True
    except OSError:
        return False


def apply_patch(path: str, new_content: str) -> bool:
    """Replace file content entirely. Returns True on success."""
    return write_file(path, new_content)


def run_command(cmd: str, cwd: str | None = None) -> tuple[str, str, int]:
    """Run shell command. Returns (stdout, stderr, exit_code)."""
    try:
        r = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            cwd=cwd,
            timeout=30,
        )
        return r.stdout, r.stderr, r.returncode
    except subprocess.TimeoutExpired:
        return "", "Command timed out", -1
    except Exception as e:
        return "", str(e), -1


class Workspace:
    """Tracks current workspace state for re-encoding."""

    def __init__(self, root: str | None = None):
        self.root = Path(root) if root else Path.cwd()

    def get_state(self, file_path: str) -> str:
        """Get current file content (for re-encoding)."""
        full = self.root / file_path
        return read_file(str(full))

    def apply_and_get_state(self, file_path: str, new_content: str) -> str:
        """Apply change and return resulting state."""
        full = self.root / file_path
        write_file(str(full), new_content)
        return new_content
