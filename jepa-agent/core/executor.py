"""Tool execution — file ops, shell commands, state tracking."""

import subprocess
import tempfile
import os
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


def apply_patches(
    patches: list[dict],
    code_index: dict,
    project_root: str = "",
) -> tuple[bool, list[str]]:
    """Apply symbolic diffs across multiple files using code-index line ranges.

    Each patch dict must have:
      - 'file':  relative path (e.g. 'agent.py')
      - 'symbol': function or class name to replace
      - 'new_body': complete replacement body (including def/class header)

    Patches targeting the same file are applied bottom-up so earlier
    line offsets stay valid.  Falls back to full-file write if a patch
    uses 'full_code' instead of 'symbol'.

    Args:
        patches: List of patch dicts.
        code_index: Full code index dict for symbol resolution.
        project_root: Absolute path to project root (prepended to file paths).

    Returns:
        (success, list_of_changed_relative_paths).
    """
    if not patches:
        return True, []

    # Import here to avoid circular dependency at module level
    from core.code_index import resolve_symbol

    changed: list[str] = []

    # Group patches by file
    by_file: dict[str, list[dict]] = {}
    for p in patches:
        by_file.setdefault(p["file"], []).append(p)

    for rel_path, file_patches in by_file.items():
        full_path = os.path.join(project_root, rel_path) if project_root else rel_path

        # ── Check if any patch uses full_code (old format, do full replace) ──
        full_code_patches = [p for p in file_patches if "full_code" in p]
        if full_code_patches:
            # Last full_code patch wins (convention matches old behavior)
            write_file(full_path, full_code_patches[-1]["full_code"])
            changed.append(rel_path)
            continue

        # ── Symbol-based line-range patches ──
        lines = read_file(full_path)
        if not lines:
            return False, changed
        lines_list = lines.splitlines(keepends=True)
        file_len = len(lines_list)

        resolved: list[tuple[int, int, str]] = []
        for p in file_patches:
            sym = p["symbol"]

            # ── Insertion directives ──
            if sym == "--at-end-of-file":
                # Append at end: start = len+1, end = len → lines_list[len:len]
                resolved.append((file_len + 1, file_len, p["new_body"]))
                continue

            if sym.startswith("--after "):
                target = sym[len("--after "):]
                try:
                    loc = resolve_symbol(code_index, rel_path, target)
                except KeyError:
                    return False, changed
                # Insert right after target's end_line
                resolved.append((loc["end_line"] + 1, loc["end_line"], p["new_body"]))
                continue

            # ── Normal symbol replacement ──
            try:
                loc = resolve_symbol(code_index, rel_path, sym)
            except KeyError:
                return False, changed
            resolved.append((loc["line"], loc["end_line"], p["new_body"]))

        # Sort bottom-up so line numbers of earlier patches stay valid
        resolved.sort(key=lambda x: x[0], reverse=True)

        for start_line, end_line, new_body in resolved:
            # start_line/end_line are 1-based inclusive; Python slice is 0-based exclusive
            lines_list[start_line - 1 : end_line] = [new_body.rstrip("\n") + "\n"]

        write_file(full_path, "".join(lines_list))
        changed.append(rel_path)

    return True, changed
