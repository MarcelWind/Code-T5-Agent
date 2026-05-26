"""MCP Server: obsidian_brain — vault memory read/write.

Provides persistent knowledge storage (the "vault") for the JEPA agent.
Supports reading, writing, searching, and listing vault files organized into
rules/, decisions/, lessons/, and patterns/ directories.
"""

import os
import re
from pathlib import Path
from typing import Optional

from mcp.server import FastMCP

server = FastMCP(
    "obsidian_brain",
    instructions="Persistent memory vault. Read, write, search knowledge files.",
)

VAULT_PATH = Path(os.environ.get("VAULT_PATH", "vault")).resolve()


def _ensure_vault():
    """Create vault directory structure if missing."""
    for subdir in ["rules", "decisions", "lessons", "patterns"]:
        (VAULT_PATH / subdir).mkdir(parents=True, exist_ok=True)


def _resolve_path(rel_path: str) -> Path:
    """Resolve a relative path within the vault, preventing path traversal."""
    full = (VAULT_PATH / rel_path).resolve()
    if not str(full).startswith(str(VAULT_PATH.resolve())):
        raise ValueError(f"Path traversal detected: {rel_path}")
    return full


def _list_markdown_files(directory: Path) -> list[dict]:
    """List all .md files in a directory with metadata."""
    files = []
    for fpath in sorted(directory.rglob("*.md")):
        rel = fpath.relative_to(VAULT_PATH)
        try:
            content = fpath.read_text(encoding="utf-8")
            # Extract first heading as title
            title_match = re.search(r"^# (.+)$", content, re.MULTILINE)
            files.append({
                "path": str(rel),
                "title": title_match.group(1) if title_match else fpath.stem,
                "size": len(content),
                "lines": content.count("\n") + 1,
            })
        except OSError:
            continue
    return files


@server.tool()
def read_vault(path: str) -> dict:
    """Read a file from the vault.

    Args:
        path: Relative path within vault (e.g., 'rules/architecture-rules.md').

    Returns:
        dict with 'path', 'content', and 'metadata'.
    """
    _ensure_vault()
    target = _resolve_path(path)

    if not target.exists():
        return {"error": f"File not found: {path}", "path": path}

    if not target.is_file():
        return {"error": f"Not a file: {path}", "path": path}

    content = target.read_text(encoding="utf-8")
    return {
        "path": str(target.relative_to(VAULT_PATH)),
        "content": content,
        "metadata": {
            "size": len(content),
            "lines": content.count("\n") + 1,
            "modified": os.path.getmtime(str(target)),
        },
    }


@server.tool()
def write_vault(path: str, content: str, overwrite: bool = False) -> dict:
    """Write content to a vault file.

    Args:
        path: Relative path within vault (e.g., 'lessons/learned.md').
        content: Markdown content to write.
        overwrite: If False (default), raises error if file exists.

    Returns:
        dict with 'path', 'status', and 'size'.
    """
    _ensure_vault()
    target = _resolve_path(path)

    if target.exists() and not overwrite:
        return {"error": f"File already exists: {path}. Use overwrite=True to replace.", "path": path}

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")

    # Auto-create section index if needed
    _update_section_index(target.parent)

    return {
        "path": str(target.relative_to(VAULT_PATH)),
        "status": "overwritten" if overwrite and target.exists() else "created",
        "size": len(content),
    }


@server.tool()
def list_vault(section: str = None) -> list[dict]:
    """List vault structure, optionally filtered by section.

    Args:
        section: Optional filter: 'rules', 'decisions', 'lessons', 'patterns', or None for all.

    Returns:
        List of file dicts with path, title, and metadata.
    """
    _ensure_vault()

    if section:
        section_dir = VAULT_PATH / section
        if not section_dir.exists():
            return [{"error": f"Section not found: {section}", "path": str(section_dir)}]
        return _list_markdown_files(section_dir)

    all_files = []
    for subdir in ["rules", "decisions", "lessons", "patterns"]:
        all_files.extend(_list_markdown_files(VAULT_PATH / subdir))
    return all_files


@server.tool()
def search_vault(query: str, section: str = None) -> list[dict]:
    """Full-text search across vault files.

    Args:
        query: Text to search for (case-insensitive).
        section: Optional section filter ('rules', 'decisions', 'lessons', 'patterns').

    Returns:
        List of matching file dicts with path, matched lines, and context.
    """
    _ensure_vault()
    results = []

    search_dir = VAULT_PATH / section if section else VAULT_PATH
    if not search_dir.exists():
        return [{"error": f"Section not found: {section}"}]

    query_lower = query.lower()

    for fpath in sorted(search_dir.rglob("*.md")):
        try:
            content = fpath.read_text(encoding="utf-8")
            lines = content.split("\n")
            matches = []
            for i, line in enumerate(lines):
                if query_lower in line.lower():
                    context_start = max(0, i - 1)
                    context_end = min(len(lines), i + 2)
                    matches.append({
                        "line": i + 1,
                        "context": "\n".join(lines[context_start:context_end]),
                    })

            if matches:
                rel = fpath.relative_to(VAULT_PATH)
                results.append({
                    "path": str(rel),
                    "matches": len(matches),
                    "lines": matches[:10],  # cap at 10 matches per file
                })
        except OSError:
            continue

    return results


def _update_section_index(directory: Path):
    """Auto-update or create an index file for a vault section."""
    index_path = directory / "_index.md"
    if index_path.exists():
        return  # don't overwrite manual index

    files = _list_markdown_files(directory)
    if not files:
        return

    section_name = directory.name.capitalize()
    lines = [f"# {section_name} Index\n"]
    for f in files:
        if f["path"].endswith("_index.md"):
            continue
        lines.append(f"- [[{f['path']}]] — {f['title']}")

    index_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    server.run(transport="stdio")
