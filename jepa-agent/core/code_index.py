"""Persistent code symbol index — tree-sitter AST cache for fast diffing.

Stores condensed symbol manifests in vault/code-index/manifest.json.
Each entry tracks mtime so re-parsing only happens when a file changes.
"""

import json
import os
from pathlib import Path
from typing import Optional, Callable


def _get_index_dir() -> Path:
    """Derive code-index path (vault/code-index/ relative to project)."""
    vault = os.environ.get(
        "VAULT_PATH",
        str(Path(__file__).resolve().parent.parent / "vault"),
    )
    idx_dir = Path(vault) / "code-index"
    idx_dir.mkdir(parents=True, exist_ok=True)
    return idx_dir


def get_index() -> dict:
    """Load the code index manifest. Returns {'files': {...}}."""
    path = _get_index_dir() / "manifest.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {"files": {}}
    return {"files": {}}


def save_index(index: dict) -> None:
    """Save the code index manifest."""
    path = _get_index_dir() / "manifest.json"
    path.write_text(json.dumps(index, indent=2, sort_keys=True), encoding="utf-8")


def needs_reparse(file_path: str, entry: Optional[dict]) -> bool:
    """Check if a file has been modified since its last index entry."""
    if entry is None:
        return True
    try:
        return os.path.getmtime(file_path) != entry.get("mtime", 0)
    except OSError:
        return True


def extract_from_ast(ast_data: dict, source_code: str = "") -> dict:
    """Convert tree-sitter AST output to a condensed symbol dict.

    Uses AST line numbers to read symbol names directly from source
    (handles truncated text from overview-depth AST output).

    Args:
        ast_data: Dict returned by code_understanding.parse_code().
        source_code: Raw source of the parsed file (for name extraction).

    Returns:
        {"functions": [...], "classes": [...], "imports": [...]}
    """
    symbols: dict = {"functions": [], "classes": [], "imports": []}
    source_lines = source_code.split("\n") if source_code else []

    for node in ast_data.get("ast", []):
        node_type = node.get("type", "")
        text = node.get("text", "")
        start = node.get("start", (0, 0))
        end = node.get("end", (0, 0))
        line_no = start[0]

        if node_type in ("function_definition", "function_declaration",
                         "method_definition", "function_item", "arrow_function"):
            name = _extract_name(text, ["def ", "fn ", "func "])
            # Fallback: read name from source line
            if not name and source_lines and line_no < len(source_lines):
                name = _extract_name(source_lines[line_no], ["def ", "fn ", "func "])
            symbols["functions"].append({
                "name": name or f"<line {line_no}>",
                "line": line_no,
                "end_line": end[0],
            })
        elif node_type in ("class_definition", "class_declaration",
                           "struct_item", "class_specifier"):
            name = _extract_class_name(text)
            if not name and source_lines and line_no < len(source_lines):
                name = _extract_class_name(source_lines[line_no])
            symbols["classes"].append({
                "name": name or f"<line {line_no}>",
                "line": line_no,
            })
        elif node_type in ("import_statement", "import_from_statement",
                           "import_declaration", "require_statement"):
            if source_lines and line_no < len(source_lines):
                symbols["imports"].append(source_lines[line_no].strip())
            else:
                symbols["imports"].append(text[:120])

    return symbols


def _extract_name(text: str, prefixes: list[str]) -> str:
    """Extract symbol name after one of the given prefixes."""
    for p in prefixes:
        if p in text:
            after = text.split(p, 1)[1].strip()
            return after.split("(")[0].split(":")[0].split()[0].strip()
    return text.split("(")[0].split()[-1] if "(" in text else text[:40]


def _extract_class_name(text: str) -> str:
    """Extract class name from definition line."""
    for keyword in ("class ", "struct ", "type "):
        if keyword in text:
            after = text.split(keyword, 1)[1].strip()
            return after.split("(")[0].split(":")[0].split()[0].strip()
    return text.split(":")[0].split()[-1] if ":" in text else text[:40]


def compute_diff(old_syms: dict, new_syms: dict) -> dict:
    """Compare two symbol dicts and return added/removed/changed.

    Returns:
        {"added": [...], "removed": [...], "changed": [...]}
    """
    diff: dict = {"added": [], "removed": [], "changed": []}

    # Functions
    old_f = {f["name"]: f for f in old_syms.get("functions", []) if f.get("name")}
    new_f = {f["name"]: f for f in new_syms.get("functions", []) if f.get("name")}
    _diff_symbols(diff, old_f, new_f, "function")

    # Classes
    old_c = {c["name"]: c for c in old_syms.get("classes", []) if c.get("name")}
    new_c = {c["name"]: c for c in new_syms.get("classes", []) if c.get("name")}
    _diff_symbols(diff, old_c, new_c, "class")

    return diff


def _diff_symbols(diff: dict, old: dict, new: dict, kind: str) -> None:
    """Mutate diff with added/removed/changed entries for one symbol kind."""
    for name, sym in new.items():
        if name not in old:
            diff["added"].append({"type": kind, "name": name, "line": sym["line"]})
        elif sym["line"] != old[name]["line"] or sym.get("end_line") != old[name].get("end_line"):
            diff["changed"].append({
                "type": kind, "name": name,
                "old_line": old[name]["line"], "new_line": sym["line"],
            })
    for name in old:
        if name not in new:
            diff["removed"].append({"type": kind, "name": name, "was_line": old[name]["line"]})


def format_diff_markdown(diff: dict, rel_path: str) -> str:
    """Format a symbol diff as a short markdown snippet."""
    if not any(diff.values()):
        return ""
    lines = [f"**AST diff for `{rel_path}`:**"]
    for item in diff["added"]:
        lines.append(f"  + {item['type']} `{item['name']}` at line {item['line']}")
    for item in diff["removed"]:
        lines.append(f"  - {item['type']} `{item['name']}` (was line {item['was_line']})")
    for item in diff["changed"]:
        lines.append(f"  ~ {item['type']} `{item['name']}` moved {item['old_line']}→{item['new_line']}")
    return "\n".join(lines)


def format_symbol_summary(entry: dict) -> str:
    """Format a file's symbol entry as a short summary string."""
    syms = entry.get("symbols", {})
    funcs = syms.get("functions", [])
    classes = syms.get("classes", [])
    imports = syms.get("imports", [])
    parts = []
    if funcs:
        parts.append(f"{len(funcs)} function(s): {', '.join(f['name'] for f in funcs[:8])}")
    if classes:
        parts.append(f"{len(classes)} class(es): {', '.join(c['name'] for c in classes[:4])}")
    if imports:
        parts.append(f"{len(imports)} import(s)")
    return " | ".join(parts)
