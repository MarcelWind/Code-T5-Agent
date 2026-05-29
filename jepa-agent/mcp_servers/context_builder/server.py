"""MCP Server: context_builder — execution-neighborhood construction.

Sits between semantic search and PLAN. Accepts raw inputs (task, semantic
matches, code index, memory hits) and produces a minimal compressed context
package — replacing ad-hoc AST summaries with structured, budget-enforced
context for the LLM.

Steps performed:
  1. Seed from semantic matches (or fallback to file imports)
  2. AST dependency expansion (1–2 hops through imports)
  3. File role classification (patch_target / direct_dependency / type_provider / ...)
  4. Context compression (summaries over raw code)
  5. Token budget enforcement (drop lowest-priority items)
  6. Memory rule inclusion (from vault hits)
"""

import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

from mcp.server import FastMCP

server = FastMCP(
    "context_builder",
    instructions="Build minimal compressed execution neighborhoods for JEPA planning. "
    "Reduces LLM context by pruning unrelated files and summarizing dependencies.",
)

# ── Helpers ──

# Patterns to detect test files (language-agnostic)
_TEST_PATTERNS = re.compile(
    r"(^|[/\\])test_|_test\.|_spec\.|\.spec\.|__tests__|tests/",
    re.IGNORECASE,
)

# Priority tiers for budget enforcement (lower = dropped first)
_PRIORITY = {
    "patch_target": 5,
    "test_file": 4,
    "direct_dependency": 3,
    "transitive_dependency": 2,
    "type_provider": 1,
    "unrelated": 0,
}


def _resolve_import_to_file(
    import_stmt: str,
    indexed_files: dict[str, dict],
    project_root: str,
) -> list[str]:
    """Resolve a Python import statement to relative file paths in the index.

    Handles:
      - import os                 → os.py (stdlib — not found, excluded)
      - from typing import List   → typing.py (stdlib)
      - from . import sibling     → sibling.py or __init__.py
      - from .module import Thing → module.py
      - from package.sub import x → package/sub.py or package/sub/__init__.py
      - import package.module     → package/module.py

    Returns list of matched relative paths (empty = stdlib or unresolvable).
    """
    stmt = import_stmt.strip()
    matched = []

    # Parse the import
    module_part = ""
    names = []
    if stmt.startswith("from "):
        # from X.Y import Z [, W]
        rest = stmt[5:]
        if " import " in rest:
            module_part, _, name_part = rest.partition(" import ")
            module_part = module_part.strip()
            names = [n.strip() for n in name_part.split(",")]
        else:
            module_part = rest.strip()
    elif stmt.startswith("import "):
        # import X [, Y]
        module_part = stmt[7:].split(",")[0].strip()
        names = [module_part.split(".")[0]]

    if not module_part:
        return matched

    # Convert module path to file path candidates
    # e.g., "package.sub.module" → "package/sub/module.py"
    # Also try with __init__.py
    parts = module_part.split(".")
    rel_parts = parts[:-1]
    module_name = parts[-1]

    candidates = []

    # Case 1: package/module.py
    if rel_parts:
        base = "/".join(rel_parts)
        candidates.append(f"{base}/{module_name}.py")
    else:
        candidates.append(f"{module_name}.py")

    # Case 2: package/sub/__init__.py (module might be a package itself)
    pkg_path = "/".join(parts)
    candidates.append(f"{pkg_path}/__init__.py")
    candidates.append(f"{pkg_path}.py")

    # Relative imports (starts with .)
    if module_part.startswith("."):
        # Count dots for parent level
        dot_count = 0
        while dot_count < len(module_part) and module_part[dot_count] == ".":
            dot_count += 1
        relative_module = module_part[dot_count:]  # e.g., "submodule"
        # Find the current file's directory and go up dot_count levels
        # We'll skip this for now — relative imports are project-local
        # and hard to resolve without knowing the current file
        return matched

    # Match against indexed files
    for candidate in candidates:
        if candidate in indexed_files:
            matched.append(candidate)

    return matched


def _count_tokens(text: str) -> int:
    """Approximate token count (4 chars per token, like GPT)."""
    return len(text) // 4


def _get_language_from_ext(file_path: str) -> str:
    """Infer language from file extension."""
    ext = Path(file_path).suffix.lower()
    _ext_map = {
        ".py": "python",
        ".js": "javascript",
        ".jsx": "javascript",
        ".ts": "typescript",
        ".tsx": "tsx",
        ".rs": "rust",
        ".go": "go",
        ".java": "java",
        ".rb": "ruby",
        ".php": "php",
        ".c": "c",
        ".cpp": "cpp",
        ".h": "c",
        ".hpp": "cpp",
    }
    return _ext_map.get(ext, "unknown")


def _classify_file(
    rel_path: str,
    patch_target_path: str,
    seed_set: set[str],
    hop1_set: set[str],
    hop2_set: set[str],
) -> str:
    """Classify a file's role based on hop distance from seed."""
    if rel_path == patch_target_path:
        return "patch_target"
    if rel_path in hop1_set:
        return "direct_dependency"
    if rel_path in seed_set and rel_path != patch_target_path:
        return "direct_dependency"  # semantic match not being patched
    if rel_path in hop2_set:
        return "transitive_dependency"
    if _TEST_PATTERNS.search(rel_path):
        return "test_file"
    # Check if it looks like a type provider (interfaces, types, schemas)
    # — heuristic based on naming patterns
    name_stem = Path(rel_path).stem.lower()
    if any(kw in name_stem for kw in ("type", "interface", "schema", "model", "dto")):
        return "type_provider"
    return "direct_dependency"  # Conservative: treat as dependency if reached


def _compress_file(
    rel_path: str,
    role: str,
    entry: dict,
    code_content: str,
) -> dict:
    """Compress a file entry into a summary or full representation.

    patch_target → full code + metadata
    direct_dependency → summarized (exports, imports, depends_on)
    type_provider → symbols only
    test_file → test patterns only
    """
    syms = entry.get("symbols", {})
    functions = syms.get("functions", [])
    classes = syms.get("classes", [])
    imports_raw = syms.get("imports", [])

    # Extract export names
    exports = [f["name"] for f in functions] + [c["name"] for c in classes]

    # Extract imported module names
    imported_modules = set()
    for imp in imports_raw:
        if isinstance(imp, str):
            for m in re.findall(r"(?:from|import)\s+([a-zA-Z_][\w.]*)", imp):
                imported_modules.add(m.split(".")[0])

    summary = {
        "file": rel_path,
        "role": role,
        "language": _get_language_from_ext(rel_path),
        "exports": exports[:12],  # cap at 12
        "imports": sorted(imported_modules)[:10],
    }

    if role == "patch_target":
        # Full code allowed
        summary["full_code"] = code_content
        summary["size_chars"] = len(code_content)
    elif role == "direct_dependency":
        # Summarized with key signatures
        summary["functions"] = [
            {"name": f["name"], "line": f.get("line", 0)}
            for f in functions[:8]
        ]
        summary["classes"] = [
            {"name": c["name"], "line": c.get("line", 0)}
            for c in classes[:4]
        ]
    elif role == "type_provider":
        # Symbols only
        summary["exported_symbols"] = exports[:8]
    elif role == "test_file":
        summary["test_count"] = sum(
            1 for f in functions if f.get("name", "").startswith("test_")
        )

    return summary


# ── Budget helper ──

def _enforce_budget(
    summaries: list[dict],
    memory_hits: list[dict],
    budget: int,
) -> tuple[list[dict], list[dict], list[str]]:
    """Enforce token budget. Returns (included_summaries, included_memory, excluded_files)."""
    # Score each summary by priority tier
    scored = []
    for s in summaries:
        priority = _PRIORITY.get(s.get("role", "unrelated"), 0)
        text = json.dumps(s)
        tokens = _count_tokens(text)
        scored.append((priority, tokens, s))

    # Sort by priority descending
    scored.sort(key=lambda x: -x[0])

    included = []
    excluded = []
    used_tokens = 0

    for priority, tokens, s in scored:
        if used_tokens + tokens <= budget or priority >= _PRIORITY["patch_target"]:
            included.append(s)
            used_tokens += tokens
        else:
            excluded.append(s["file"])

    # Add memory hits under remaining budget
    included_memory = []
    for m in (memory_hits or []):
        text = json.dumps(m)
        tokens = _count_tokens(text)
        if used_tokens + tokens <= budget:
            included_memory.append(m)
            used_tokens += tokens
        else:
            break

    return included, included_memory, excluded


# ── Main tool ──

@server.tool()
def build_context(
    task: str,
    file_path: str,
    semantic_matches: Optional[list] = None,
    code_index: Optional[dict] = None,
    project_root: Optional[str] = None,
    memory_hits: Optional[list] = None,
    token_budget: int = 4096,
    expansion_hops: int = 2,
) -> dict:
    """Build a minimal compressed execution neighborhood for planning.

    Args:
        task: The user's task description.
        file_path: Absolute path to the file being modified (patch target).
        semantic_matches: List of dicts from semantic_search.search_code(),
            each with 'file', 'snippet', 'score'. Or a list of file path strings.
        code_index: Full code index manifest dict (from _code_index or get_index()).
            Expected structure: {"files": {"rel/path.py": {"symbols": {...}, ...}}}.
        project_root: Project root path for resolving relative paths. If None,
            derived from file_path or CWD.
        memory_hits: List of dicts from obsidian_brain.search_vault().
        token_budget: Max tokens for the context package (default: 4096).
        expansion_hops: How many import hops to expand (default: 2, max: 3).

    Returns:
        dict with:
            task_summary, patch_targets, dependency_summaries, memory_rules,
            excluded_files, estimated_tokens, expansion_stats
    """
    # ── Normalise inputs ──
    if project_root is None:
        project_root = str(Path(file_path).parent)
        # Walk up to find project root marker
        for parent in Path(file_path).parents:
            if (parent / ".jepa-project.json").exists() or \
               (parent / ".git").exists() or \
               (parent / "pyproject.toml").exists():
                project_root = str(parent)
                break

    project_root_str = project_root

    # Normalise file_path to relative
    try:
        patch_rel = os.path.relpath(file_path, project_root_str)
    except ValueError:
        patch_rel = os.path.basename(file_path)

    files_index = (code_index or {}).get("files", {})

    # Normalise semantic_matches: accept list of strings or list of dicts
    seed_files: set[str] = set()
    if semantic_matches:
        for m in semantic_matches:
            if isinstance(m, str):
                seed_files.add(m)
            elif isinstance(m, dict):
                fp = m.get("file", "")
                if fp:
                    seed_files.add(fp)

    # Always include the patch target as a seed
    seed_files.add(patch_rel)

    _log = lambda *args, **kwargs: print(*args, file=sys.stderr, **kwargs)
    _log(f"  [ctx] seeds ({len(seed_files)}): {list(seed_files)[:5]}...")

    # ── Step 2: AST Dependency Expansion ──
    expansion_hops = min(expansion_hops, 3)  # cap at 3

    hop1_set: set[str] = set()
    hop2_set: set[str] = set()

    # Build reverse index: which files import which modules
    # To resolve imports, we need to know what each file imports
    for seed in seed_files:
        if seed not in files_index:
            continue
        entry = files_index[seed]
        imports_raw = entry.get("symbols", {}).get("imports", [])
        for imp in imports_raw:
            resolved = _resolve_import_to_file(
                imp, files_index, project_root_str,
            )
            for r in resolved:
                if r != seed:
                    hop1_set.add(r)

    # 2nd hop: expand imports of hop1 files
    if expansion_hops >= 2:
        for hop_file in hop1_set:
            if hop_file not in files_index:
                continue
            entry = files_index[hop_file]
            imports_raw = entry.get("symbols", {}).get("imports", [])
            for imp in imports_raw:
                resolved = _resolve_import_to_file(
                    imp, files_index, project_root_str,
                )
                for r in resolved:
                    if r != hop_file and r not in seed_files and r not in hop1_set:
                        hop2_set.add(r)

    # 3rd hop (only if explicitly requested)
    if expansion_hops >= 3:
        for hop_file in list(hop2_set):
            if hop_file not in files_index:
                continue
            entry = files_index[hop_file]
            imports_raw = entry.get("symbols", {}).get("imports", [])
            for imp in imports_raw:
                resolved = _resolve_import_to_file(
                    imp, files_index, project_root_str,
                )
                for r in resolved:
                    if r not in seed_files and r not in hop1_set and r not in hop2_set:
                        hop2_set.add(r)  # merge into hop2 (3rd+ hops all grouped)

    all_candidates = seed_files | hop1_set | hop2_set
    _log(f"  [ctx] seeds={len(seed_files)}, hop1={len(hop1_set)}, hop2={len(hop2_set)}, total={len(all_candidates)}")

    # ── Step 3: File Role Classification ──
    summaries = []
    for rel_path in all_candidates:
        role = _classify_file(
            rel_path, patch_rel, seed_files, hop1_set, hop2_set,
        )
        entry = files_index.get(rel_path, {})
        # Read code content for patch_target
        code_content = ""
        if role == "patch_target":
            abs_path = os.path.join(project_root_str, rel_path)
            try:
                code_content = Path(abs_path).read_text(encoding="utf-8")
            except (OSError, UnicodeDecodeError):
                pass

        summary = _compress_file(rel_path, role, entry, code_content)
        summaries.append(summary)

    # ── Step 4 & 5: Compress + Enforce Budget ──
    included, included_memory, excluded_files = _enforce_budget(
        summaries, memory_hits or [], token_budget,
    )

    # ── Step 6: Build output ──
    task_summary = task[:200] if len(task) > 200 else task

    # Count tokens
    package_text = json.dumps({
        "patch_targets": [s for s in included if s.get("role") == "patch_target"],
        "dependency_summaries": [s for s in included if s.get("role") != "patch_target"],
        "memory_rules": included_memory,
    })
    estimated_tokens = _count_tokens(package_text)

    result = {
        "task_summary": task_summary,
        "patch_targets": [s for s in included if s.get("role") == "patch_target"],
        "dependency_summaries": [s for s in included if s.get("role") != "patch_target"],
        "memory_rules": included_memory,
        "failing_tests": [],
        "excluded_files": excluded_files,
        "estimated_tokens": estimated_tokens,
        "expansion_stats": {
            "seeds": len(seed_files),
            "hop1": len(hop1_set),
            "hop2": len(hop2_set),
            "total_candidates": len(all_candidates),
            "included": len(included),
            "excluded": len(excluded_files),
        },
    }
    return result


if __name__ == "__main__":
    server.run(transport="stdio")
