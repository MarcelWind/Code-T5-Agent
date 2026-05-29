# Context Builder — Stage 4: Context Compression

Each candidate file gets **compressed differently depending on its role**. The goal is to give DeepSeek maximum useful information while spending as few tokens as possible on low-value files.

## Why compress

Without compression, every file in the neighborhood would be included in full — potentially thousands of tokens for files the LLM barely needs to reference. Compression exploits the fact that:

- **patch_target** needs full code (the LLM rewrites it)
- **direct_dependency** needs its API surface (function signatures, exports) — not implementation details
- **type_provider** needs its type definitions only
- **transitive_dependency** is just a hint that the file exists

## Compression per role (`_compress_file()`)

Every role gets a **base summary**:

```python
summary = {
    "file": rel_path,
    "role": role,
    "language": _get_language_from_ext(rel_path),  # python, javascript, rust, etc.
    "exports": exports[:12],                        # max 12 exported names
    "imports": sorted(imported_modules)[:10],        # max 10 non-stdlib imports
}
```

Then role-specific fields are added:

### `patch_target` — Full code

```python
summary["full_code"] = code_content     # entire file text
summary["size_chars"] = len(code_content)
```

The patch target is the file being edited. DeepSeek needs its complete contents to produce a valid replacement. This is the most expensive entry but also **mandatory** — it bypasses budget enforcement.

### `direct_dependency` — Function/class signatures

```python
summary["functions"] = [
    {"name": f["name"], "line": f.get("line", 0)}
    for f in functions[:8]   # max 8 functions
]
summary["classes"] = [
    {"name": c["name"], "line": c.get("line", 0)}
    for c in classes[:4]     # max 4 classes
]
```

Includes function names + line numbers so DeepSeek can reference them. No function bodies — the implementation isn't needed to understand the API. If a file has 20 functions, only the first 8 are listed (tree-sitter order, typically declaration order).

### `type_provider` — Symbols only

```python
summary["exported_symbols"] = exports[:8]   # max 8 names
```

Types, interfaces, and schemas change rarely. The LLM just needs to know a file exists with certain type names to reference them correctly. No function details, no imports.

### `test_file` — Test count

```python
summary["test_count"] = sum(
    1 for f in functions if f["name"].startswith("test_")
)
```

A count of test functions is enough to signal "this file has tests for the target." Detailed test names are omitted.

### `transitive_dependency` — Base only

No additional fields beyond the base summary (file, role, language, exports, imports). These are low-priority files that may be dropped in budget enforcement anyway.

## Token cost examples

| Role | Approximate tokens | Contents |
|---|---|---|
| `patch_target` (50-line file) | ~80–150 | Full code |
| `patch_target` (200-line file) | ~300–600 | Full code |
| `direct_dependency` (simple) | ~10–20 | 2 functions, 1 import |
| `direct_dependency` (complex) | ~30–50 | 8 functions, 4 classes, 10 imports |
| `type_provider` | ~5–10 | 5 type names |
| `test_file` | ~8–12 | Role + test count |
| `transitive_dependency` | ~5–10 | Base fields only |

## Language detection

Language is inferred from file extension in `_get_language_from_ext()`:

```python
".py" → python    ".js" → javascript    ".ts" → typescript
".rs" → rust      ".go" → go            ".java" → java
".rb" → ruby      ".c" → c              ".cpp" → cpp
```

Unknown extensions get `"unknown"` — compression still works (exports/imports from tree-sitter may be empty), but DeepSeek gets a language hint.

## Relationship to other stages

- **Input**: Role-tagged entries from [Stage 3](context-builder-stage3-role-classification.md) + raw file content (read from disk for patch_target)
- **Output**: Compressed summaries → consumed by [Stage 5](context-builder-stage5-budget-enforcement.md) for budget enforcement
