# Context Builder — Stage 2: AST Dependency Expansion

After seed files are selected, the context builder walks **import edges** outward to discover files that aren't semantically similar to the task but are structurally required.

## How it works

In `server.py:build_context()`, lines ~340–380:

```
seed_files ────→ hop1_set ────→ hop2_set
 (semantic)     (imported by)  (imported by hop1)
```

### Hop 1: Direct imports from seeds

```python
for seed in seed_files:
    entry = files_index[seed]
    imports_raw = entry["symbols"]["imports"]
    for imp in imports_raw:
        resolved = _resolve_import_to_file(imp, files_index, project_root)
        for r in resolved:
            if r != seed:
                hop1_set.add(r)
```

For each seed file, the context builder looks up that file's `imports` in the tree-sitter code index. Each import string (e.g. `from core.config import CONTEXT_BUDGET_TOKENS`) is resolved to a file path in the index via `_resolve_import_to_file()`.

### Hop 2: Imports of Hop 1 files

```python
for hop_file in hop1_set:
    entry = files_index[hop_file]
    # ... same resolution logic ...
    for r in resolved:
        if r not in seed_files and r not in hop1_set:
            hop2_set.add(r)
```

Same process, one level deeper. Hop 2 files are *transitive* dependencies — they're needed by the direct dependencies, not by the seeds themselves.

### Hop 3 (optional)

If `expansion_hops >= 3`, a third pass runs on `hop2_set`. Results are merged into `hop2_set` (3rd+ hops aren't distinguished).

## Import resolution (`_resolve_import_to_file()`)

This function converts an import statement string to file paths that exist in the code index:

| Import statement | Candidates checked |
|---|---|
| `import os` | `os.py` → not in index (stdlib) → excluded |
| `from core.config import X` | `core/config.py` → matched |
| `from .utils import Y` | Relative — skipped (returns `[]`) |
| `import package.sub.module` | `package/sub/module.py`, `package/sub/module/__init__.py` |
| `from typing import List` | `typing.py` → not in index (stdlib) → excluded |

**Key behavior**: If a resolved path doesn't exist in the code index, it's silently dropped. This means:

- **Stdlib auto-excludes** — `os`, `sys`, `typing`, `collections`, etc. resolve to files not in the project index
- **Third-party packages auto-exclude** — `numpy`, `torch`, `flask` etc. live in `site-packages`, not the workspace index
- **Only project files survive** — every file in `hop1_set` and `hop2_set` is guaranteed to be a real project file

## All candidates

```python
all_candidates = seed_files | hop1_set | hop2_set
```

The union of all three sets is passed to role classification. Typical counts:

- Small task (single file, few imports): **3–5 candidates**
- Medium task (multiple imports, utility deps): **8–15 candidates**
- Large task (many transitive deps): **20–30+ candidates** (then budget enforcement prunes)

## Edge cases

| Scenario | Behavior |
|---|---|
| Seed file has no imports | Hop 1 empty. Only the seed survives. |
| All imports are stdlib | Hop 1 empty (stdlib not in index). Only seeds survive. |
| Circular imports | Not explicitly handled — Hop 1/2 deduplication via set membership prevents infinite loops naturally. |
| Import resolves to the seed itself | Filtered by `if r != seed`. |
| Same file in Hop 1 and seeds | Set deduplication keeps one copy; `_classify_file` uses Hop 1 role. |
| File in index but import can't be parsed | `_resolve_import_to_file` returns `[]` — no effect. |

## Relationship to other stages

- **Input**: seed files from [Stage 1](context-builder-stage1-seed-files.md)
- **Output**: `hop1_set`, `hop2_set`, `all_candidates` → consumed by [Stage 3](context-builder-stage3-role-classification.md) for role assignment
