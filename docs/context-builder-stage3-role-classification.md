# Context Builder — Stage 3: File Role Classification

After dependency expansion, every candidate file gets assigned a **role** that determines how much detail it receives in Stage 4. Classification is based on hop distance from seeds and filename heuristics.

## The role system

```python
_PRIORITY = {
    "patch_target": 5,            # the file being edited
    "test_file": 4,               # test code
    "direct_dependency": 3,       # imported by seeds or seed semantic match
    "transitive_dependency": 2,   # imported by hop1 files
    "type_provider": 1,           # interfaces, schemas, models
    "unrelated": 0,               # fallback (shouldn't happen)
}
```

Priority is used later in [Stage 5](context-builder-stage5-budget-enforcement.md) for budget enforcement. Higher = kept longer.

## Classification logic (`_classify_file()`)

The function checks conditions in this order:

```
rel_path == patch_target_path?
    → "patch_target"                          (priority 5)

rel_path in hop1_set?
    → "direct_dependency"                     (priority 3)

rel_path in seed_set AND not patch_target?
    → "direct_dependency"                     (priority 3)

rel_path in hop2_set?
    → "transitive_dependency"                 (priority 2)

filename matches test pattern?
    → "test_file"                             (priority 4)

filename contains type/interface/schema/model/dto?
    → "type_provider"                         (priority 1)

fallback
    → "direct_dependency"                     (conservative)
```

## Role semantics

| Role | Priority | What it means | What it gets in Stage 4 |
|---|---|---|---|
| `patch_target` | 5 | The file being modified. Always included, bypasses budget. | Full source code |
| `test_file` | 4 | Test code detected by path patterns. Higher priority than regular deps so test structure informs edits. | Test count |
| `direct_dependency` | 3 | Imported by a seed file or is itself a seed (non-patch). The most common role. | Function/class summaries + exports + imports |
| `transitive_dependency` | 2 | Imported by a direct dependency, not by a seed. Lower priority — may be dropped under budget. | Filename + role only (implicit — no extra fields beyond base) |
| `type_provider` | 1 | Interfaces, schemas, models, DTOs. Low priority because types change rarely; useful for reference but not critical. | Exported symbols only |
| `unrelated` | 0 | Shouldn't occur in practice since `all_candidates` is populated by import traversal. | Base metadata only |

## Test file detection

```python
_TEST_PATTERNS = re.compile(
    r"(^|[/\\])test_|_test\.|_spec\.|\.spec\.|__tests__|tests/",
    re.IGNORECASE,
)
```

This is language-agnostic — works for `test_foo.py`, `foo_test.py`, `foo.spec.ts`, `__tests__/foo.js`, `tests/` directories.

## Type provider heuristics

```python
name_stem = Path(rel_path).stem.lower()
any(kw in name_stem for kw in ("type", "interface", "schema", "model", "dto"))
```

This is a simple naming convention check. Files like `schema.py`, `user.model.ts`, `types.go`, `dto.py` get classified as type providers. False positives are harmless — type_provider is low priority but still included if budget allows.

## Why priority matters

The priority system creates a **graduated retention policy**:

1. `patch_target` (5) — **mandatory**, bypasses budget
2. `test_file` (4) — usually kept, informs the edit
3. `direct_dependency` (3) — kept unless budget is very tight
4. `transitive_dependency` (2) — first to be dropped
5. `type_provider` (1) — dropped early, but types are cheap (symbols only)
6. `unrelated` (0) — never occurs in practice

## Relationship to other stages

- **Input**: `seed_set`, `hop1_set`, `hop2_set`, `patch_rel` from [Stage 2](context-builder-stage2-dependency-expansion.md)
- **Output**: Role-tagged summaries → consumed by [Stage 4](context-builder-stage4-compression.md) for content selection
