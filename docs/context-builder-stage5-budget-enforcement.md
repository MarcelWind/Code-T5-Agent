# Context Builder — Stage 5: Token Budget Enforcement

After compression, the context builder enforces a hard token budget. Files that exceed the cap are **dropped in reverse-priority order** — the least important files are removed first.

## Why a budget

DeepSeek (and most LLMs) have context windows measured in thousands of tokens. Feeding the entire code index manifest or all candidate files would:

1. Waste tokens on peripherally relevant files
2. Dilute attention — the LLM has to sift through noise
3. Risk hitting context limits on large projects

The context builder caps at `CONTEXT_BUDGET_TOKENS` (default: 4096) — roughly 16KB of text, or about 400 lines of code. This is enough for the patch target + a handful of dependency summaries.

## The priority system

Each role has a numeric priority from `_PRIORITY`:

```python
_PRIORITY = {
    "patch_target": 5,            # always included
    "test_file": 4,
    "direct_dependency": 3,
    "transitive_dependency": 2,
    "type_provider": 1,
    "unrelated": 0,
}
```

Higher priority = kept longer when budget runs out.

## Budget enforcement algorithm (`_enforce_budget()`)

```
1. Score each summary: (priority, token_count, summary_dict)
2. Sort by priority descending (highest first)
3. Walk through sorted list:
   - If (used_tokens + tokens) ≤ budget  →  INCLUDE
   - Else if priority ≥ 5 (patch_target) →  INCLUDE (mandatory)
   - Else                                 →  EXCLUDE
4. Append memory hits under remaining budget
5. Return (included_summaries, included_memory, excluded_files)
```

**Key detail**: `patch_target` (priority 5) bypasses the budget check entirely. It's always included, even if it would exceed the budget alone. This guarantees the LLM always sees the file it needs to modify.

### Token accounting

```python
def _count_tokens(text: str) -> int:
    return len(text) // 4
```

This is a **4-characters-per-token** approximation (similar to GPT tokenization for code). It's not exact but provides a consistent cap — serializing the full package and counting is done after enforcement for the `estimated_tokens` field.

### Memory rules

Memory hits from the vault are appended **after** file summaries, only if budget remains:

```python
for m in (memory_hits or []):
    text = json.dumps(m)
    tokens = _count_tokens(text)
    if used_tokens + tokens <= budget:
        included_memory.append(m)
        used_tokens += tokens
    else:
        break
```

This means memory rules are the first thing dropped when the file neighborhood fills the budget — they're supplementary context, not essential.

## Example: 4096-token budget

| Candidate | Role | Tokens | Priority | Decision |
|---|---|---|---|---|
| `agent.py` | patch_target | ~450 | 5 | ✅ Always include |
| `core/config.py` | direct_dependency | ~25 | 3 | ✅ Include |
| `core/code_index.py` | direct_dependency | ~40 | 3 | ✅ Include |
| `mcp_servers/semantic_search/server.py` | direct_dependency | ~35 | 3 | ✅ Include |
| `mcp_servers/validators/server.py` | direct_dependency | ~30 | 3 | ✅ Include |
| `core/executor.py` | transitive_dependency | ~8 | 2 | ✅ Include |
| `core/onboarding.py` | transitive_dependency | ~10 | 2 | ✅ Include |
| `core/encoder.py` | transitive_dependency | ~8 | 2 | ❌ Excluded |
| **Total used** | | **~606** | | **7 included, 1 excluded** |

In this case, `core/encoder.py` is a transitive dependency and gets dropped. The LLM never sees it — but it doesn't need to, because it's not directly related to the task.

## Worst-case behavior

If the budget is very tight (e.g., 512 tokens) and the patch target itself is large (400 tokens), there may be room for only 1–2 dependency summaries. In that case:

1. `patch_target` is always included (mandatory)
2. `test_file` entries are included next (highest non-mandatory priority)
3. If budget remains, `direct_dependency` entries are added in priority order
4. `transitive_dependency` and `type_provider` entries are dropped entirely

The `excluded_files` list in the output tells the agent (and logs) which files were dropped, so this behavior is transparent.

## Output structure

```python
{
    "patch_targets": [...],            # always present
    "dependency_summaries": [...],     # may be partial
    "memory_rules": [...],             # may be empty
    "excluded_files": [...],           # files that didn't fit
    "estimated_tokens": 606,           # token count of serialized package
    "expansion_stats": {
        "seeds": 3,
        "hop1": 2,
        "hop2": 4,
        "total_candidates": 9,
        "included": 7,
        "excluded": 2
    }
}
```

## Relationship to other stages

- **Input**: Compressed summaries from [Stage 4](context-builder-stage4-compression.md) + memory hits from agent-level vault search
- **Output**: Final context package → serialized as JSON → passed to `cloud_execution.plan_actions()` as `context_package`
