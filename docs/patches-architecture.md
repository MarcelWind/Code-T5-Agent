# Symbolic Diff Patches — Multi-File Refactoring

## Concept

Instead of the LLM emitting a full file replacement (`expected_code`), the agent
now emits **symbolic diffs**: a list of `{file, symbol, new_body}` patches that
target specific functions or classes using their code-index line ranges.

This enables:
- **Multi-file changes in a single JEPA step** — modify `agent.py`, `executor.py`,
  and `config.py` at once
- **Surgical precision** — only the changed symbols are generated, reducing
  LLM hallucination risk on boilerplate code
- **Line-shift resilience** — patches reference symbols by name, not hardcoded
  line numbers, so they stay valid after unrelated edits in the same file

---

## Patch Format

Each candidate from `plan_actions()` now carries a `patches` array:

```json
{
  "description": "add retry to _call and expose via new helper",
  "change_description": "Adds exponential backoff retry to the MCP call...",
  "patches": [
    {
      "file": "agent.py",
      "symbol": "_call",
      "new_body": "async def _call(self, server, tool, **kwargs):\n    for attempt in range(3):\n        try:\n            return await self._servers[server].call(tool, **kwargs)\n        except (ConnectionError, TimeoutError):\n            if attempt == 2: raise\n            await asyncio.sleep(0.5 * (attempt + 1))"
    },
    {
      "file": "executor.py",
      "symbol": "run_command",
      "new_body": "def run_command(cmd, cwd=None, timeout=30):\n    ..."
    }
  ]
}
```

Each patch object:
- `file` — relative path from project root (e.g. `"agent.py"`, `"core/executor.py"`)
- `symbol` — function or class name that exists in the code index; or an insertion directive (see below)
- `new_body` — **complete replacement** for that symbol, including the `def`/`class` header

### Adding New Symbols

New symbols (not yet in the code index) use special directives in the `symbol`
field instead of a symbol name:

| Directive | Behavior |
|---|---|
| `"--after <existing_symbol>"` | Resolve the existing symbol's `end_line`, insert new body right after it |
| `"--at-end-of-file"` | Append the new body at the very end of the file |

Both directives handle the full function/class definition including the
`def`/`class` header in `new_body`.  The bottom-up sort in `apply_patches()`
handles mixed insertions and replacements correctly — insertions are no-op
slices (`start > end`) that don't affect surrounding line offsets.

### Backward Compatibility

Candidates with `expected_code` (the old full-file format) still work.
The agent checks for `patches` first; if absent, falls back to `expected_code`.

---

## Execution Pipeline

### 1. Resolution (`resolve_symbol`)

`core/code_index.py::resolve_symbol()` looks up any symbol in the persisted
code index:

```python
resolve_symbol(index, "agent.py", "_call")
# → {"line": 100, "end_line": 120}
```

Raises `KeyError` if the file or symbol isn't in the index — the agent treats
this as a failure and falls back to full-file replacement.

### 2. Application (`apply_patches`)

`core/executor.py::apply_patches()`:

1. **Groups patches by file** — all patches targeting the same file are batched
2. **Resolves each symbol** to its `(start_line, end_line)` via the code index
3. **Sorts bottom-up** — patches in the same file are applied in descending
   line order so that earlier patches don't invalidate later line references
4. **Swaps line ranges** — `lines[start-1:end] = [new_body + "\n"]`

```python
success, changed_files = apply_patches(patches, code_index, project_root)
if success:
    # changed_files = ["agent.py", "core/executor.py"]
```

### 3. Scoring

The `change_description` (semantic intent) is scored against the **concatenation**
of all `new_body` snippets (the actual diff). This gives the JEPA scorer a
compact signal: does the generated diff match the described intent?

### 4. Index Update

After applying, the code index is refreshed for **every changed file**, so
subsequent steps have up-to-date line ranges.

---

## Why Symbolic Diffs?

| Aspect | Old (full file) | New (symbolic diffs) |
|---|---|---|
| Multi-file changes | ❌ One file per run | ✅ N symbols across N files |
| LLM output size | ~800 lines (entire file) | ~30 lines per function body |
| Hallucination risk | High — must reconstruct unrelated code | Low — only generates what changes |
| Corruption risk | High — one bad char corrupts the file | Low — surgical swap on verified ranges |
| Scoring signal | Whole-file embedding (noisy) | Diff-only embedding (focused) |
| Resilient to edits | No — references absolute line numbers | Yes — references symbol names |

The code index provides the **ground truth** for symbol locations, and the
LLM only needs to produce the new body for each symbol it touches. Everything
else in the file stays untouched.

---

## Config & Code

| File | What changed |
|---|---|
| `core/code_index.py` | `resolve_symbol()` — symbol→line-range lookup |
| `core/executor.py` | `apply_patches()` — bottom-up line-range swaps |
| `mcp_servers/cloud_execution/server.py` | System prompts now emit `patches[]` |
| `agent.py` | `step()` handles `patches[]` and aggregates scoring |

All changes are backward-compatible with the legacy `expected_code` format.
