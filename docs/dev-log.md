# Dev Log

## 2026-05-30 — Symbolic Diff Patches & Context Builder

### Changes Made

- **Relevant-region extraction** (`mcp_servers/context_builder/server.py`)
  - Added `_find_relevant_regions()` — cross-references semantic search matches with code index line ranges to extract only the functions/classes relevant to the task
  - Modified `_compress_file()` to accept `relevant_regions` param — patch_target files now emit sliced code instead of full file
  - Falls back to full file when no semantic matches exist

- **Symbolic diff patches** (multi-file refactoring)
  - `core/code_index.py`: Added `resolve_symbol(index, rel_path, name)` — looks up a symbol's `{line, end_line}` in the persisted AST manifest
  - `core/executor.py`: Added `apply_patches(patches, code_index, project_root)` — groups patches by file, resolves symbols, sorts bottom-up, performs line-range swaps
  - `agent.py`: Updated `step()` to handle `patches[]` format — concatenates `new_body` for scoring, calls `apply_patches`, refreshes code index for all changed files
  - `mcp_servers/cloud_execution/server.py`: Updated both system prompts to emit `patches[]` array with `{file, symbol, new_body}` instead of full-file `expected_code`

- **Documentation**
  - `docs/patches-architecture.md` — Full spec: format, execution pipeline, comparison table, code references
  - `jepa-agent/README.md` — Updated MCP Servers table (added `context_builder`), Core Modules table (`apply_patches`, `resolve_symbol`), Data Flow diagram (symbolic diffs via code index), How It Works (step 3→11 rewritten), Code Index section (`resolve_symbol`, line-range manifest), Self-Bootstrapping (multi-file note)
  - `docs/index.md` — Added Symbolic Diff Patches subsection, updated file table with `code_index.py` and `executor.py` entries

- **Context builder MCP server** (earlier)
  - 6-stage pipeline: seed files → AST dependency expansion → role classification → compression → budget enforcement → output
  - `build_context()` tool orchestrates all stages with token budget, memory inclusion, and structured output

### Removed / Deprecated

- Full-file `expected_code` format is still supported as fallback (backward compat), but the primary path is now `patches[]`

---

### Problems Still to Solve

1. ✅ **New symbol insertion** — `apply_patches()` now handles `--after <existing_symbol>` (insert after named symbol) and `--at-end-of-file` (append at end) directives. Prompts updated to replace the vague `<new_function_name>` convention with these explicit directives. Handles mixed insertions + replacements via the existing bottom-up sort.

2. **Multi-symbol single file edge case** — If two patches target the same function in the same file (e.g. rename + modify), the second patch will fail because the first already changed the line range. The bottom-up sort helps with *different* symbols at different lines, but doesn't help with overlapping ranges.

3. **Scoring granularity** — All `new_body` strings are concatenated into one embedding and compared against the single `change_description`. If a candidate has 5 patches but only 1 is wrong, the embedding signal may be diluted. Per-patch scoring would be more precise but more complex.

4. **Code index staleness** — If the file is edited externally (or by a previous failed step), the code index line ranges may be stale. `resolve_symbol()` returns whatever is in the manifest, even if the file on disk differs. Mtime checks during `step()` help but there's a race window.

5. **Context builder test coverage** — The 6-stage pipeline has no unit tests yet. Integration tests rely on the full agent loop, which is slow and flaky. Need dedicated tests for each `_compress_file()` variant, `_find_relevant_regions()`, `_resolve_import_to_file()`, and budget enforcement logic.

6. **Patch candidate quality** — DeepSeek sometimes emits `patches[]` with incorrect `symbol` names (typos, or guesses instead of exact names from context). The agent currently fails hard on `KeyError`. A fuzzy-match fallback (Levenshtein on symbol names) could reduce false failures.

7. **Self-bootstrapping ceiling** — The agent successfully modified `core/config.py` (add a version string) but more complex self-modifications (adding a new MCP server, restructuring a module) haven't been tested. The symbolic diffs architecture was designed to enable this, but the converge rate on multi-file self-changes is unknown.

8. **JEPA convergence criteria** — Loss < 0.01 is the stop condition, but loss values vary wildly between tasks. Sometimes the agent converges in 1 step on a good candidate, sometimes it oscillates. Adaptive threshold or early-stopping based on test pass rate might be more reliable.

9. **Token budget for large files** — The context builder's token budget enforcement works, but for very large files (>2000 lines) even the compressed relevant-region extraction may exceed the budget. No chunking or summarization fallback exists yet.
