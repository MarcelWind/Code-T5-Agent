# Context Builder — Stage 1: Seed Files

## What Are Seed Files?

Seed files are the **entry points** into the dependency graph. They're the starting set of files from which Hop 1 (direct imports) and Hop 2 (transitive imports) expand outward during context construction.

In code, the seed set is constructed at two levels:

### 1. Agent level (`agent.py:_build_execution_context()`)

```python
semantic_matches = await self._semantic_search(task, top_k=CONTEXT_MAX_SEED_FILES)
```

This calls the `semantic_search` MCP server's `search_code` tool, which uses CodeT5+ embeddings to find up to `CONTEXT_MAX_SEED_FILES` (default: 5) semantically relevant files in the workspace.

### 2. Context builder level (`server.py:build_context()`)

```python
seed_files: set[str] = set()
for m in semantic_matches:
    seed_files.add(m["file"])      # extract file paths from search results

seed_files.add(patch_rel)          # ALWAYS add the file being edited
```

The final seed set is the **union of semantic matches and the patch target** (deduplicated). Even if the embedding model returns no matches, the patch target itself is always present, so the import-expansion still has at least one starting point.

---

## Their Purpose

Seeds serve as the **bridge between the task description and the codebase**. They answer the question: *"Given this natural language task, which files in the project are relevant starting points?"*

The expansion algorithm then walks **import edges** outward from these seeds:

```
seed_files  ──imports──→  hop1_set  ──imports──→  hop2_set
   (semantic)           (direct deps)          (transitive deps)
```

This catches files that the embedding model might miss but are structurally required. For example, if the task says "add input validation", semantic search might find `validators.py`, but Hop 1 will also pull in `schema.py` because `validators.py` imports it — a file the embedding model may not have scored highly but is still essential context.

If semantic search returns nothing (new project, unindexed files, or a domain-specific query), the patch target file alone anchors the dependency expansion so the pipeline degrades gracefully.

---

## Why Use Embeddings to Find Them?

Keyword search (grep) would miss files that are **semantically related but lexically different**. Here's the concrete difference:

| Query | Keyword hit? | Embedding hit? |
|---|---|---|
| `"sort a vector and remove duplicates"` | Needs exact words `sort`, `vector`, `duplicate` to appear in the file | Matches `process_random_uint_vector()` because CodeT5+ understands the *meaning* of the operation |
| `"add input validation to the pipeline"` | Grep for `validate` might find `validators.py` | Finds it even if the word "pipeline" doesn't appear — the embedding captures conceptual similarity |
| `"fix the encoding failing silently"` | No single keyword reliably matches | Finds `core/encoder.py` because `encode` + `silent failure` maps to error-handling patterns in the vector space |

### The embedding pipeline

1. User's task description → **CodeT5+ encoder** → 768-dim embedding vector
2. Each file in the workspace is chunked into its top-level functions/classes → each chunk also gets an embedding vector
3. **Cosine similarity** between the query vector and each chunk vector → top-5 highest scores
4. Those file paths become the semantic matches passed to the context builder

This is the **"observe"** part of JEPA — the agent doesn't guess which files matter based on rules or heuristics. It uses the same embedding model that encodes the current code state (`S_t`) to also identify relevant context. The same semantic space that measures *"how similar are two code snippets"* also measures *"how relevant is this file to this task."*

### Role classification after expansion

After dependency expansion completes, `_classify_file()` uses the original seed set to assign roles:

- **`patch_target`** — the file being edited (always a seed)
- **`direct_dependency`** — any seed file that isn't the patch target, plus any Hop 1 file
- **`transitive_dependency`** — Hop 2 files (imported by Hop 1)
- **`type_provider`**, **`test_file`** — pattern-matched from filenames

Seeds that aren't the patch target get priority 3 (direct dependency) — they receive a function-level summary rather than full code. They're relevant context, just not the primary edit target.

---

## Edge Cases

| Scenario | Behavior |
|---|---|
| No `.py` files in workspace | `search_code` returns `[{message: "No Python files found..."}]`, which has no `"file"` key → filtered out. Only the patch target remains as seed. |
| Semantic search returns 0 results | Logged as `"no semantic matches, using file imports as seeds"`. Patch target is still a seed, so Hop 1 expansion runs from its imports. |
| Task is very short (e.g., "fix bug") | Embedding is noisy but not harmful — the top-5 matches may be weak, but the patch target + its imports still provide reasonable coverage. |
| All semantic matches are the patch target itself | Deduplication collapses the set to 1. Fine — Hop 1 and Hop 2 expansion still runs. |
| Project has only 1 file | Seed = patch target only. Hop 1/2 are empty. Output = just the patch_target entry. Budget enforcement is trivially satisfied. |
