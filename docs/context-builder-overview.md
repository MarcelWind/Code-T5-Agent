# Context Builder — Overview

The context builder is a 6-stage pipeline that sits between **semantic search** and the **LLM plan call**. Its job is to replace the old ad-hoc AST summary (one file, unstructured) with a minimal, compressed *execution neighborhood* — the smallest set of files DeepSeek needs to produce correct patches.

## Why it exists

The old approach dumped a tree-sitter symbol summary of *one file* into the prompt. This missed:

- Files that don't contain task keywords but are structurally required (imported dependencies)
- Type definitions, schemas, and interfaces the code depends on
- Memory rules and project conventions from the vault
- Budget enforcement — context could grow unbounded

The context builder solves all four by scanning the full index, walking import edges, compressing per role, and enforcing a hard token cap.

## Pipeline diagram

```
Task ──→ Stage 1: Seed Files
              │  (semantic search + patch target)
              ▼
         Stage 2: AST Dependency Expansion
              │  (import hops 1→2, stdlib auto-excluded)
              ▼
         Stage 3: File Role Classification
              │  (patch_target / direct_dependency / ...)
              ▼
         Stage 4: Context Compression
              │  (full code vs summary vs symbols-only)
              ▼
         Stage 5: Token Budget Enforcement
              │  (priority tiers, drop lowest)
              ▼
         Stage 6: Memory Rule Inclusion
              │  (vault hits under remaining budget)
              ▼
         JSON context_package → DeepSeek
```

## Stage breakdown

| Stage | File | What it does |
|---|---|---|
| [Stage 1: Seed Files](context-builder-stage1-seed-files.md) | `agent.py:_build_execution_context()` → `server.py:build_context()` | CodeT5+ embedding search + patch target → seed set |
| [Stage 2: AST Dependency Expansion](context-builder-stage2-dependency-expansion.md) | `server.py:build_context()` lines ~340–380 | Walk import edges from seeds: Hop 1 (direct), Hop 2 (transitive). Stdlib drops out. |
| [Stage 3: File Role Classification](context-builder-stage3-role-classification.md) | `server.py:_classify_file()` | Each candidate file tagged: patch_target, direct_dependency, transitive_dependency, type_provider, test_file |
| [Stage 4: Context Compression](context-builder-stage4-compression.md) | `server.py:_compress_file()` | patch_target → full code; direct_dependency → function/class summaries; type_provider → symbols only |
| [Stage 5: Token Budget Enforcement](context-builder-stage5-budget-enforcement.md) | `server.py:_enforce_budget()` | Priority tiers (5→0), additive inclusion, memory rules tacked on under remaining budget |
| Stage 6: Memory Rule Inclusion | `agent.py:_build_execution_context()` step 2 | Vault search hits injected as `memory_rules` — only if budget remains after all files |

## How it's wired

In `agent.py:step()`, the old AST context block was replaced with a 3-line call:

```python
context_package = await self._build_execution_context(task, file_path)
context_json = json.dumps(context_package) if context_package else ""
candidates = await self._plan_actions(code, task, k=k, context_package=context_json)
```

The `cloud_execution` server detects `context_package` and switches to `CONTEXT_AWARE_SYSTEM_PROMPT`, which instructs DeepSeek to use the neighborhood for structural understanding but only patch the target file.

## Configuration (`core/config.py`)

| Constant | Default | Purpose |
|---|---|---|
| `CONTEXT_BUDGET_TOKENS` | 4096 | Hard cap on serialized context package |
| `CONTEXT_EXPANSION_HOPS` | 2 | Import hops from seeds (max 3) |
| `CONTEXT_MAX_SEED_FILES` | 5 | Top-K for semantic search |
| `CONTEXT_INCLUDE_MEMORY` | True | Whether to search vault for rules |

## Key files

- `mcp_servers/context_builder/server.py` — all 6 stages implemented here
- `agent.py` — `_build_execution_context()`, `_semantic_search()`, `_vault_search()`
- `mcp_servers/cloud_execution/server.py` — `plan_actions()` with context-aware prompt
- `core/config.py` — budget and expansion constants
