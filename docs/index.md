# JEPA Agent — Documentation

## Context Builder

The context builder constructs a minimal, compressed *execution neighborhood* from the codebase, replacing ad-hoc AST summaries with structured, budget-enforced context for the LLM.

- [**Overview**](context-builder-overview.md) — Full 6-stage pipeline diagram, wiring, configuration reference
- [**Stage 1: Seed Files**](context-builder-stage1-seed-files.md) — CodeT5+ embedding search, seed set construction, why embeddings over grep
- [**Stage 2: AST Dependency Expansion**](context-builder-stage2-dependency-expansion.md) — Import-hop traversal (Hop 1 → Hop 2), `_resolve_import_to_file()`, stdlib auto-exclusion
- [**Stage 3: File Role Classification**](context-builder-stage3-role-classification.md) — Role system (patch_target / direct_dependency / transitive_dependency / type_provider / test_file), priority tiers, test detection patterns
- [**Stage 4: Context Compression**](context-builder-stage4-compression.md) — Per-role compression strategies, token cost examples, language detection
- [**Stage 5: Token Budget Enforcement**](context-builder-stage5-budget-enforcement.md) — Priority-based sorting, mandatory patch_target bypass, memory rule inclusion, worst-case behavior, output structure

## Quick reference

### Symbolic Diff Patches

DeepSeek emits `patches[]` (file + symbol + new_body) instead of full-file replacement, enabling multi-file changes in a single JEPA step.

- [**Patches Architecture**](patches-architecture.md) — Format, resolution via code index, bottom-up application, scoring

| File | Purpose |
|---|---|
| `mcp_servers/context_builder/server.py` | All 6 stages (build_context tool + helpers) |
| `agent.py` | `_build_execution_context()`, `_semantic_search()`, `_vault_search()`, `step()` wiring |
| `mcp_servers/cloud_execution/server.py` | `plan_actions()` with context-aware prompt |
| `core/config.py` | `CONTEXT_BUDGET_TOKENS`, `CONTEXT_EXPANSION_HOPS`, `CONTEXT_MAX_SEED_FILES`, `CONTEXT_INCLUDE_MEMORY` |
| `core/code_index.py` | `resolve_symbol()` — symbol → line-range lookup |
| `core/executor.py` | `apply_patches()` — bottom-up line-range swaps |
