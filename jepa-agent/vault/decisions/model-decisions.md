# Model & Architecture Decisions

## 2024-05-27: MCP-Based Orchestration
**Decision**: Replace direct function calls with MCP server orchestration.
**Rationale**: Loose coupling, independent development, clear contracts between components.
**Trade-offs**: More boilerplate (server stubs), but far more maintainable for complex pipeline.

## 2024-05-24: CodeT5+ 220M as Embedder
**Decision**: Use `Salesforce/codet5p-220m` (T5EncoderModel) for semantic code embeddings.
**Rationale**: 768-dim is sufficient for similarity ranking; 220M params is lightweight for local inference.
**Trade-offs**: Larger models (like CodeBERT 770M) might capture more nuance but add latency.

## 2024-05-24: JEPA Loss = Cosine Distance
**Decision**: Use cosine_distance as default JEPA loss function.
**Rationale**: [0,2] bounded range, works well with 768-dim normalized embeddings. l2_distance as alt.
**Trade-offs**: Cosine ignores magnitude; l2 captures both direction and magnitude but is unbounded.

## 2024-05-24: Local Model Cache
**Decision**: Store HuggingFace models in project-local `models/` directory.
**Rationale**: Avoids polluting user home; makes project self-contained; easy to clear.
**Implementation**: Set HF_HOME + HF_HUB_CACHE env vars before any model loads.

## 2026-05-27: tree-sitter for AST Analysis
**Decision**: Use `tree-sitter` (v0.25.2) with `tree-sitter-python` (v0.25.0) for code structural analysis.
**Rationale**: Language-agnostic AST parsing, faster than building custom parsers, supports incremental parsing.
**Trade-offs**: Adds ~50MB of native parser binaries per language; Python AST would suffice for Python-only files.

## 2026-05-27: MCP SDK v1.27.1
**Decision**: Pin to `mcp` package v1.27.1 from Anthropic.
**Rationale**: Stable `FastMCP` and `stdio_client` APIs. v1.27.x is the latest stable line.
**Key classes used**: `FastMCP` (server), `StdioServerParameters` + `stdio_client` (transport), `ClientSession` (client).

## 2026-05-27: Direct Tool Calls (No Hub-and-Spoke)
**Decision**: agent.py calls each MCP server directly by capability, not through a central router.
**Rationale**: Simpler code, lower latency, easier debugging. The `local_router` server exists but is reserved for future multi-agent coordination.
**Trade-offs**: Orchestrator must know server topology. Changing the server decomposition requires updating agent.py call sites.
