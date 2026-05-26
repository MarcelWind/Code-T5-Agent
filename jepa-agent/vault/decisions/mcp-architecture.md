# MCP Architecture Decisions

## 2026-05-27: 6-Server Topology
**Decision**: Decompose the JEPA pipeline into 6 independent MCP servers communicating via stdio JSON-RPC.
**Rationale**: Each server owns exactly one capability domain. Independent startup, testing, and scaling. Clear failure isolation.
**Topology**:
```
┌─ agent.py (orchestrator) ──────────────────────────┐
│  → semantic_search      (CodeT5+ embeddings)       │
│  → cloud_execution      (DeepSeek code gen)        │
│  → validators           (JEPA scoring + syntax)    │
│  → code_understanding   (AST analysis)             │
│  → obsidian_brain       (vault persistence)        │
│  → local_router         (intent routing, future)   │
└────────────────────────────────────────────────────┘
```
**Current routing**: Direct calls — agent.py calls the server that provides the needed tool.
**Future routing**: All calls go through local_router first for multi-agent coordination.

## 2026-05-27: stdio Transport over HTTP
**Decision**: Use stdio subprocess transport instead of HTTP/SSE for MCP communication.
**Rationale**:
- Zero network setup (no ports, no CORS, no auth)
- Process isolation without container overhead
- Automatic lifecycle management (subprocess dies with parent)
- Simpler debugging (stdout = JSON-RPC, stderr = logs)
**Trade-offs**: No remote access; all servers must run on same machine.

## 2026-05-27: Package Name mcp_servers
**Decision**: Name the server package `mcp_servers` (not `mcp` or `servers`).
**Rationale**: `mcp` is the official Anthropic MCP SDK package name — using `mcp/` as a local package shadows it and breaks `from mcp import ClientSession`.
**Consequence**: All .roo/mcp.json `args` use `-m mcp_servers.<name>.server`.

## 2026-05-27: Server-Side Model Loading
**Decision**: Each server that needs models loads them independently at import time (lazy init via `@functools.lru_cache` or similar).
**Rationale**: Avoids passing model handles across process boundaries. Each server owns its resource lifecycle.
**Consequence**: Cold start penalty (3-5s) on first call to semantic_search or validators.

## 2026-05-27: Env Var Injection via SERVER_DEFS
**Decision**: Server env vars (DEEPSEEK_API_KEY, HF_HOME, VAULT_PATH) are defined in `agent.py:SERVER_DEFS` and merged into `os.environ` at connection time.
**Rationale**: Single source of truth for all server configurations. No hardcoded paths in server code.
**Trade-offs**: Relative paths ("models", ".", "vault") must resolve correctly from the workspace root.
