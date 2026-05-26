# Known Issues & Lessons

## Fixed
1. **tokenizers >= 0.21 crash**: CodeT5+ tokenizer crashes with `AddedToken` format bug in tokenizers >= 0.21. Fixed by pinning transformers==4.47.1 (which pins tokenizers==0.21.4). If upgrading, test encoder first.
2. **Model cache in user home**: Default HF_HOME points to `~/.cache/huggingface/`. Fixed by setting HF_HOME + HF_HUB_CACHE to project `models/` dir before any imports.
3. **JEPA score always 0**: When change_description == expected_code, Z_hat == Z_actual → distance = 0. Fixed by splitting prompt output into change_description (encoded as Z_hat/prediction) vs expected_code (encoded as Z_actual/ground truth).

## Active
- **MCP server cold start**: Each server imports models (CodeT5+ 220M, tree-sitter parsers) on first call. Can add 3-5s latency on the first step. Mitigation: warm-up on agent init.
- **DeepSeek API rate limits**: cloud_execution server makes HTTP calls to api.deepseek.com. No retry logic yet — network errors bubble up as tool failures.
- **tree-sitter-python version**: Must match tree-sitter 0.25.x. tree-sitter v0.25.2 works with tree-sitter-python v0.25.0.
- **HF_HOME path resolution**: SERVER_DEFS uses relative paths ("models", "."). Must be resolved to absolute paths before passing as subprocess env, or the servers may mis-resolve relative to their CWD.
- **No auth or secrets manager**: DEEPSEEK_API_KEY is passed in plain env to subprocess. Not suitable for shared environments.

## Completed
- **Phase 1 (core/ restructure)**: ✅ All 11 dirs created, files moved, imports fixed
- **Phase 2 (6 MCP servers)**: ✅ All servers implemented and tested independently
- **Phase 3 (MCP orchestration)**: ✅ agent.py rewritten as async MCP orchestrator, main.py updated
- **tree-sitter**: ✅ Installed (v0.25.2 + tree-sitter-python v0.25.0)
- **tokenizers compat**: ✅ transformers==4.47.1 pins tokenizers==0.21.4 (avoids AddedToken crash)
- **mcp SDK shadowing**: ✅ Package renamed from `mcp/` to `mcp_servers/` to avoid collision with installed `mcp` SDK
- **HF_HOME project-local**: ✅ HF_HOME + HF_HUB_CACHE set to `models/` dir
