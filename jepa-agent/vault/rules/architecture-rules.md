# Architecture Rules

## Core Principles
1. **JEPA Loop**: Observe → Encode → Predict → Score → Select → Apply → Store
2. **MCP orchestration**: Every capability is an MCP server. agent.py (orchestrator) connects to all 6 servers via stdio JSON-RPC. No direct core imports from MCP servers.
3. **Embedding-space validation**: Every candidate action is scored by JEPA loss (cosine distance between predicted change embedding and actual code embedding) before execution.
4. **Async-first**: All MCP calls are async. Synchronous entry points use `asyncio.run()` as thin wrapper.

## Directory Layout
```
jepa-agent/
├── core/              # Fallback core modules (encoder, predictor, scorer, executor, config)
├── mcp_servers/       # 6 MCP servers
│   ├── local_router/       # Intent routing
│   ├── code_understanding/ # AST analysis (tree-sitter)
│   ├── semantic_search/    # CodeT5+ embeddings
│   ├── obsidian_brain/     # Vault memory
│   ├── cloud_execution/    # DeepSeek API proxy
│   └── validators/         # JEPA scoring + syntax validation
├── vault/             # Persistent memory (decisions/, rules/, lessons/, patterns/)
│   ├── _index.md           # Vault index
│   ├── architecture-rules.md
│   ├── jepa-pipeline.md
│   └── ...
├── .roo/              # Roo Code configuration (mcp.json, roomodes.json)
├── tests/             # Unit tests
└── models/            # Local model cache (HF_HOME + HF_HUB_CACHE)
```

## Pipeline Flow (one JEPA step)
```
user task
  → agent.py (orchestrator)
     1. semantic_search.encode_code()       → Z_t (current state embedding)
     2. cloud_execution.plan_actions(k)     → k candidates {change_description, expected_code}
     3. validators.rank_candidates()        → JEPA loss per candidate
     4. validators.validate_syntax()        → syntax check on best candidate
     5. core.executor.apply_patch()         → write file
     6. obsidian_brain.write_vault()        → log lesson
```

## SERVER_DEFS (agent.py)
| Server             | Tool examples                               | Env vars                        |
|--------------------|---------------------------------------------|---------------------------------|
| local_router       | list_capabilities, route_request            | PYTHONPATH                      |
| code_understanding | parse_code, get_functions, get_classes      | PYTHONPATH, HF_HOME             |
| semantic_search    | search_code, encode_code, compute_similarity | PYTHONPATH, HF_HOME             |
| obsidian_brain     | read_vault, write_vault, search_vault        | PYTHONPATH, VAULT_PATH          |
| cloud_execution    | generate_code, plan_actions, chat_completion | PYTHONPATH, DEEPSEEK_API_KEY    |
| validators         | validate_code, rank_candidates, validate_syntax | PYTHONPATH, HF_HOME          |

## MCP Communication Protocol
- Transport: stdio (subprocess, JSON-RPC over stdin/stdout)
- SDK: `mcp` Python package (Anthropic MCP SDK v1.27.1+)
- Server launch: `sys.executable -m mcp_servers.<name>.server`
- Connection: `stdio_client(StdioServerParameters)` → `ClientSession` → `session.initialize()`
- Tool call: `session.call_tool(tool_name, arguments=dict)`
- Result parsing: `CallToolResult` → extract `.content[0].text` → `json.loads()`
- Cleanup: `session.__aexit__` then `stdio_client.__aexit__`

## Routing Rules
- All tool calls go directly to the relevant MCP server (no hub-and-spoke through local_router yet)
- local_router is reserved for future multi-agent coordination
- CodeT5+ model loading happens once on semantic_search server start (lazy init)
- DeepSeek API key is injected from `os.environ` at connection time, not hardcoded

## Error Handling
- MCP connection failures → retry logic in `_ensure_server()`
- Tool call failures → `_parse_tool_result()` returns `{"_error": ...}`
- Ranking server failure → fallback to local per-candidate scoring via `_score_pair()`
- Vault write failure → silently caught (non-critical)
