# Pipeline Patterns

## Standard Fix Pattern
```python
# Goal: fix a bug in a known file
# Pipeline: encode → plan → rank → validate → apply → log
task = "fix compute_sum: sum 0 to n-1, should be 1 to n"
file = "tests/buggy_math.py"
```

## Multi-Step Refinement Pattern
```python
# When loss > 0.01 after step 1, agent auto-refines:
# Step N task: "Continue fixing. Previous attempt: {description}"
# Re-reads patched file as new S_t, re-encodes, re-plans
```

## Candidate Generation Prompt Pattern
```
You are a code fix assistant. Given this code:
{code}

Task: {task}

Generate {k} candidate fixes. For each, provide:
1. change_description: what changed (Z_hat)
2. expected_code: full file after change (Z_actual)
3. description: one-line summary

Output as JSON array.
```

## MCP Connection Pattern
```python
# Standard MCP server connection lifecycle
params = StdioServerParameters(command=sys.executable, args=["-m", module], env=merged_env)
ctx = stdio_client(params)
streams = await ctx.__aenter__()
session = ClientSession(streams[0], streams[1])
await session.__aenter__()
await session.initialize()
result = await session.call_tool(tool_name, arguments=kwargs)
await session.__aexit__(None, None, None)
await ctx.__aexit__(None, None, None)
```

## Fallback Chain Pattern
```python
# When MCP server fails, fall back gracefully
try:
    ranking = await self._call("validators", "rank_candidates", ...)
except Exception:
    # Fallback: score each candidate locally
    for si in scored_inputs:
        loss = await self._score_pair(...)
    best_idx = int(np.argmin(losses_local))
```

## Vault Organization Pattern
```
vault/
├── rules/          # Immutable: architecture, pipeline spec, conventions
├── decisions/      # Append-only: why certain choices were made
├── lessons/        # Append-only: bugs fixed, workarounds, learned patterns
├── patterns/       # Reference: reusable code patterns and templates
└── _index.md       # Entry point with links to all sections
```
