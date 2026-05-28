"""MCP Server: local_router — central request dispatcher.

Routes incoming requests to the appropriate MCP server based on intent.
Maintains a registry of capabilities across all servers.
"""

from mcp.server import FastMCP

server = FastMCP(
    "local_router",
    instructions="Central JEPA agent router. Dispatches requests to appropriate MCP servers.",
)

# ── Capability registry ──

CAPABILITIES = {
    "code_understanding": {
        "description": "AST-aware code analysis using tree-sitter",
        "tools": ["parse_code", "get_functions", "get_classes", "find_symbol", "get_imports"],
    },
    "semantic_search": {
        "description": "CodeT5+ embedding search over workspace files",
        "tools": ["search_code", "encode_code", "compute_similarity", "index_workspace"],
    },
    "obsidian_brain": {
        "description": "Vault memory read/write for persistent knowledge",
        "tools": ["read_vault", "write_vault", "list_vault", "search_vault"],
    },
    "cloud_execution": {
        "description": "DeepSeek API proxy for cloud model execution",
        "tools": ["generate_code", "plan_actions", "chat_completion"],
    },
    "validators": {
        "description": "JEPA scoring and code validation",
        "tools": ["validate_code", "rank_candidates", "validate_syntax", "batch_score"],
    },
    "agent_tool": {
        "description": "JEPA agent wrapper exposing onboard/step/run as MCP tools for external agent integration",
        "tools": ["onboard_project", "step_agent", "run_agent_task", "get_agent_status", "reindex_code", "list_agent_tools"],
    },
}


@server.tool()
def list_capabilities() -> dict:
    """List all registered MCP servers and their available tools."""
    return CAPABILITIES


@server.tool()
def route_request(intent: str, payload: dict = None) -> dict:
    """Determine which server should handle a given intent.

    Args:
        intent: Natural language description of what the caller wants to do.
        payload: Optional context data to aid routing.

    Returns:
        dict with 'target_server', 'suggested_tool', and 'reasoning'.
    """
    intent_lower = intent.lower()
    payload = payload or {}

    # Routing rules — match on keywords
    if any(w in intent_lower for w in ["parse", "ast", "function", "class", "symbol", "import", "understand"]):
        return {
            "target_server": "code_understanding",
            "suggested_tool": _suggest_code_tool(intent_lower),
            "reasoning": "Request involves code structure analysis.",
        }

    if any(w in intent_lower for w in ["search", "find", "similar", "embed", "semantic", "index"]):
        return {
            "target_server": "semantic_search",
            "suggested_tool": _suggest_search_tool(intent_lower),
            "reasoning": "Request involves semantic search or embedding.",
        }

    if any(w in intent_lower for w in ["vault", "memory", "remember", "recall", "store", "lesson"]):
        return {
            "target_server": "obsidian_brain",
            "suggested_tool": _suggest_vault_tool(intent_lower),
            "reasoning": "Request involves persistent memory or vault access.",
        }

    if any(w in intent_lower for w in ["generate", "deepseek", "predict", "plan", "llm", "cloud", "gpt"]):
        return {
            "target_server": "cloud_execution",
            "suggested_tool": _suggest_cloud_tool(intent_lower),
            "reasoning": "Request requires cloud LLM execution.",
        }

    if any(w in intent_lower for w in ["score", "validate", "jepa", "loss", "rank", "syntax", "check"]):
        return {
            "target_server": "validators",
            "suggested_tool": _suggest_validator_tool(intent_lower),
            "reasoning": "Request involves validation or JEPA scoring.",
        }

    if any(w in intent_lower for w in ["onboard", "step", "run agent", "jepa loop", "tool", "integrate"]):
        return {
            "target_server": "agent_tool",
            "suggested_tool": _suggest_agent_tool(intent_lower),
            "reasoning": "Request involves the JEPA agent pipeline or external integration.",
        }

    return {
        "target_server": "local_router",
        "suggested_tool": "list_capabilities",
        "reasoning": "Could not determine intent. Use list_capabilities to discover available servers.",
    }


def _suggest_code_tool(intent: str) -> str:
    if "function" in intent:
        return "get_functions"
    if "class" in intent:
        return "get_classes"
    if "symbol" in intent:
        return "find_symbol"
    if "import" in intent:
        return "get_imports"
    return "parse_code"


def _suggest_search_tool(intent: str) -> str:
    if "index" in intent:
        return "index_workspace"
    if "similar" in intent or "compare" in intent:
        return "compute_similarity"
    if "embed" in intent:
        return "encode_code"
    return "search_code"


def _suggest_vault_tool(intent: str) -> str:
    if "write" in intent or "store" in intent or "save" in intent or "remember" in intent:
        return "write_vault"
    if "list" in intent or "structure" in intent:
        return "list_vault"
    if "search" in intent or "find" in intent:
        return "search_vault"
    return "read_vault"


def _suggest_cloud_tool(intent: str) -> str:
    if "plan" in intent:
        return "plan_actions"
    if "chat" in intent:
        return "chat_completion"
    return "generate_code"


def _suggest_validator_tool(intent: str) -> str:
    if "syntax" in intent:
        return "validate_syntax"
    if "batch" in intent or "rank" in intent:
        return "rank_candidates"
    if "score" in intent:
        return "batch_score"
    return "validate_code"


def _suggest_agent_tool(intent: str) -> str:
    if "onboard" in intent:
        return "onboard_project"
    if "reindex" in intent or "rebuild" in intent or "re-index" in intent:
        return "reindex_code"
    if "status" in intent or "state" in intent:
        return "get_agent_status"
    if "step" in intent:
        return "step_agent"
    if "list" in intent or "tool" in intent:
        return "list_agent_tools"
    return "run_agent_task"


if __name__ == "__main__":
    server.run(transport="stdio")
