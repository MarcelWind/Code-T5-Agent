"""MCP Server: agent_tool — call the JEPA agent as a native MCP tool.

Designed for Pattern 3 integration: registers the full JEPA pipeline
(onboard, step, run) as callable MCP tools so that an external agent
(e.g. VS Code Copilot chat) can drive code changes through this server.

Usage from VS Code (`.vscode/mcp.json`):
  {
    "jepa-agent": {
      "command": "python",
      "args": ["-m", "mcp_servers.agent_tool.server"],
      "cwd": "${workspaceFolder}/jepa-agent"
    }
  }
"""

import asyncio
import os
import sys
import json
from typing import Optional

from mcp.server import FastMCP

# Eager import at module level — avoids blocking the event loop later
from agent import JEPAAgent

# Singleton agent instance (created on first access via _get_agent)
_agent: Optional[JEPAAgent] = None


server = FastMCP(
    "agent_tool",
    instructions="JEPA coding agent MCP wrapper. Exposes the full JEPA pipeline "
    "(onboard, step, run) as callable tools. Use onboard_project() first to "
    "initialize, then run_agent_task() for multi-step fixes, or step_agent() "
    "for single-step operations. Configurable candidate count, step limit, loss type.",
)


# ── Helpers ──

def _get_agent():
    """Lazy-init singleton JEPAAgent instance."""
    global _agent
    if _agent is None:
        _agent = JEPAAgent()
    return _agent


def _ensure_project_root(file_path: str) -> str:
    """Resolve an absolute path, defaulting to CWD if relative."""
    if os.path.isabs(file_path):
        return file_path
    return os.path.abspath(os.path.join(os.getcwd(), file_path))


def _return(result: dict, *, error: bool = False) -> dict:
    """Wrap result so it's JSON-serialisable."""
    return {"success": not error, "data": _serialise(result)}


def _serialise(obj):
    """Deep-convert known non-serialisable types to plain Python."""
    if isinstance(obj, dict):
        return {k: _serialise(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_serialise(v) for v in obj]
    if isinstance(obj, float):
        # Handle NaN / Infinity
        if obj != obj:  # NaN
            return None
        if obj == float("inf"):
            return "Infinity"
        if obj == float("-inf"):
            return "-Infinity"
    return obj


# ── Tools ──

@server.tool()
async def onboard_project(
    file_path: str,
    install_parsers: bool = False,
) -> dict:
    """Run project onboarding: detect root, languages, check parsers, build code index.

    Args:
        file_path: Path to any file in the target project (used to detect project root).
        install_parsers: If True, auto-install missing tree-sitter parsers.

    Returns:
        dict with status ('created' or 'loaded'), profile summary, and missing parsers.
    """
    agent = _get_agent()
    resolved = _ensure_project_root(file_path)
    try:
        result = await agent.onboard(resolved, install_parsers=install_parsers)
        return _return(result)
    except Exception as e:
        return _return({"error": str(e)}, error=True)


@server.tool()
async def step_agent(
    task: str,
    file_path: str,
    k: int = 5,
) -> dict:
    """Run a single JEPA step: encode -> plan -> rank -> patch -> test.

    Args:
        task: Description of the fix or change to make.
        file_path: Path to the file to modify.
        k: Number of candidate patches to generate (default: 5).

    Returns:
        dict with step result including jepa_loss, selected candidate, success.
    """
    agent = _get_agent()
    resolved = _ensure_project_root(file_path)
    try:
        result = await agent.step(task, resolved, k=k)
        return _return(result)
    except Exception as e:
        return _return({"error": str(e)}, error=True)


@server.tool()
async def run_agent_task(
    task: str,
    file_path: str,
    candidates: int = 5,
    steps: int = 3,
    loss: str = "cosine",
) -> dict:
    """Run the full multi-step JEPA loop: onboard (if needed) → step → repeat.

    Automatically runs onboarding if no project profile has been loaded yet.

    Args:
        task: Description of the fix or change to make.
        file_path: Path to the file to modify.
        candidates: Number of candidate patches per step (default: 5).
        steps: Maximum number of JEPA loop iterations (default: 3).
        loss: Loss type for JEPA scoring — 'cosine' | 'l2' (default: 'cosine').

    Returns:
        dict with 'steps' (list of per-step results) and 'converged' / 'failed'.
    """
    agent = _get_agent()
    resolved = _ensure_project_root(file_path)

    # Override config globals for this run
    import core.config as cfg
    old_k = cfg.NUM_CANDIDATES
    old_s = cfg.MAX_STEPS
    old_l = cfg.JEPA_LOSS_TYPE
    cfg.NUM_CANDIDATES = candidates
    cfg.MAX_STEPS = steps
    cfg.JEPA_LOSS_TYPE = loss

    try:
        step_results = await agent.run(task=task, file_path=resolved, max_steps=steps)
        # Summarise
        last = step_results[-1] if step_results else {}
        return _return({
            "steps": step_results,
            "total_steps": len(step_results),
            "converged": last.get("jepa_loss", 1.0) < 0.01 if last else False,
            "failed": last.get("error") if last else None,
            "last_success": last.get("success", False) if last else False,
        })
    except Exception as e:
        return _return({"error": str(e)}, error=True)
    finally:
        cfg.NUM_CANDIDATES = old_k
        cfg.MAX_STEPS = old_s
        cfg.JEPA_LOSS_TYPE = old_l


@server.tool()
async def get_agent_status() -> dict:
    """Get the current JEPA agent status: onboarded, project info, indexed files.

    Returns:
        dict with onboarded flag, project name, root, primary language, file count.
    """
    agent = _get_agent()
    profile = agent._project_profile or {}
    idx = agent._code_index or {}
    files = idx.get("files", {})
    return _return({
        "onboarded": agent._onboarded,
        "project_name": profile.get("project_name"),
        "project_root": str(agent._project_root or ""),
        "primary_language": profile.get("primary_language"),
        "indexed_files": len(files),
        "history_length": len(agent.history),
    })


@server.tool()
async def reindex_code(
    clear_first: bool = False,
) -> dict:
    """Rebuild the code symbol index from scratch.

    Uses the existing project profile to scan all source files.
    Call after files change or to pick up new extensions.

    Args:
        clear_first: If True, delete the existing index before rebuilding.

    Returns:
        dict with indexed count.
    """
    agent = _get_agent()
    try:
        count = await agent.reindex(clear_first=clear_first)
        return _return({"indexed": count})
    except Exception as e:
        return _return({"error": str(e)}, error=True)


@server.tool()
async def list_agent_tools() -> list[dict]:
    """List all available tools on this server with descriptions."""
    return [
        {"name": "onboard_project",
         "description": "Run project onboarding (detect root, languages, install parsers, build code index)."},
        {"name": "step_agent",
         "description": "Run a single JEPA step — encode current file, plan candidates, rank, patch, test."},
        {"name": "run_agent_task",
         "description": "Run the full multi-step JEPA loop. Auto-onboards if needed."},
        {"name": "get_agent_status",
         "description": "Get current agent state: onboarded, project name, indexed files."},
        {"name": "reindex_code",
         "description": "Rebuild the code symbol index from scratch using current project profile."},
        {"name": "list_agent_tools",
         "description": "List available tools on this server."},
    ]


if __name__ == "__main__":
    server.run(transport="stdio")
