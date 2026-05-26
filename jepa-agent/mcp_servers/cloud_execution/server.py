"""MCP Server: cloud_execution — DeepSeek API proxy.

Provides tools for generating code, planning candidate actions,
and running generic chat completions via the DeepSeek Flash API.
"""

import json
import os
from typing import Optional

from mcp.server import FastMCP

server = FastMCP(
    "cloud_execution",
    instructions="DeepSeek API proxy. Generate code, plan actions, chat completions.",
)

# Read API key from environment (set in .roo/mcp.json)
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"


def _get_client():
    """Lazy-init OpenAI-compatible client."""
    from openai import OpenAI
    if not DEEPSEEK_API_KEY:
        raise ValueError("DEEPSEEK_API_KEY not set in environment")
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url=DEEPSEEK_BASE_URL)


def _extract_json(text: str) -> dict | list | None:
    """Best-effort JSON extraction from model output."""
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    import re
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    for brace in ("[", "{"):
        start = text.find(brace)
        if start >= 0:
            try:
                return json.loads(text[start:])
            except json.JSONDecodeError:
                pass
    return None


CANDIDATE_SYSTEM_PROMPT = """You are a JEPA-style coding agent predictor.

Given current code and a task, propose {k} DISTINCT candidate patches.

Each candidate MUST be a JSON object with:
- "description": short explanation of the fix
- "change_description": paragraph describing the SEMANTIC effect of the change
- "expected_code": the COMPLETE code file AFTER applying this patch
- "diff": concise description of what lines change

Return a JSON array of {k} candidates.
"""


@server.tool()
def generate_code(prompt: str, system_prompt: str = None, temperature: float = 0.7) -> dict:
    """Generate code via DeepSeek Flash API.

    Args:
        prompt: The user prompt describing what code to generate.
        system_prompt: Optional system prompt override.
        temperature: Sampling temperature (default: 0.7).

    Returns:
        dict with 'content', 'model', 'usage', and 'finish_reason'.
    """
    client = _get_client()
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=messages,
        temperature=temperature,
    )

    choice = resp.choices[0]
    return {
        "content": choice.message.content,
        "model": resp.model,
        "usage": {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
        },
        "finish_reason": choice.finish_reason,
    }


@server.tool()
def plan_actions(code_context: str, task: str, k: int = 5) -> dict:
    """Propose candidate patches via DeepSeek given current code + task.

    Args:
        code_context: Current source code.
        task: Description of the change to make.
        k: Number of candidates to generate (default: 5).

    Returns:
        dict with 'candidates' (list of candidate dicts) and 'usage'.
    """
    client = _get_client()

    user_prompt = f"""Task: {task}

Current code:
```python
{code_context}
```

Propose {k} distinct candidate patches as a JSON array."""

    resp = client.chat.completions.create(
        model=DEEPSEEK_MODEL,
        messages=[
            {"role": "system", "content": CANDIDATE_SYSTEM_PROMPT.format(k=k)},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.8,
        response_format={"type": "json_object"},
    )

    text = resp.choices[0].message.content
    data = _extract_json(text)

    candidates = []
    if isinstance(data, dict):
        for key in ("candidates", "patches", "actions", "results"):
            if key in data and isinstance(data[key], list):
                candidates = data[key][:k]
                break
        else:
            candidates = [data]
    elif isinstance(data, list):
        candidates = data[:k]
    else:
        candidates = [{"description": text, "expected_code": code_context}]

    return {
        "candidates": candidates,
        "usage": {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
        },
    }


@server.tool()
def chat_completion(
    messages: list,
    temperature: float = 0.7,
    max_tokens: int = 2048,
    response_format: str = None,
) -> dict:
    """Generic chat completion via DeepSeek Flash.

    Args:
        messages: List of {"role": "..." , "content": "..."} dicts.
        temperature: Sampling temperature (default: 0.7).
        max_tokens: Maximum completion tokens (default: 2048).
        response_format: Optional format hint ("json_object" or None).

    Returns:
        dict with 'content', 'model', 'usage', and 'finish_reason'.
    """
    client = _get_client()
    kwargs = {
        "model": DEEPSEEK_MODEL,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format == "json_object":
        kwargs["response_format"] = {"type": "json_object"}

    resp = client.chat.completions.create(**kwargs)
    choice = resp.choices[0]

    return {
        "content": choice.message.content,
        "model": resp.model,
        "usage": {
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
            "total_tokens": resp.usage.total_tokens,
        },
        "finish_reason": choice.finish_reason,
    }


if __name__ == "__main__":
    server.run(transport="stdio")
