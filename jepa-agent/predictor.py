"""DeepSeek predictor — proposes candidate actions + predicts next code state.

In JEPA terms: this is the *predictor* that takes current state S_t and task,
and produces predicted next latent state Z_{t+1}_hat + candidate code patches.
"""

import json
import re
from openai import OpenAI

from config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    NUM_CANDIDATES,
)


def _extract_json(text: str) -> dict | list | None:
    """Best-effort JSON extraction from model output."""
    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Look for JSON block
    m = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Look for array/object boundaries
    for brace in ("[", "{"):
        start = text.find(brace)
        if start >= 0:
            try:
                return json.loads(text[start:])
            except json.JSONDecodeError:
                pass
    return None


SYSTEM_PROMPT = """You are a JEPA-style coding agent predictor.

Given current code and a task, propose {k} DISTINCT candidate patches.

Each candidate MUST be a JSON object with:
- "description": short explanation of the fix
- "change_description": paragraph describing the SEMANTIC effect of the change in plain English (what the code will do after the fix, not the code itself). This is used as the "latent prediction" for JEPA scoring.
- "expected_code": the COMPLETE code file AFTER applying this patch
- "diff": concise description of what lines change

Return a JSON array of {k} candidates.
"""


class DeepSeekPredictor:
    """Generates candidate action plans via DeepSeek Flash API."""

    def __init__(self):
        if not DEEPSEEK_API_KEY:
            raise ValueError(
                "DEEPSEEK_API_KEY not set. "
                "Export it as env var or set in config.py"
            )
        self.client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
        )

    def plan_actions(
        self,
        code_context: str,
        task: str,
        k: int = NUM_CANDIDATES,
    ) -> list[dict]:
        """Propose k candidate patches given current code + task."""
        user_prompt = f"""Task: {task}

Current code:
```python
{code_context}
```

Propose {k} distinct candidate patches as a JSON array."""

        resp = self.client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT.format(k=k)},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.8,
            response_format={"type": "json_object"},
        )

        text = resp.choices[0].message.content
        data = _extract_json(text)

        if isinstance(data, dict):
            # Try common wrapper keys
            for key in ("candidates", "patches", "actions", "results"):
                if key in data and isinstance(data[key], list):
                    return data[key][:k]
            return [data]  # single candidate wrapped in object

        if isinstance(data, list):
            return data[:k]

        # Fallback: return raw as single candidate
        return [{"description": text, "expected_code": code_context, "diff": text}]

    def generate_final_code(self, plan: dict) -> str:
        """Extract the final code from a candidate plan."""
        return plan.get("expected_code", "")
