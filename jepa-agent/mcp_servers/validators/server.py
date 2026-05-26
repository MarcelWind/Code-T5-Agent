"""MCP Server: validators — JEPA scoring and code validation.

Provides JEPA loss computation between predicted and actual code embeddings,
candidate ranking, syntax validation, and batch scoring.
"""

import ast
import sys
from typing import Optional

import numpy as np
from mcp.server import FastMCP

server = FastMCP(
    "validators",
    instructions="JEPA scoring and code validation. Compute loss, rank candidates, check syntax.",
)

# Lazy-import CodeEncoder only when needed
_code_encoder = None


def _get_encoder():
    global _code_encoder
    if _code_encoder is None:
        from core.encoder import CodeEncoder
        _code_encoder = CodeEncoder()
    return _code_encoder


# ── Loss functions ──


def _cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / (np.linalg.norm(a) + 1e-12)
    b_norm = b / (np.linalg.norm(b) + 1e-12)
    return float(1.0 - np.dot(a_norm, b_norm))


def _l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b))


# ── Tools ──


@server.tool()
def validate_code(
    predicted_description: str,
    actual_code: str,
    loss_type: str = "cosine",
) -> dict:
    """Compute JEPA loss between predicted embedding and actual code embedding.

    Args:
        predicted_description: Semantic description of the expected code change
            (this is the "predicted latent" Z_hat in JEPA terms).
        actual_code: The actual code string after the change (this becomes Z).
        loss_type: 'cosine' (default) or 'l2'.

    Returns:
        dict with 'loss', 'loss_type', and 'dimension'.
    """
    encoder = _get_encoder()

    Z_hat = encoder.encode(predicted_description)
    Z = encoder.encode(actual_code)

    if loss_type == "l2":
        loss = _l2_distance(Z_hat, Z)
    else:
        loss = _cosine_distance(Z_hat, Z)

    return {
        "loss": round(loss, 6),
        "loss_type": loss_type,
        "dimension": int(Z.shape[0]),
    }


@server.tool()
def rank_candidates(
    candidates: list,
    loss_type: str = "cosine",
) -> dict:
    """Rank candidate code patches by JEPA loss (lowest loss = best).

    Each candidate should have:
      - 'change_description' (used as predicted embedding Z_hat)
      - 'expected_code' (used as actual embedding Z)
      - 'description' (optional, short name)

    Args:
        candidates: List of candidate dicts.
        loss_type: 'cosine' (default) or 'l2'.

    Returns:
        dict with 'rankings' (ordered list of candidate indices),
        'losses' (parallel list of loss values), and 'best_idx'.
    """
    encoder = _get_encoder()

    scored = []
    for i, cand in enumerate(candidates):
        change_desc = cand.get("change_description", cand.get("description", ""))
        expected_code = cand.get("expected_code", "")

        if not change_desc or not expected_code:
            scored.append({"index": i, "loss": 999.0, "error": "Missing description or code"})
            continue

        Z_hat = encoder.encode(change_desc)
        Z = encoder.encode(expected_code)

        if loss_type == "l2":
            loss = _l2_distance(Z_hat, Z)
        else:
            loss = _cosine_distance(Z_hat, Z)

        scored.append({"index": i, "loss": round(loss, 6), "description": change_desc[:80]})

    # Sort by loss ascending
    scored.sort(key=lambda x: x["loss"])

    return {
        "rankings": [s["index"] for s in scored],
        "losses": [s["loss"] for s in scored],
        "best_idx": scored[0]["index"] if scored else -1,
        "best_loss": scored[0]["loss"] if scored else 0.0,
        "candidates": scored,
    }


@server.tool()
def validate_syntax(code: str, language: str = "python") -> dict:
    """Check Python code for syntax errors using AST parsing.

    Args:
        code: Source code string to validate.
        language: Language to validate (default: 'python', only python supported).

    Returns:
        dict with 'valid' (bool), 'error' (optional), and 'ast_type' (optional).
    """
    if language != "python":
        return {"valid": False, "error": f"Unsupported language: {language}"}

    try:
        tree = ast.parse(code)
        # Count top-level nodes for a quick summary
        node_count = sum(1 for _ in ast.walk(tree))
        classes = sum(1 for _ in ast.walk(tree) if isinstance(_, ast.ClassDef))
        functions = sum(1 for _ in ast.walk(tree) if isinstance(_, ast.FunctionDef))

        return {
            "valid": True,
            "ast_type": "Module",
            "node_count": node_count,
            "classes": classes,
            "functions": functions,
        }
    except SyntaxError as e:
        return {
            "valid": False,
            "error": {
                "message": e.msg,
                "line": e.lineno,
                "col": e.offset,
                "text": e.text,
            },
        }


@server.tool()
def batch_score(
    predicted_descriptions: list[str],
    actual_codes: list[str],
    loss_type: str = "cosine",
) -> dict:
    """Compute JEPA loss for multiple (prediction, code) pairs.

    Args:
        predicted_descriptions: List of semantic descriptions.
        actual_codes: List of actual code strings (same length).
        loss_type: 'cosine' (default) or 'l2'.

    Returns:
        dict with 'losses' (list of floats), 'mean', 'std', 'min', 'max'.
    """
    if len(predicted_descriptions) != len(actual_codes):
        return {"error": "Mismatched input lengths", "losses": []}

    encoder = _get_encoder()
    losses = []

    for desc, code in zip(predicted_descriptions, actual_codes):
        Z_hat = encoder.encode(desc)
        Z = encoder.encode(code)

        if loss_type == "l2":
            loss = _l2_distance(Z_hat, Z)
        else:
            loss = _cosine_distance(Z_hat, Z)
        losses.append(round(loss, 6))

    return {
        "losses": losses,
        "mean": round(float(np.mean(losses)), 6),
        "std": round(float(np.std(losses)), 6),
        "min": round(float(np.min(losses)), 6),
        "max": round(float(np.max(losses)), 6),
        "count": len(losses),
    }


if __name__ == "__main__":
    server.run(transport="stdio")
