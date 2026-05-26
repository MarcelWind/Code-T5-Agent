"""MCP Server: semantic_search — CodeT5+ embedding-based code search.

Provides tools for encoding code snippets, computing semantic similarity,
and searching workspace files by meaning rather than keywords.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
from mcp.server import FastMCP

# Lazy-import heavy dependencies only when needed
_code_encoder = None

server = FastMCP(
    "semantic_search",
    instructions="CodeT5+ embedding search. Encode code, find semantically similar snippets.",
)


def _get_encoder():
    """Lazy-init shared CodeEncoder instance."""
    global _code_encoder
    if _code_encoder is None:
        from core.encoder import CodeEncoder
        _code_encoder = CodeEncoder()
    return _code_encoder


def _walk_python_files(root: Path) -> list[Path]:
    """Recursively find all .py files in a directory."""
    files = []
    for path in root.rglob("*.py"):
        if "site-packages" not in path.parts and ".venv" not in path.parts:
            files.append(path)
    return files


@server.tool()
def encode_code(code: str) -> dict:
    """Encode a code string into an embedding vector.

    Args:
        code: Source code string to encode.

    Returns:
        dict with 'embedding' (list of floats), 'dimension', and 'shape'.
    """
    encoder = _get_encoder()
    emb = encoder.encode(code)
    return {
        "embedding": emb.tolist(),
        "dimension": int(emb.shape[0]),
        "shape": list(emb.shape),
    }


@server.tool()
def compute_similarity(code_a: str, code_b: str) -> dict:
    """Compute cosine similarity between two code snippets.

    Args:
        code_a: First code string.
        code_b: Second code string.

    Returns:
        dict with 'similarity' (0-1, higher = more similar) and 'distance'.
    """
    encoder = _get_encoder()
    emb_a = encoder.encode(code_a)
    emb_b = encoder.encode(code_b)

    a_norm = emb_a / (np.linalg.norm(emb_a) + 1e-12)
    b_norm = emb_b / (np.linalg.norm(emb_b) + 1e-12)

    similarity = float(np.dot(a_norm, b_norm))
    distance = float(1.0 - similarity)

    return {
        "similarity": round(similarity, 6),
        "distance": round(distance, 6),
    }


@server.tool()
def search_code(query: str, workspace_path: str = None, top_k: int = 5) -> list[dict]:
    """Semantically search workspace Python files for code matching a query.

    Args:
        query: Natural language description of what to find.
        workspace_path: Path to workspace root. Defaults to CWD.
        top_k: Number of top results to return (default: 5).

    Returns:
        List of result dicts with 'file', 'snippet', 'line', and 'score'.
    """
    encoder = _get_encoder()
    query_emb = encoder.encode(query)

    root = Path(workspace_path).resolve() if workspace_path else Path.cwd().resolve()
    py_files = _walk_python_files(root)

    if not py_files:
        return [{"message": "No Python files found in workspace."}]

    # Encode all files and find closest matches
    scored = []
    for fpath in py_files:
        try:
            content = fpath.read_text(encoding="utf-8")
            if not content.strip():
                continue
            # Split into meaningful chunks (top-level functions/classes)
            chunks = _chunk_code(content)
            for chunk_line, chunk_text in chunks:
                chunk_emb = encoder.encode(chunk_text)
                a_norm = query_emb / (np.linalg.norm(query_emb) + 1e-12)
                b_norm = chunk_emb / (np.linalg.norm(chunk_emb) + 1e-12)
                sim = float(np.dot(a_norm, b_norm))
                rel_path = fpath.relative_to(root) if root in fpath.parents else fpath.name
                scored.append({
                    "file": str(rel_path),
                    "snippet": chunk_text[:200],
                    "line": chunk_line,
                    "score": round(sim, 4),
                })
        except (OSError, UnicodeDecodeError):
            continue

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored[:top_k]


def _chunk_code(content: str) -> list[tuple[int, str]]:
    """Split Python code into top-level chunks (functions, classes, blocks)."""
    chunks = []
    lines = content.split("\n")
    current_chunk: list[str] = []
    current_line = 1
    start_line = 1

    for i, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith(("def ", "class ", "@", "async def ")):
            if current_chunk and len("\n".join(current_chunk)) > 20:
                chunks.append((start_line, "\n".join(current_chunk)))
            if current_chunk:
                current_chunk = []
            start_line = i + 1
        current_chunk.append(line)

    if current_chunk and len("\n".join(current_chunk)) > 20:
        chunks.append((start_line, "\n".join(current_chunk)))

    # If file is small enough, include whole file too
    if len(content) < 2000:
        chunks.insert(0, (1, content))

    return chunks


@server.tool()
def index_workspace(workspace_path: str = None) -> dict:
    """Pre-compute embeddings for all Python files in workspace.

    Args:
        workspace_path: Path to workspace root. Defaults to CWD.

    Returns:
        Summary dict with file count and status.
    """
    encoder = _get_encoder()
    root = Path(workspace_path).resolve() if workspace_path else Path.cwd().resolve()
    py_files = _walk_python_files(root)

    indexed = 0
    errors = 0
    for fpath in py_files:
        try:
            content = fpath.read_text(encoding="utf-8")
            if content.strip():
                encoder.encode(content)  # warm cache
                indexed += 1
        except Exception:
            errors += 1

    return {
        "files_found": len(py_files),
        "indexed": indexed,
        "errors": errors,
        "workspace": str(root),
    }


if __name__ == "__main__":
    server.run(transport="stdio")
