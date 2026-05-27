"""Language registry, detection, lazy parser loading, and auto-install.

Supports: python, javascript, typescript, tsx, go, rust, java, ruby, php, c, cpp.
Each language entry maps to its PyPI package, import module, file extensions, and shebangs.
"""

import importlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

from tree_sitter import Language, Parser

# ── Registry ──

LANGUAGE_REGISTRY: dict[str, dict] = {
    "python": {
        "pkg": "tree-sitter-python",
        "module": "tree_sitter_python",
        "extensions": [".py", ".pyw", ".pyx"],
        "shebangs": ["python", "python3", "pypy"],
    },
    "javascript": {
        "pkg": "tree-sitter-javascript",
        "module": "tree_sitter_javascript",
        "extensions": [".js", ".mjs", ".cjs"],
        "shebangs": ["node", "deno", "bun"],
    },
    "typescript": {
        "pkg": "tree-sitter-typescript",
        "module": "tree_sitter_typescript",
        "extensions": [".ts"],
        "shebangs": [],
    },
    "tsx": {
        "pkg": "tree-sitter-typescript",
        "module": "tree_sitter_typescript",
        "extensions": [".tsx"],
        "shebangs": [],
        "_variant": "tsx",  # same package, different language() call
    },
    "go": {
        "pkg": "tree-sitter-go",
        "module": "tree_sitter_go",
        "extensions": [".go"],
        "shebangs": [],
    },
    "rust": {
        "pkg": "tree-sitter-rust",
        "module": "tree_sitter_rust",
        "extensions": [".rs"],
        "shebangs": [],
    },
    "java": {
        "pkg": "tree-sitter-java",
        "module": "tree_sitter_java",
        "extensions": [".java"],
        "shebangs": [],
    },
    "ruby": {
        "pkg": "tree-sitter-ruby",
        "module": "tree_sitter_ruby",
        "extensions": [".rb"],
        "shebangs": ["ruby"],
    },
    "php": {
        "pkg": "tree-sitter-php",
        "module": "tree_sitter_php",
        "extensions": [".php"],
        "shebangs": ["php"],
    },
    "c": {
        "pkg": "tree-sitter-c",
        "module": "tree_sitter_c",
        "extensions": [".c", ".h"],
        "shebangs": [],
    },
    "cpp": {
        "pkg": "tree-sitter-cpp",
        "module": "tree_sitter_cpp",
        "extensions": [".cc", ".cpp", ".cxx", ".hpp", ".hxx"],
        "shebangs": [],
    },
}

# ── Cache ──

_PARSER_CACHE: dict[str, Parser] = {}


# ── Language detection ──

def _build_extension_map() -> dict[str, str]:
    """Build extension → language lookup (longest extension first)."""
    mapping: dict[str, str] = {}
    for lang, info in LANGUAGE_REGISTRY.items():
        for ext in info["extensions"]:
            # Longer extensions take priority (.cxx vs .c)
            existing = mapping.get(ext)
            if existing is None or len(ext) > len(
                LANGUAGE_REGISTRY[existing]["extensions"][0]
            ):
                mapping[ext] = lang
    return mapping


_EXTENSION_MAP = _build_extension_map()

# Keyword heuristics for language detection when extension + shebang absent
_LANG_KEYWORD_HINTS: list[tuple[set[str], str]] = [
    ({"use strict", "require(", "module.exports", "import React", "=>"}, "javascript"),
    ({"package main", "func main(", "import (", "fmt."}, "go"),
    ({"fn main(", "fn ", "use std::", "let mut", "impl ", "pub "}, "rust"),
    ({"public class", "private class", "public static void main", "import java."}, "java"),
    ({"def ", "class ", "import ", "from ", "if __name__", ":\\s*pass$"}, "python"),
    ({"def ", "class ", "end$", "require ", "=>"}, "ruby"),
    ({"<?php", "namespace ", "use function"}, "php"),
]


def detect_language(code: str, file_path: str | None = None) -> dict:
    """Detect programming language from file path and/or code content.

    Priority:
      1. File extension (most reliable)
      2. Shebang line (#!/usr/bin/node)
      3. Keyword heuristics (last resort)

    Returns:
        dict with keys: language, detected_by, confidence, installed, can_install, install_cmd
    """
    # 1. Extension match
    if file_path:
        path = Path(file_path)
        ext = path.suffix.lower()
        # Handle .tar.gz style — only take last meaningful suffix
        if ext in _EXTENSION_MAP:
            lang = _EXTENSION_MAP[ext]
            return _result(lang, "extension", 0.95)

        # Try combined extension (.d.ts)
        for stem in [path.name, f".{path.stem.split('.')[-1]}"]:
            if stem in _EXTENSION_MAP:
                return _result(_EXTENSION_MAP[stem], "extension", 0.9)

    # 2. Shebang
    first_line = code.split("\n")[0].strip() if code else ""
    if first_line.startswith("#!"):
        for lang, info in LANGUAGE_REGISTRY.items():
            for interpreter in info.get("shebangs", []):
                if interpreter in first_line:
                    return _result(lang, "shebang", 0.9)

    # 3. Keyword heuristics
    scores: dict[str, int] = {}
    for keywords, lang in _LANG_KEYWORD_HINTS:
        for kw in keywords:
            if kw in code:
                scores[lang] = scores.get(lang, 0) + 1

    if scores:
        best = max(scores, key=scores.get)
        confidence = min(0.7, 0.3 + scores[best] * 0.1)
        return _result(best, "keywords", confidence)

    # 4. Fallback
    return _result("python", "fallback", 0.2)


def _result(language: str, detected_by: str, confidence: float) -> dict:
    """Build result dict with install info."""
    info = LANGUAGE_REGISTRY.get(language, {})
    installed = _is_module_installed(info.get("module", ""))
    return {
        "language": language,
        "detected_by": detected_by,
        "confidence": round(confidence, 3),
        "installed": installed,
        "can_install": not installed and bool(info),
        "install_cmd": f"pip install {info['pkg']}" if info and not installed else "",
    }


# ── Parser loading ──

def _is_module_installed(module_name: str) -> bool:
    """Check if a Python module is actually importable (not just a namespace pkg)."""
    if not module_name:
        return False
    try:
        spec = importlib.util.find_spec(module_name)
        if spec is None:
            return False
        # tree-sitter 0.25+ can create namespace packages — reject those
        if spec.origin and hasattr(spec.origin, "endswith"):
            return True
        return False
    except (ModuleNotFoundError, ValueError):
        return False


def get_parser(language: str) -> tuple[Parser | None, dict | None]:
    """Get a tree-sitter Parser for the given language.

    Returns:
        (parser, None) on success.
        (None, error_dict) on failure (not installed, unknown language).
    """
    if language in _PARSER_CACHE:
        return _PARSER_CACHE[language], None

    info = LANGUAGE_REGISTRY.get(language)
    if not info:
        return None, {"error": f"Unknown language: {language}", "known": list(LANGUAGE_REGISTRY.keys())}

    # Try import
    try:
        mod = importlib.import_module(info["module"])
    except ImportError as e:
        return None, {
            "error": f"Parser not installed for {language}",
            "detail": str(e),
            "can_install": True,
            "install_cmd": f"pip install {info['pkg']}",
        }

    # Get Language object — tree-sitter 0.25+ exports `.language()` returning PyCapsule,
    # which must be wrapped with `Language()` to pass to `Parser()`.
    if hasattr(mod, "language"):
        lang_obj = Language(mod.language())
    elif info.get("_variant"):
        # TSX: language_typescript() with variant
        ts_mod = importlib.import_module("tree_sitter_typescript")
        lang_obj = Language(ts_mod.language_typescript())
    else:
        return None, {"error": f"Cannot find language() in module {info['module']}"}

    parser = Parser(lang_obj)
    _PARSER_CACHE[language] = parser
    return parser, None


def install_language(language: str, timeout: int = 60) -> dict:
    """Install tree-sitter parser package for a language via pip.

    Returns:
        dict with success, message, and optionally error.
    """
    info = LANGUAGE_REGISTRY.get(language)
    if not info:
        return {"success": False, "error": f"Unknown language: {language}"}

    pkg = info["pkg"]
    try:
        r = subprocess.run(
            [sys.executable, "-m", "pip", "install", pkg],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if r.returncode != 0:
            return {
                "success": False,
                "error": r.stderr.strip() or r.stdout.strip(),
                "install_cmd": f"pip install {pkg}",
            }

        # Clear cache so next get_parser() re-imports
        _PARSER_CACHE.pop(language, None)
        return {"success": True, "message": f"Installed {pkg}"}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"pip install timed out after {timeout}s"}


def get_installed_languages() -> list[str]:
    """Return list of language names whose tree-sitter packages are importable."""
    installed = []
    for lang, info in LANGUAGE_REGISTRY.items():
        if _is_module_installed(info["module"]):
            installed.append(lang)
    return sorted(installed)


def installable_languages() -> list[str]:
    """Return list of language names available to install (not yet installed)."""
    installed = set(get_installed_languages())
    return sorted(set(LANGUAGE_REGISTRY.keys()) - installed)
