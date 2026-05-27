"""MCP Server: code_understanding — AST-aware code analysis using tree-sitter.

Provides tools for parsing code, extracting functions/classes, finding symbols,
and analyzing import structure across multiple languages.

Supported languages: python, javascript, typescript, tsx, go, rust, java,
ruby, php, c, cpp. Detection is automatic via file extension, shebang,
or keyword heuristics. Missing parsers can be installed on demand.
"""

import os
from typing import Optional

from mcp.server import FastMCP
from tree_sitter import Node

from mcp_servers.code_understanding.language import (
    LANGUAGE_REGISTRY,
    detect_language,
    get_parser,
    get_installed_languages,
    installable_languages as get_installable_languages,
    install_language as do_install_language,
)

server = FastMCP(
    "code_understanding",
    instructions="AST-aware multi-language code analysis. Supports python, javascript, "
    "typescript, tsx, go, rust, java, ruby, php, c, cpp. Auto-detects language "
    "or accepts explicit 'language' parameter. Missing parsers installable on demand.",
)


# ── Language-specific AST node types ──

# Function definition node types per language
_FUNCTION_NODE_TYPES: dict[str, list[str]] = {
    "python": ["function_definition"],
    "javascript": ["function_declaration", "arrow_function", "method_definition"],
    "typescript": ["function_declaration", "arrow_function", "method_definition"],
    "tsx": ["function_declaration", "arrow_function", "method_definition"],
    "go": ["function_declaration", "method_declaration"],
    "rust": ["function_item"],
    "java": ["method_declaration"],
    "ruby": ["method"],
    "php": ["function_definition"],
    "c": ["function_definition"],
    "cpp": ["function_definition"],
}

# Class definition node types per language
_CLASS_NODE_TYPES: dict[str, list[str]] = {
    "python": ["class_definition"],
    "javascript": ["class_declaration"],
    "typescript": ["class_declaration"],
    "tsx": ["class_declaration"],
    "go": ["type_spec"],  # Go uses type Foo struct {}
    "rust": ["struct_item", "enum_item"],
    "java": ["class_declaration"],
    "ruby": ["class"],
    "php": ["class_declaration"],
    "c": ["struct_specifier"],
    "cpp": ["class_specifier"],
}

# Symbol types to search for find_symbol
_SYMBOL_NODE_TYPES: dict[str, list[str]] = {
    "python": ["function_definition", "class_definition", "assignment"],
    "javascript": ["function_declaration", "class_declaration", "variable_declarator"],
    "typescript": ["function_declaration", "class_declaration", "variable_declarator"],
    "tsx": ["function_declaration", "class_declaration", "variable_declarator"],
    "go": ["function_declaration", "type_spec", "var_spec"],
    "rust": ["function_item", "struct_item", "let_declaration"],
    "java": ["method_declaration", "class_declaration", "variable_declarator"],
    "ruby": ["method", "class", "assignment"],
    "php": ["function_definition", "class_declaration", "assignment"],
    "c": ["function_definition", "struct_specifier", "declaration"],
    "cpp": ["function_definition", "class_specifier", "declaration"],
}


# ── Language resolution helper ──


def _resolve_language(code: str, language: str, file_path: str | None = None) -> tuple[str | None, dict | None]:
    """Resolve 'auto' or explicit language to a parser and lang string.

    Returns:
        (lang_string, None) on success.
        (None, error_dict) on failure.
    """
    if language == "auto":
        detected = detect_language(code, file_path)
        lang = detected["language"]
        if not detected["installed"]:
            return None, {
                "error": f"Parser not installed for detected language '{lang}'",
                "detected": detected,
                "suggestion": f"Use install_language tool: install_language('{lang}')",
            }
    else:
        lang = language

    parser, err = get_parser(lang)
    if err:
        return None, err
    return lang, parser


# ── New MCP tools ──


@server.tool()
def detect(file_path: str = "", code: str = "") -> dict:
    """Detect programming language from file path or code content.

    Args:
        file_path: File path (extension used for detection).
        code: Code snippet (shebang/keywords used if no file_path).

    Returns:
        Dictionary with language, detected_by, confidence, installed, can_install, install_cmd.
    """
    return detect_language(code, file_path or None)


@server.tool()
def install_language(language: str) -> dict:
    """Install a tree-sitter parser package for a language.

    Args:
        language: Language name (e.g. 'javascript', 'go', 'rust').

    Returns:
        Dictionary with success status and message.
    """
    return do_install_language(language)


# ── AST walking helpers ──


def _walk_tree(node: Node, depth: int = 0) -> list[dict]:
    """Recursively walk AST node → list of dicts."""
    result = [{
        "type": node.type,
        "start": (node.start_point[0], node.start_point[1]),
        "end": (node.end_point[0], node.end_point[1]),
        "text": node.text.decode("utf-8", errors="replace") if depth < 3 else f"... ({node.type})",
        "children_count": node.child_count,
    }]
    for child in node.children:
        result.extend(_walk_tree(child, depth + 1))
    return result


def _find_nodes_of_type(node: Node, target_type: str) -> list[Node]:
    """Find all descendant nodes of a given type."""
    nodes = []
    if node.type == target_type:
        nodes.append(node)
    for child in node.children:
        nodes.extend(_find_nodes_of_type(child, target_type))
    return nodes


@server.tool()
def parse_code(code: str, language: str = "auto", detail: str = "overview") -> dict:
    """Parse code string into AST structure.

    Args:
        code: Source code string to parse.
        language: Language name ('auto', 'python', 'javascript', etc.).
        detail: 'overview' (default) for top-level nodes, 'full' for complete AST.

    Returns:
        dict with 'language', 'syntax_valid', 'ast', and optionally 'error'.
    """
    lang, parser_or_err = _resolve_language(code, language)
    if lang is None:
        return {"language": "unknown", "syntax_valid": False, "ast": [], **parser_or_err}
    tree = parser_or_err.parse(code.encode("utf-8"))

    if tree.root_node.has_error:
        return {
            "language": lang,
            "syntax_valid": False,
            "ast": [],
            "error": "Syntax errors detected in code.",
        }

    ast_nodes = _walk_tree(tree.root_node, depth=0 if detail == "full" else 3)

    return {
        "language": lang,
        "syntax_valid": True,
        "ast": ast_nodes[:200],  # cap at 200 nodes
    }


@server.tool()
def get_functions(code: str, language: str = "auto", include_body: bool = False) -> list[dict]:
    """Extract all function definitions from code.

    Args:
        code: Source code string.
        language: Language name ('auto', 'python', 'javascript', etc.).
        include_body: If True, include full function body text.

    Returns:
        List of function dicts with name, params, start_line, end_line, and optionally body.
    """
    lang, parser_or_err = _resolve_language(code, language)
    if lang is None:
        return [{"_error": parser_or_err.get("error", "Unknown error")}]
    tree = parser_or_err.parse(code.encode("utf-8"))

    node_types = _FUNCTION_NODE_TYPES.get(lang, ["function_definition", "function_declaration"])
    func_nodes = []
    for nt in node_types:
        func_nodes.extend(_find_nodes_of_type(tree.root_node, nt))

    functions = []
    for node in func_nodes:
        name_node = node.child_by_field_name("name")
        params_node = node.child_by_field_name("parameters")
        body_node = node.child_by_field_name("body")

        func = {
            "name": name_node.text.decode("utf-8") if name_node else "<anonymous>",
            "params": params_node.text.decode("utf-8") if params_node else "()",
            "start_line": node.start_point[0] + 1,
            "end_line": node.end_point[0] + 1,
        }

        # Extract decorators
        dec_nodes = _find_nodes_of_type(node, "decorator")
        if dec_nodes:
            func["decorators"] = [
                d.text.decode("utf-8") for d in dec_nodes
            ]

        if include_body and body_node:
            func["body"] = body_node.text.decode("utf-8")

        # Extract docstring
        if body_node and body_node.children:
            first = body_node.children[0]
            if first.type == "expression_statement" and first.children:
                child = first.children[0]
                if child.type in ("string", "string_content"):
                    func["docstring"] = child.text.decode("utf-8")

        functions.append(func)

    return functions


@server.tool()
def get_classes(code: str, language: str = "auto", include_methods: bool = False) -> list[dict]:
    """Extract all class definitions from code.

    Args:
        code: Source code string.
        language: Language name ('auto', 'python', 'javascript', etc.).
        include_methods: If True, include method details for each class.

    Returns:
        List of class dicts with name, bases, start_line, end_line, and optionally methods.
    """
    lang, parser_or_err = _resolve_language(code, language)
    if lang is None:
        return [{"_error": parser_or_err.get("error", "Unknown error")}]
    tree = parser_or_err.parse(code.encode("utf-8"))

    node_types = _CLASS_NODE_TYPES.get(lang, ["class_definition", "class_declaration"])
    class_nodes = []
    for nt in node_types:
        class_nodes.extend(_find_nodes_of_type(tree.root_node, nt))

    classes = []
    for node in class_nodes:
        name_node = node.child_by_field_name("name")
        bases_node = node.child_by_field_name("superclasses")
        body_node = node.child_by_field_name("body")

        cls = {
            "name": name_node.text.decode("utf-8") if name_node else "<anonymous>",
            "bases": bases_node.text.decode("utf-8") if bases_node else "",
            "start_line": node.start_point[0] + 1,
            "end_line": node.end_point[0] + 1,
        }

        if include_methods and body_node:
            methods = []
            for child in body_node.children:
                if child.type in ("function_definition", "method_definition"):
                    m_name = child.child_by_field_name("name")
                    m_params = child.child_by_field_name("parameters")
                    methods.append({
                        "name": m_name.text.decode("utf-8") if m_name else "<anonymous>",
                        "params": m_params.text.decode("utf-8") if m_params else "()",
                        "start_line": child.start_point[0] + 1,
                        "end_line": child.end_point[0] + 1,
                    })
            cls["methods"] = methods

        classes.append(cls)

    return classes


@server.tool()
def find_symbol(code: str, symbol_name: str, language: str = "auto") -> list[dict]:
    """Find all occurrences of a symbol (function, class, variable) in code.

    Args:
        code: Source code string.
        symbol_name: Name of the symbol to find.
        language: Language name ('auto', 'python', 'javascript', etc.).

    Returns:
        List of symbol occurrences with type, location, and context.
    """
    lang, parser_or_err = _resolve_language(code, language)
    if lang is None:
        return [{"_error": parser_or_err.get("error", "Unknown error")}]
    tree = parser_or_err.parse(code.encode("utf-8"))

    results = []
    lines = code.split("\n")

    def _search_node(node: Node):
        if node.type in ("identifier",):
            text = node.text.decode("utf-8")
            if text == symbol_name:
                line_idx = node.start_point[0]
                context_start = max(0, line_idx - 2)
                context_end = min(len(lines), line_idx + 3)
                results.append({
                    "type": node.type,
                    "line": line_idx + 1,
                    "col": node.start_point[1] + 1,
                    "context": "\n".join(lines[context_start:context_end]),
                })
        for child in node.children:
            _search_node(child)

    _search_node(tree.root_node)
    return results


@server.tool()
def get_imports(code: str, language: str = "auto") -> list[dict]:
    """Extract all import statements from code.

    Args:
        code: Source code string.
        language: Language name ('auto', 'python', 'javascript', etc.).

    Returns:
        List of import dicts with type (import/from), module, and names.
    """
    lang, parser_or_err = _resolve_language(code, language)
    if lang is None:
        return [{"_error": parser_or_err.get("error", "Unknown error")}]
    tree = parser_or_err.parse(code.encode("utf-8"))

    imports = []

    for node in _find_nodes_of_type(tree.root_node, "import_statement"):
        modules = []
        for child in node.children:
            if child.type == "dotted_name":
                modules.append(child.text.decode("utf-8"))
        imports.append({
            "type": "import",
            "module": modules[0] if modules else "",
            "names": modules,
        })

    for node in _find_nodes_of_type(tree.root_node, "import_from_statement"):
        module_node = node.child_by_field_name("module_name")
        names_node = node.child_by_field_name("name")

        names = []
        if names_node:
            for child in names_node.children:
                if child.type == "dotted_name":
                    names.append(child.text.decode("utf-8"))

        imports.append({
            "type": "from",
            "module": module_node.text.decode("utf-8") if module_node else "",
            "names": names,
        })

    return imports


if __name__ == "__main__":
    server.run(transport="stdio")
