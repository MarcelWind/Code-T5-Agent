"""MCP Server: code_understanding — AST-aware code analysis using tree-sitter.

Provides tools for parsing code, extracting functions/classes, finding symbols,
and analyzing import structure across Python files.
"""

import os
from typing import Optional

from mcp.server import FastMCP

import tree_sitter_python as tspython
from tree_sitter import Language, Parser, Node

server = FastMCP(
    "code_understanding",
    instructions="AST-aware code analysis. Parse code, extract functions/classes, find symbols.",
)

# ── Initialize tree-sitter parser ──

PY_LANGUAGE = Language(tspython.language())
_parser = Parser(PY_LANGUAGE)


def _get_parser() -> Parser:
    """Lazy-init parser (reuse across calls)."""
    global _parser
    return _parser


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
def parse_code(code: str, detail: str = "overview") -> dict:
    """Parse code string into AST structure.

    Args:
        code: Source code string to parse.
        detail: 'overview' (default) for top-level nodes, 'full' for complete AST.

    Returns:
        dict with 'language', 'syntax_valid', 'ast', and optionally 'error'.
    """
    parser = _get_parser()
    tree = parser.parse(code.encode("utf-8"))

    if tree.root_node.has_error:
        return {
            "language": "python",
            "syntax_valid": False,
            "ast": [],
            "error": "Syntax errors detected in code.",
        }

    ast_nodes = _walk_tree(tree.root_node, depth=0 if detail == "full" else 3)

    return {
        "language": "python",
        "syntax_valid": True,
        "ast": ast_nodes[:200],  # cap at 200 nodes
    }


@server.tool()
def get_functions(code: str, include_body: bool = False) -> list[dict]:
    """Extract all function definitions from Python code.

    Args:
        code: Source code string.
        include_body: If True, include full function body text.

    Returns:
        List of function dicts with name, params, start_line, end_line, and optionally body.
    """
    parser = _get_parser()
    tree = parser.parse(code.encode("utf-8"))
    func_nodes = _find_nodes_of_type(tree.root_node, "function_definition")

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
def get_classes(code: str, include_methods: bool = False) -> list[dict]:
    """Extract all class definitions from Python code.

    Args:
        code: Source code string.
        include_methods: If True, include method details for each class.

    Returns:
        List of class dicts with name, bases, start_line, end_line, and optionally methods.
    """
    parser = _get_parser()
    tree = parser.parse(code.encode("utf-8"))
    class_nodes = _find_nodes_of_type(tree.root_node, "class_definition")

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
                if child.type == "function_definition":
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
def find_symbol(code: str, symbol_name: str) -> list[dict]:
    """Find all occurrences of a symbol (function, class, variable) in code.

    Args:
        code: Source code string.
        symbol_name: Name of the symbol to find.

    Returns:
        List of symbol occurrences with type, location, and context.
    """
    parser = _get_parser()
    tree = parser.parse(code.encode("utf-8"))

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
def get_imports(code: str) -> list[dict]:
    """Extract all import statements from Python code.

    Args:
        code: Source code string.

    Returns:
        List of import dicts with type (import/from), module, and names.
    """
    parser = _get_parser()
    tree = parser.parse(code.encode("utf-8"))

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
