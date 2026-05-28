"""Debug: inspect AST node structure to understand why extraction returns 0."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ["PYTHONIOENCODING"] = "utf-8"

from core.executor import read_file
from mcp_servers.code_understanding.language import detect_language, get_parser
from mcp_servers.code_understanding.server import _walk_tree

code = read_file(os.path.join(os.path.dirname(__file__), "agent.py"))
info = detect_language(code, file_path="agent.py")
print("Detected:", info["language"])

parser = get_parser(info["language"])[0]
tree = parser.parse(code.encode("utf-8"))
nodes = _walk_tree(tree.root_node, depth=0)
print(f"Total AST nodes: {len(nodes)}")

# Show all unique node types
types = {}
for n in nodes:
    t = n["type"]
    types.setdefault(t, 0)
    types[t] += 1
print("Node types:")
for t, c in sorted(types.items(), key=lambda x: -x[1]):
    print(f"  {t}: {c}")

# Show first 20 nodes with text preview
print("\nFirst 20 nodes:")
for i, n in enumerate(nodes[:20]):
    print(f"  [{i}] type={n['type']:30s} line={n['start'][0]:3d}  text={n['text'][:60]}")

# Now test extract_from_ast with full detail
from core.code_index import extract_from_ast
ast_data = {"language": "python", "syntax_valid": True, "ast": nodes[:200]}
syms = extract_from_ast(ast_data, source_code=code)
funcs = syms.get("functions", [])
imports = syms.get("imports", [])
print(f"\nExtracted: {len(funcs)} functions, {len(imports)} imports")
for f in funcs[:5]:
    print(f"  func: {f['name']} @ line {f['line']}")
for i in imports[:4]:
    print(f"  import: {i}")
