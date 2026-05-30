# JEPA Coding Agent

A **self-bootstrapping** coding agent that uses a **JEPA** (Joint-Embedding Predictive Architecture) loop to autonomously understand, modify, and test code. Orchestrated via **6 MCP servers** with **DeepSeek Flash** for planning and **CodeT5+** for semantic embeddings.

> **Idea:** The agent reads a file, encodes it into an embedding, generates candidate patches via LLM, ranks them by JEPA loss, applies the best one, runs tests, and repeats until convergence.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        JEPAAgent                                 │
│                                                                  │
│  ┌──────────┐  ┌──────────────┐  ┌──────────────┐               │
│  │ Semantic  │  │    Cloud     │  │  Validators  │               │
│  │  Search   │  │  Execution   │  │              │               │
│  │(CodeT5+)  │  │ (DeepSeek)  │  │  (JEPA loss) │               │
│  └────┬─────┘  └──────┬───────┘  └──────┬───────┘               │
│       │               │                 │                        │
│  ┌────▼─────┐  ┌──────▼───────┐  ┌──────▼───────┐               │
│  │   Code   │  │  Obsidian    │  │    Local     │               │
│  │Understan.│  │   Brain      │  │   Router     │               │
│  │(tree-sit)│  │  (vault)     │  │              │               │
│  └──────────┘  └──────────────┘  └──────────────┘               │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Persistent Code Index (AST Cache)            │   │
│  │         vault/code-index/manifest.json                    │   │
│  │  Tracks symbols per file, mtime-based re-parse, diffing   │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### MCP Servers

| Server | Tool | Purpose |
|---|---|---|
| `code_understanding` | `parse_code`, `get_functions`, `get_classes`, `find_symbol`, `get_imports` | tree-sitter AST parsing (Python, JS, TS, Rust, Go, Java, Ruby, PHP, C, C++) |
| `semantic_search` | `encode_code` | CodeT5+ (220M) 768-dim embeddings, FAISS vector search |
| `cloud_execution` | `plan_actions` | DeepSeek Flash API — generates candidate patches (`patches[]` format) |
| `validators` | `validate_code`, `rank_candidates`, `validate_syntax` | JEPA loss computation, candidate ranking, syntax checks |
| `obsidian_brain` | `write_vault`, `read_vault`, `search_vault` | Persistent markdown vault for lessons, patterns, decisions |
| `context_builder` | `build_context` | Execution-neighborhood construction — compressed context from semantic matches + code index |
| `local_router` | — | Intent routing (future: multi-tool dispatch) |

### Core Modules (`core/`)

| Module | Role |
|---|---|
| `config.py` | All tunables (model names, dimensions, JEPA params) |
| `encoder.py` | `CodeEncoder` — wraps CodeT5+ for embedding extraction |
| `predictor.py` | `DeepSeekPredictor` — LLM patch generation |
| `scorer.py` | `jepa_loss`, `cosine_distance`, `l2_distance`, `rank_candidates` |
| `executor.py` | `read_file`, `write_file`, `apply_patch`, `apply_patches` (symbolic diffs), `run_command`, `Workspace` |
| `onboarding.py` | Project root detection, language detection, profile generation |
| `code_index.py` | Persistent tree-sitter AST cache, `resolve_symbol()` for line-range lookups |

### Data Flow (JEPA Step)

```
Observe ──→ Encode ──→ Predict ──→ Score ──→ Select ──→ Apply ──→ Test
  │            │           │           │          │          │        │
  │      CodeT5+      DeepSeek     JEPA       lowest     symbolic   pytest
  │      embedding    candidates   loss       loss       diffs      -q
  │                         │                              via
  │                    patches[]                        code index
  │                   (multi-file)                      line ranges
  └───────────────────── loop until convergence ───────────────────────┘
```

Each step also:
1. Loads the **tree-sitter code index** and builds a compressed execution neighborhood via the `context_builder` MCP server
2. Candidates are emitted as **`patches[]`** — symbolic diffs targeting specific functions/classes by name
3. The executor resolves each patch via `resolve_symbol()` and applies **bottom-up line-range swaps**
4. After patching, **updates the code index** for all changed files
5. **Runs `pytest`** and records pass/fail in the step result

---

## Project Structure

```
jepa-agent/
├── agent.py                 # JEPAAgent class + MCPConnection manager
├── main.py                  # CLI entry point (argparse)
├── requirements.txt         # Python dependencies
├── .gitignore
├── README.md
│
├── core/
│   ├── __init__.py          # Re-exports core modules
│   ├── config.py            # Constants (model names, dims, JEPA params)
│   ├── encoder.py           # CodeT5+ embedding encoder
│   ├── predictor.py         # DeepSeek LLM predictor
│   ├── scorer.py            # JEPA loss functions
│   ├── executor.py          # File ops, shell commands, Workspace
│   ├── onboarding.py        # Project/language detection
│   └── code_index.py        # Persistent tree-sitter AST cache
│
├── mcp_servers/
│   ├── code_understanding/  # tree-sitter AST server
│   ├── semantic_search/     # CodeT5+ embedding server
│   ├── cloud_execution/     # DeepSeek API server
│   ├── validators/          # JEPA scoring server
│   ├── obsidian_brain/      # Markdown vault server
│   └── local_router/        # Intent routing server
│
├── tests/
│   ├── test_config.py
│   ├── test_encoder.py
│   ├── test_executor.py
│   ├── test_scorer.py
│   ├── test_agent_helpers.py
│   └── buggy_math.py        # Sample buggy file for agent to fix
│
├── vault/                   # MCP-accessible markdown memory
│   ├── rules/               # Project conventions, profiles
│   ├── patterns/            # Reusable code patterns
│   ├── decisions/           # Architecture decision records
│   ├── lessons/             # Auto-logged step outcomes
│   └── code-index/          # Tree-sitter AST cache (regeneratable)
│
├── models/                  # Local HF model cache (gitignored)
└── .jepa-project.json       # Auto-generated project profile
```

---

## Setup

### Prerequisites

- **Python 3.11+**
- **DeepSeek API key** — set in `.env` at the project root:

```env
DEEPSEEK_API_KEY = "sk-..."
```

### Install

```bash
# Clone and enter the project
cd jepa-agent

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (macOS/Linux)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install mcp  # Model Context Protocol SDK
```

### Download Models (first run)

CodeT5+ and tree-sitter language parsers are downloaded on first use. You can also pre-download:

```bash
# CodeT5+ model (~860MB)
python -c "from transformers import AutoModel; AutoModel.from_pretrained('Salesforce/codet5p-220m')"
```

---

## Usage

### 1. Onboarding (project detection)

```bash
python main.py --init --file core/config.py
```

This detects the project root, languages, writes a profile to `.jepa-project.json` and the vault, and builds the **tree-sitter code index** (`vault/code-index/manifest.json`).

Add `--install-parsers` to auto-install missing tree-sitter language parsers:

```bash
python main.py --init --file core/config.py --install-parsers
```

### 2. Run the agent

```bash
python main.py --task "fix the buggy math function" --file tests/buggy_math.py
```

The agent will:
1. Onboard (if first run or no profile)
2. Encode the file with CodeT5+
3. Load AST context from the code index
4. Generate candidate patches via DeepSeek
5. Rank candidates by JEPA loss
6. Apply the best patch
7. Update the code index
8. Run `pytest tests/ -q`
9. Loop until convergence or max steps

### Options

| Flag | Short | Default | Description |
|---|---|---|---|
| `--task TEXT` | `-t` | — | Task description for the agent |
| `--file PATH` | `-f` | — | **Required.** File to modify |
| `--init` | — | — | Onboarding only (no agent loop) |
| `--install-parsers` | — | — | Auto-install missing tree-sitter parsers |
| `--candidates N` | `-k` | 5 | Candidate patches per step |
| `--steps N` | `-s` | 3 | Max JEPA loop steps |
| `--loss {cosine,l2}` | — | `cosine` | JEPA loss type |

### Examples

```bash
# Fix a specific file
python main.py --task "add input validation" --file core/executor.py -s 3

# Quick onboarding for a new project
python main.py --init --file src/main.py --install-parsers

# Single step debug
python main.py --task "refactor this" --file core/scorer.py -s 1 -k 10
```

---

## How It Works (JEPA Loop)

1. **Observe** — Read the target file
2. **Encode** — Convert code → 768-dim embedding via CodeT5+
3. **Context** — Build compressed execution neighborhood: semantic search → vault search → context_builder server (dependency expansion, role classification, budget enforcement)
4. **Predict** — LLM generates `k` candidate patches as **`patches[]`** — each targeting a specific function/class by name, across any number of files
5. **Score** — Each candidate is scored by JEPA loss (cosine/L2 between the `change_description` embedding and the concatenated patch bodies embedding)
6. **Select** — Pick the lowest-loss candidate
7. **Validate** — Syntax-check each patched file
8. **Apply** — Resolve each `{file, symbol}` via `resolve_symbol()` on the code index, perform **bottom-up line-range swaps** so earlier patches keep valid offsets
9. **Index** — Refresh the tree-sitter AST cache entry for **every** changed file
10. **Test** — Run `pytest` and record results
11. **Repeat** — Feed the result back as context for the next step, stop when loss < 0.01 or max steps reached

---

## Code Index (AST Cache)

The agent maintains a persistent symbol index at `vault/code-index/manifest.json`:

```json
{
  "files": {
    "core/executor.py": {
      "mtime": 1779928093.36,
      "language": "python",
      "size": 24677,
      "symbols": {
        "functions": [
          {"name": "read_file", "line": 10, "end_line": 16},
          {"name": "apply_patch", "line": 25, "end_line": 28},
          {"name": "apply_patches", "line": 30, "end_line": 85},
          {"name": "run_command", "line": 90, "end_line": 105}
        ],
        "classes": [{"name": "Workspace", "line": 110}],
        "imports": ["subprocess", "tempfile", "pathlib.Path"]
      }
    }
  }
}
```

- **On `step()`**: loads the index into the `context_builder` server for dependency expansion and region extraction
- **On `onboard()`**: builds the full index by walking source files
- **`resolve_symbol(index, rel_path, "function_name")`** → `{line, end_line}` — enables symbolic diffs by mapping symbol names to their current line ranges
- **After patching**: refreshes the entry for **every** changed file

This lets the agent resolve symbol names to line ranges for surgical patch
application, and know exactly what changed structurally between steps without
re-parsing the entire codebase.

---

## Config Reference (`core/config.py`)

| Constant | Default | Description |
|---|---|---|
| `DEEPSEEK_API_KEY` | `os.environ.get(...)` | DeepSeek Flash API key |
| `DEEPSEEK_BASE_URL` | `https://api.deepseek.com` | API endpoint |
| `DEEPSEEK_MODEL` | `deepseek-chat` | Model name |
| `CODET5_MODEL` | `Salesforce/codet5p-220m` | HF model for embeddings |
| `EMBEDDING_DIM` | `768` | Embedding dimension |
| `JEPA_LOSS_TYPE` | `cosine` | Loss function |
| `JEPA_TEMPERATURE` | `0.07` | Contrastive loss temperature |
| `NUM_CANDIDATES` | `5` | Candidates per step |
| `MAX_STEPS` | `5` | Max JEPA loop iterations |
| `MAX_CODE_TOKENS` | `512` | CodeT5+ max tokens |

---

## Tests

```bash
# Run all tests
pytest tests/ -q

# Run a specific test
pytest tests/test_config.py -v

# Quick smoke test
pytest tests/ -q --tb=short
```

---

## Self-Bootstrapping

The agent is designed to modify its **own source code**. The pipeline supports
multi-file refactoring via symbolic diffs: a single JEPA step can patch
multiple functions across multiple agent source files simultaneously.

In a successful end-to-end test, the agent:

1. Detected its own project structure via onboarding
2. Built a tree-sitter AST cache of all source files
3. Modified `core/config.py` to add a version string
4. Updated its own code index after the change

More advanced bootstrapping (e.g. adding a new MCP server) works by emitting
a `patches[]` array targeting different files in one step — the executor
resolves them via the code index and applies bottom-up.

See [`docs/patches-architecture.md`](../docs/patches-architecture.md) for details.

---

## License

MIT
