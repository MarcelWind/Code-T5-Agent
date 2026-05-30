"""JEPA agent — MCP-orchestrated pipeline.

Replaces direct core imports with calls to 6 MCP servers:
  local_router      → intent routing
  code_understanding → AST analysis (tree-sitter)
  semantic_search   → CodeT5+ embeddings
  obsidian_brain    → vault memory
  cloud_execution   → DeepSeek API
  validators        → JEPA scoring
"""

import asyncio
import json
import os
import sys
from typing import Any

import numpy as np

from mcp import ClientSession
from mcp.client.stdio import stdio_client, StdioServerParameters

# ── Load .env from project root (one level up from this file) ──
import dotenv
_dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
if os.path.isfile(_dotenv_path):
    dotenv.load_dotenv(_dotenv_path)

from core.executor import read_file, apply_patch, apply_patches, run_command
from core.config import (
    NUM_CANDIDATES,
    MAX_STEPS,
    JEPA_LOSS_TYPE,
    CONTEXT_BUDGET_TOKENS,
    CONTEXT_EXPANSION_HOPS,
    CONTEXT_MAX_SEED_FILES,
    CONTEXT_INCLUDE_MEMORY,
)
from core.onboarding import (
    detect_project_root,
    detect_project_languages,
    generate_project_profile,
    load_project_profile,
    save_project_profile,
    format_profile_markdown,
)
from core.code_index import (
    get_index,
    save_index,
)

# ── Extension → tree-sitter language mapping ──
_ext_to_lang = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
}

# ── Helpers ──

def _parse_tool_result(result) -> Any:
    """Extract Python object from MCP CallToolResult."""
    if result.isError:
        text = result.content[0].text if result.content else "Unknown error"
        return {"_error": text}
    texts = []
    for c in result.content:
        if hasattr(c, "text"):
            texts.append(c.text)
    combined = "\n".join(texts)
    if not combined:
        return {}
    # First: try parsing the combined text as JSON (single object or array)
    try:
        return json.loads(combined)
    except (json.JSONDecodeError, TypeError):
        pass
    # Second: try parsing each text item individually and collect as a list
    # (handles FastMCP returning one TextContent per list element)
    if len(texts) > 1:
        items = []
        for t in texts:
            try:
                items.append(json.loads(t))
            except (json.JSONDecodeError, TypeError):
                items.append(t)
        return items
    return combined

# ── MCP Connection Manager ──

SERVER_DEFS = {
    "local_router": {
        "module": "mcp_servers.local_router.server",
        "args": [],
        "env": {"PYTHONPATH": "."},
    },
    "code_understanding": {
        "module": "mcp_servers.code_understanding.server",
        "args": [],
        "env": {"PYTHONPATH": ".", "HF_HOME": "models", "HF_HUB_CACHE": "models/hub"},
    },
    "semantic_search": {
        "module": "mcp_servers.semantic_search.server",
        "args": [],
        "env": {"PYTHONPATH": ".", "HF_HOME": "models", "HF_HUB_CACHE": "models/hub"},
    },
    "obsidian_brain": {
        "module": "mcp_servers.obsidian_brain.server",
        "args": [],
        "env": {"PYTHONPATH": ".", "VAULT_PATH": "vault"},
    },
    "cloud_execution": {
        "module": "mcp_servers.cloud_execution.server",
        "args": [],
        "env": {"PYTHONPATH": ".", "DEEPSEEK_API_KEY": os.environ.get("DEEPSEEK_API_KEY", "")},
    },
    "validators": {
        "module": "mcp_servers.validators.server",
        "args": [],
        "env": {"PYTHONPATH": ".", "HF_HOME": "models", "HF_HUB_CACHE": "models/hub"},
    },
    "context_builder": {
        "module": "mcp_servers.context_builder.server",
        "args": [],
        "env": {"PYTHONPATH": "."},
    },
}

# ── Log helper: writes to stderr so MCP stdout transport stays clean ──
_log = lambda *args, **kwargs: print(*args, file=sys.stderr, **kwargs)

class MCPConnection:
    """A single stdio MCP server connection."""

    def __init__(self, name: str, module: str, args: list[str] | None = None, env: dict | None = None, cwd: str | None = None):
        self.name = name
        base_env = {**os.environ, **{"PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}, **(env or {})}
        self.params = StdioServerParameters(
            command=sys.executable,
            args=["-m", module] + (args or []),
            env=base_env,
            cwd=cwd,
        )
        self.session: ClientSession | None = None
        self._streams = None
        self._client_ctx = None

    async def connect(self):
        ctx = stdio_client(self.params)
        self._client_ctx = ctx
        try:
            streams = await ctx.__aenter__()
            read_stream, write_stream = streams
            session = ClientSession(read_stream, write_stream)
            await session.__aenter__()
            await session.initialize()
            self.session = session
            _log(f"  [mcp] connected -> {self.name}")
        except BaseException:
            # If connection fails mid-init, clean up any partial state
            if self.session:
                try:
                    await self.session.__aexit__(None, None, None)
                except BaseException:
                    pass
                self.session = None
            if self._client_ctx:
                try:
                    await self._client_ctx.__aexit__(None, None, None)
                except BaseException:
                    pass
                self._client_ctx = None
            raise

    async def disconnect(self):
        if self.session:
            try:
                await self.session.__aexit__(None, None, None)
            except BaseException:
                pass
            self.session = None
        if self._client_ctx:
            try:
                await self._client_ctx.__aexit__(None, None, None)
            except BaseException:
                pass
            self._client_ctx = None

    async def call(self, tool: str, timeout: float = 120.0, **kwargs) -> Any:
        if not self.session:
            await self.connect()
        try:
            result = await asyncio.wait_for(
                self.session.call_tool(tool, arguments=kwargs or None),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"MCP call '{self.name}.{tool}()' timed out after {timeout}s. "
                f"The sub-server may be stuck or unresponsive."
            )
        return _parse_tool_result(result)


class JEPAAgent:
    """MCP-orchestrated JEPA coding agent."""

    def __init__(self):
        self.history: list[dict] = []
        self._servers: dict[str, MCPConnection] = {}
        self._project_root: str | None = None
        self._project_profile: dict | None = None
        self._onboarded: bool = False
        self._code_index: dict | None = None
        _log("[JEPAAgent] MCP-orchestrated agent initialized")

    async def _ensure_server(self, name: str):
        if name not in self._servers:
            cfg = SERVER_DEFS[name]
            # Sub-servers need cwd = directory containing mcp_servers/ package
            cwd = os.path.dirname(os.path.abspath(__file__))
            conn = MCPConnection(name=name, module=cfg["module"], args=cfg.get("args", []), env=cfg.get("env"), cwd=cwd)
            try:
                await conn.connect()
            except BaseException:
                # On any failure (including CancelledError), make sure we don't leak server entry
                raise
            self._servers[name] = conn

    async def _call(self, server: str, tool: str, **kwargs) -> Any:
        await self._ensure_server(server)
        return await self._servers[server].call(tool, **kwargs)

    async def _encode(self, code: str) -> list[float]:
        result = await self._call("semantic_search", "encode_code", code=code)
        if isinstance(result, dict) and "embedding" in result:
            return result["embedding"]
        raise ValueError(f"encode_code failed: {result}")

    async def _plan_actions(self, code: str, task: str, k: int = 5, context_package: str = "") -> list[dict]:
        kwargs = {"code_context": code, "task": task, "k": k}
        if context_package:
            kwargs["context_package"] = context_package
        result = await self._call("cloud_execution", "plan_actions", **kwargs)
        if isinstance(result, dict) and "candidates" in result:
            return result["candidates"]
        if isinstance(result, list):
            return result
        raise ValueError(f"plan_actions failed: {result}")

    async def _score_pair(self, desc: str, code: str, loss_type: str = "cosine") -> float:
        result = await self._call("validators", "validate_code", predicted_description=desc, actual_code=code, loss_type=loss_type)
        if isinstance(result, dict) and "loss" in result:
            return result["loss"]
        raise ValueError(f"validate_code failed: {result}")

    async def _rank(self, candidates: list, loss_type: str = "cosine") -> dict:
        result = await self._call("validators", "rank_candidates", candidates=candidates, loss_type=loss_type)
        if isinstance(result, dict) and "rankings" in result:
            return result
        raise ValueError(f"rank_candidates failed: {result}")

    async def _validate_syntax(self, code: str) -> dict:
        return await self._call("validators", "validate_syntax", code=code, language="python")

    async def _vault_write(self, path: str, content: str, overwrite: bool = False) -> dict:
        return await self._call("obsidian_brain", "write_vault", path=path, content=content, overwrite=overwrite)

    async def _vault_search(self, query: str) -> list[dict]:
        """Search vault memory for context-relevant rules/patterns."""
        try:
            result = await self._call("obsidian_brain", "search_vault", query=query)
            if isinstance(result, list):
                return result
            if isinstance(result, dict) and "_error" not in result:
                return [result]
            return []
        except Exception:
            return []

    async def _semantic_search(self, query: str, top_k: int = 5) -> list[dict]:
        """Search workspace via CodeT5+ semantic embedding."""
        try:
            result = await self._call(
                "semantic_search", "search_code",
                query=query,
                workspace_path=self._project_root or "",
                top_k=top_k,
            )
            if isinstance(result, list):
                return result
            return []
        except Exception:
            return []

    async def _build_execution_context(
        self,
        task: str,
        file_path: str,
    ) -> dict:
        """Build a compressed execution neighborhood for the task.

        Pipeline:
          1. Semantic search → seed file matches
          2. Vault search → relevant memory rules
          3. Context builder → compressed package with budget enforcement

        Returns:
            dict with patch_targets, dependency_summaries, memory_rules,
            excluded_files, estimated_tokens, expansion_stats.
            Empty dict if context_builder server fails.
        """
        _log(f"  [ctx] building execution neighborhood for task...")

        # Step 1: Semantic search for seed files
        semantic_matches = await self._semantic_search(task, top_k=CONTEXT_MAX_SEED_FILES)
        if semantic_matches:
            _log(f"  [ctx] semantic search: {len(semantic_matches)} matches")
            for m in semantic_matches[:3]:
                fp = m.get("file", "") if isinstance(m, dict) else str(m)
                sc = m.get("score", "") if isinstance(m, dict) else ""
                _log(f"    - {fp} (score={sc})")
        else:
            _log(f"  [ctx] no semantic matches, using file imports as seeds")

        # Step 2: Vault search for memory rules
        memory_hits = []
        if CONTEXT_INCLUDE_MEMORY:
            memory_hits = await self._vault_search(task)
            if memory_hits:
                _log(f"  [ctx] vault memory: {len(memory_hits)} hits")

        # Step 3: Call context_builder server
        try:
            code_index = await self._ensure_code_index()
            result = await self._call(
                "context_builder", "build_context",
                task=task,
                file_path=file_path,
                semantic_matches=semantic_matches,
                code_index=code_index,
                project_root=self._project_root or "",
                memory_hits=memory_hits,
                token_budget=CONTEXT_BUDGET_TOKENS,
                expansion_hops=CONTEXT_EXPANSION_HOPS,
            )
            if isinstance(result, dict) and "patch_targets" in result:
                stats = result.get("expansion_stats", {})
                _log(f"  [ctx] package: {stats.get('included', 0)} included, "
                     f"{stats.get('excluded', 0)} excluded, "
                     f"{stats.get('total_candidates', 0)} candidates, "
                     f"~{result.get('estimated_tokens', 0)} tokens")
                return result
            _log(f"  [ctx] unexpected response: {type(result)}")
            return {}
        except Exception as e:
            _log(f"  [ctx] ! context builder failed: {e}")
            return {}

    # ── Code Index (tree-sitter AST cache) ──

    async def _ensure_code_index(self) -> dict:
        """Load index from disk, or init empty if first time."""
        if self._code_index is None:
            self._code_index = get_index()
        return self._code_index

    async def _parse_ast(self, file_path: str, language: str = "python") -> dict | None:
        """Parse a file via code_understanding server, return AST data or None."""
        code = read_file(file_path)
        if not code:
            return None
        try:
            await self._ensure_server("code_understanding")
            return await self._call("code_understanding", "parse_code", code=code, language=language, detail="full")
        except (Exception, asyncio.CancelledError) as e:
            _log(f"  [index] ! parse failed for {file_path}: {e}")
            return None

    async def _get_file_symbols(self, file_path: str, language: str = "python") -> dict:
        """Extract functions, classes, and imports using targeted tools (no 200-node cap).

        Returns:
            {"functions": [...], "classes": [...], "imports": [...]}
        """
        code = read_file(file_path)
        if not code:
            return {"functions": [], "classes": [], "imports": []}

        await self._ensure_server("code_understanding")
        symbols = {"functions": [], "classes": [], "imports": []}

        try:
            funcs = await self._call("code_understanding", "get_functions", code=code, language=language, include_body=False)
            if isinstance(funcs, list):
                for f in funcs:
                    if isinstance(f, dict) and "name" in f:
                        symbols["functions"].append({
                            "name": f["name"],
                            "line": f.get("start_line", 0),
                            "end_line": f.get("end_line", 0),
                        })
        except (Exception, asyncio.CancelledError) as e:
            _log(f"  [index] ! get_functions failed: {e}")

        try:
            classes = await self._call("code_understanding", "get_classes", code=code, language=language)
            if isinstance(classes, list):
                for c in classes:
                    if isinstance(c, dict) and "name" in c:
                        symbols["classes"].append({
                            "name": c["name"],
                            "line": c.get("start_line", 0),
                        })
        except (Exception, asyncio.CancelledError) as e:
            _log(f"  [index] ! get_classes failed: {e}")

        try:
            imports = await self._call("code_understanding", "get_imports", code=code, language=language)
            if isinstance(imports, list):
                for imp in imports:
                    if isinstance(imp, str):
                        symbols["imports"].append(imp)
                    elif isinstance(imp, dict):
                        symbols["imports"].append(imp.get("statement", str(imp)))
        except (Exception, asyncio.CancelledError) as e:
            _log(f"  [index] ! get_imports failed: {e}")

        return symbols

    async def _update_index_entry(self, file_path: str, language: str = "python") -> dict | None:
        """Parse a file and update its entry in the index. Returns old entry if changed."""
        idx = await self._ensure_code_index()
        rel_path = os.path.relpath(file_path, self._project_root) if self._project_root else file_path
        old_entry = idx.get("files", {}).get(rel_path)

        code = read_file(file_path)
        if not code:
            return old_entry
        syms = await self._get_file_symbols(file_path, language)
        new_entry = {
            "mtime": os.path.getmtime(file_path),
            "language": language,
            "size": os.path.getsize(file_path),
            "symbols": syms,
        }
        idx.setdefault("files", {})[rel_path] = new_entry
        save_index(idx)
        return old_entry

    async def _build_code_index(self, extensions: list[str]) -> int:
        """Scan project source files and build the full code index. Returns file count."""
        project_root = self._project_root
        if not project_root:
            return 0

        ext_set = set(extensions)
        scanned = 0

        for dirpath, dirnames, filenames in os.walk(project_root):
            dirnames[:] = [d for d in dirnames
                           if not d.startswith(".")
                           and d not in ("__pycache__", "node_modules", ".venv", "venv", "env", "models")]

            for fn in filenames:
                ext = os.path.splitext(fn)[1].lower()
                if ext not in ext_set:
                    continue

                full_path = os.path.join(dirpath, fn)
                rel_path = os.path.relpath(full_path, project_root)

                # Determine language from extension
                lang = _ext_to_lang.get(ext, "python")

                try:
                    file_code = read_file(full_path)
                    if not file_code:
                        continue
                    syms = await self._get_file_symbols(full_path, language=lang)
                    idx = await self._ensure_code_index()
                    idx.setdefault("files", {})[rel_path] = {
                        "mtime": os.path.getmtime(full_path),
                        "language": lang,
                        "size": os.path.getsize(full_path),
                        "symbols": syms,
                    }
                    scanned += 1
                except (Exception, asyncio.CancelledError):
                    continue

        save_index(self._code_index)
        return scanned

    async def step(self, task: str, file_path: str, k: int = NUM_CANDIDATES) -> dict:
        """Single JEPA step: observe -> predict -> score -> execute (async MCP)."""
        code = read_file(file_path)
        if not code:
            return {"error": f"File not found or empty: {file_path}"}

        try:
            S_t = await self._encode(code)
            _log(f"  [step] encoded current state -> dim={len(S_t)}")
        except Exception as e:
            return {"error": f"Encoding failed: {e}"}

        # ── Build compressed execution neighborhood ──
        context_package = await self._build_execution_context(task, file_path)
        context_json = json.dumps(context_package) if context_package else ""

        try:
            candidates = await self._plan_actions(code, task, k=k, context_package=context_json)
            _log(f"  [step] generated {len(candidates)} candidates")
        except Exception as e:
            return {"error": f"Planning failed: {e}"}

        if not candidates:
            return {"error": "No candidates generated", "state_embedding": S_t}

        # ── Build scored inputs (handle both patches[] and legacy expected_code) ──
        scored_inputs = []
        for cand in candidates:
            desc = cand.get("change_description", cand.get("description", ""))
            patches = cand.get("patches")
            exp_code = cand.get("expected_code", "")
            if desc and patches:
                # Aggregate all new_body snippets for scoring
                code_for_scoring = "\n\n".join(
                    p.get("new_body", "") for p in patches if isinstance(p, dict)
                )
                if code_for_scoring:
                    scored_inputs.append({
                        "change_description": desc,
                        "expected_code": code_for_scoring,
                        "description": cand.get("description", "")[:60],
                        "_has_patches": True,
                        "_patches": patches,
                    })
            elif desc and exp_code:
                scored_inputs.append({
                    "change_description": desc,
                    "expected_code": exp_code,
                    "description": cand.get("description", "")[:60],
                    "_has_patches": False,
                })

        if not scored_inputs:
            for cand in candidates:
                patches = cand.get("patches")
                exp_code = cand.get("expected_code", "")
                desc = cand.get("change_description", cand.get("description", ""))
                if patches:
                    code_str = "\n\n".join(p.get("new_body", "") for p in patches if isinstance(p, dict))
                    scored_inputs.append({
                        "change_description": desc or task,
                        "expected_code": code_str or code,
                        "description": cand.get("description", "")[:60],
                        "_has_patches": True,
                        "_patches": patches,
                    })
                else:
                    scored_inputs.append({
                        "change_description": desc or task,
                        "expected_code": exp_code or code,
                        "description": cand.get("description", "")[:60],
                        "_has_patches": False,
                    })

        try:
            ranking = await self._rank(scored_inputs, loss_type=JEPA_LOSS_TYPE)
            best_loss = ranking["best_loss"]
            losses = ranking["losses"]
            rankings = ranking["rankings"]
            # Map back to original indices
            desc_to_orig = {}
            for i, cand in enumerate(candidates):
                d = cand.get("change_description", cand.get("description", ""))
                desc_to_orig[d] = i
            orig_rankings = []
            for r in rankings:
                si = scored_inputs[r]
                orig_idx = desc_to_orig.get(si["change_description"], r)
                orig_rankings.append(orig_idx)
            best_idx = orig_rankings[0] if orig_rankings else 0
        except Exception as e:
            _log(f"  [step] ranking server failed ({e}), scoring locally...")
            losses_local = []
            for si in scored_inputs:
                loss = await self._score_pair(si["change_description"], si["expected_code"], JEPA_LOSS_TYPE)
                losses_local.append(loss)
            if losses_local:
                best_idx = int(np.argmin(losses_local))
                best_loss = losses_local[best_idx]
                losses = losses_local
            else:
                best_idx = 0
                best_loss = 1.0
                losses = [1.0]

        for i, cand in enumerate(candidates):
            desc = cand.get("description", "")[:50]
            loss_str = f" loss={losses[i]:.4f}" if i < len(losses) else ""
            _log(f"  [step]   candidate {i}: {desc}...{loss_str}")

        best_candidate = candidates[best_idx]
        best_si = scored_inputs[0]  # corresponds to best_idx via orig_rankings[0]
        _log(f"  [step] selected candidate {best_idx} (loss={best_loss:.4f})")

        # ── Apply patches (symbol-diff) or fall back to full-file patch ──
        changed_files: list[str] = []
        syntax_valid = True
        best_patches = best_candidate.get("patches")
        if best_patches:
            _log(f"  [step] applying {len(best_patches)} symbolic patches...")
            success, changed_files = apply_patches(
                best_patches,
                await self._ensure_code_index(),
                project_root=self._project_root or "",
            )
            if success:
                _log(f"  [step] patched files: {changed_files}")
                # Validate syntax of each changed file
                for rel_path in changed_files:
                    full = os.path.join(self._project_root or "", rel_path)
                    patched_code = read_file(full)
                    if patched_code:
                        syn = await self._validate_syntax(patched_code)
                        if not syn.get("valid", False):
                            _log(f"  [step] ! syntax error in {rel_path}")
                            syntax_valid = False
            else:
                _log(f"  [step] ! symbolic patch failed, falling back to full-file")
                best_code = best_candidate.get("expected_code", "")
                if best_code:
                    success = apply_patch(file_path, best_code)
                else:
                    success = False
        else:
            # Legacy: single-file full replace
            best_code = best_candidate.get("expected_code", "")
            if best_code:
                syntax = await self._validate_syntax(best_code)
                syntax_valid = syntax.get("valid", False)
                if not syntax_valid:
                    _log(f"  [step] ! best candidate has syntax errors")
                success = apply_patch(file_path, best_code)
                changed_files = [os.path.relpath(file_path, self._project_root)] if self._project_root else [file_path]
            else:
                success = False

        # ── Update code index for all changed files ──
        if success and changed_files:
            idx = await self._ensure_code_index()
            for rel_path in changed_files:
                full = os.path.join(self._project_root or "", rel_path)
                if os.path.isfile(full):
                    try:
                        lang = _ext_to_lang.get(os.path.splitext(full)[1].lower(), "python")
                        old_entry = await self._update_index_entry(full, language=lang)
                        if old_entry:
                            _log(f"  [step] updated code index for {rel_path}")
                    except Exception as e:
                        _log(f"  [step] ! index update skipped for {rel_path}: {e}")

        try:
            vcontent = (
                f"# JEPA Step {len(self.history) + 1}\n\n"
                f"**Task:** {task}\n"
                f"**File:** {file_path}\n"
                f"**Changed files:** {changed_files}\n"
                f"**Selected:** candidate {best_idx}\n"
                f"**JEPA Loss:** {best_loss:.4f}\n"
            )
            await self._vault_write(f"lessons/step-{len(self.history) + 1}.md", vcontent, overwrite=True)
        except Exception:
            pass

        result = {
            "step": len(self.history) + 1, "task": task, "file": file_path,
            "changed_files": changed_files,
            "best_idx": best_idx, "best_description": best_candidate.get("description", ""),
            "jepa_loss": float(best_loss), "all_losses": [float(l) for l in losses],
            "success": success, "num_candidates": len(candidates),
            "syntax_valid": syntax_valid,
            "patch_count": len(best_patches) if best_patches else 0,
        }

        # ── Run tests after patch ──
        if success:
            _agent_dir = os.path.dirname(os.path.abspath(__file__))
            _stdout, _stderr, _ec = run_command(f'"{sys.executable}" -m pytest tests/ -q --tb=short', cwd=_agent_dir)
            if _ec == 0:
                _log(f"  [step] all tests passed after patch")
            else:
                _log(f"  [step] ! tests FAILED (exit={_ec})")
                _log(f"  [step] ! {_stderr[:200]}")
            result["tests_exit"] = _ec
            result["tests_stderr"] = _stderr[:200] if _stderr else ""

        self.history.append(result)
        return result

    async def onboard(self, file_path: str, install_parsers: bool = False) -> dict:
        """Run project onboarding: detect root, languages, check parsers, persist profile.

        Idempotent — re-runs update the stored profile. Call before run() or standalone.
        """
        _log("  [onboard] detecting project root...")
        root = detect_project_root(file_path)
        if not root:
            _log("  [onboard] ! could not determine project root")
            return {"error": "No project root detected"}

        # Load existing profile
        existing = load_project_profile(root)
        if existing:
            _log(f"  [onboard] found existing profile for {existing.get('project_name', root)}")
            self._project_root = root
            self._project_profile = existing
            self._onboarded = True
            # Build index if missing
            try:
                idx = await self._ensure_code_index()
                if not idx.get("files"):
                    await self.reindex()
            except (Exception, asyncio.CancelledError) as e:
                _log(f"  [onboard] ! index build skipped: {e}")
            return {"status": "loaded", "profile": existing}

        _log(f"  [onboard] project root: {root}")
        languages = detect_project_languages(root)
        _log(f"  [onboard] detected languages: {[l['language'] for l in languages]}")

        profile = generate_project_profile(root, languages)
        self._project_root = root
        self._project_profile = profile

        # Save to .jepa-project.json
        save_project_profile(profile)
        _log(f"  [onboard] saved profile to {root}/.jepa-project.json")

        # Write to vault
        md = format_profile_markdown(profile)
        try:
            await self._ensure_server("obsidian_brain")
            await self._vault_write("rules/project-profile.md", md, overwrite=True)
            _log("  [onboard] wrote profile to vault: rules/project-profile.md")
        except Exception as e:
            _log(f"  [onboard] ! vault write skipped: {e}")

        # Check parsers for detected languages
        missing = []
        for lang_info in languages:
            lang = lang_info["language"]
            if lang == "unknown":
                continue
            if lang != "python":
                missing.append(lang)

        if missing:
            _log(f"  [onboard] ! missing parsers: {', '.join(missing)}")
            if install_parsers:
                for lang in missing:
                    try:
                        await self._ensure_server("code_understanding")
                        r = await self._call("code_understanding", "install_language", language=lang)
                        if isinstance(r, dict) and r.get("success"):
                            _log(f"  [onboard] installed parser: {lang}")
                        else:
                            _log(f"  [onboard] ! failed to install {lang}: {r}")
                    except Exception as e:
                        _log(f"  [onboard] ! install error for {lang}: {e}")

        # ── Build code index (tree-sitter AST cache) ──
        try:
            scanned = await self.reindex()
            _log(f"  [onboard] indexed {scanned} source file(s)")
        except (Exception, asyncio.CancelledError) as e:
            _log(f"  [onboard] ! code index build skipped: {e}")

        self._onboarded = True
        return {"status": "created", "profile": profile, "missing_parsers": missing}

    async def reindex(self, clear_first: bool = False) -> int:
        """Rebuild the code index from scratch using the current project profile.

        Call after onboard() to refresh the symbol index — useful when
        files change or new extensions are added. Idempotent.

        Args:
            clear_first: If True, delete the existing index before rebuilding.

        Returns:
            Number of files indexed, or 0 if no project root/profile set.
        """
        if not self._project_root:
            _log("  [reindex] ! no project root — call onboard() first")
            return 0
        if not self._project_profile:
            _log("  [reindex] ! no project profile — call onboard() first")
            return 0

        if clear_first:
            idx = await self._ensure_code_index()
            idx.clear()
            _log("  [reindex] cleared existing index")

        exts = self._project_profile.get("all_extensions", [])
        if not exts:
            _log("  [reindex] ! no extensions in profile")
            return 0

        scanned = await self._build_code_index(exts)
        _log(f"  [reindex] indexed {scanned} source file(s)")
        return scanned

    async def run(self, task: str, file_path: str, max_steps: int = MAX_STEPS) -> list[dict]:
        """Multi-step JEPA loop. Auto-runs onboard on first call if no profile."""
        if not self._onboarded:
            onboard_result = await self.onboard(file_path)
            if onboard_result.get("error"):
                _log(f"  [onboard] warning: {onboard_result['error']}")

        results = []
        for step_num in range(1, max_steps + 1):
            _log(f"\n{'='*50}\n  JEPA Step {step_num}/{max_steps}\n{'='*50}")
            result = await self.step(task, file_path)
            results.append(result)
            if result.get("jepa_loss", 1.0) < 0.01:
                _log("[JEPAAgent] Converged (loss near zero).")
                break
            if not result.get("success"):
                _log("[JEPAAgent] Step failed, stopping.")
                break
            task = f"Continue fixing. Previous attempt: {result.get('best_description', '')}"
        return results

    async def cleanup(self):
        for name in list(self._servers.keys()):
            try:
                await self._servers[name].disconnect()
            except Exception:
                pass
        self._servers.clear()


def run_agent(task: str, file_path: str, k: int = NUM_CANDIDATES, steps: int = MAX_STEPS) -> list[dict]:
    """Synchronous entry point. Runs the async agent loop."""
    async def _run():
        agent = JEPAAgent()
        try:
            return await agent.run(task=task, file_path=file_path, max_steps=steps)
        finally:
            await agent.cleanup()
    return asyncio.run(_run())
