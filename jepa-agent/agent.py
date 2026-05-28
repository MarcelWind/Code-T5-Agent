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

from core.executor import read_file, apply_patch, run_command
from core.config import NUM_CANDIDATES, MAX_STEPS, JEPA_LOSS_TYPE
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
    needs_reparse,
    extract_from_ast,
    compute_diff,
    format_diff_markdown,
    format_symbol_summary,
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
    try:
        return json.loads(combined)
    except (json.JSONDecodeError, TypeError):
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
}


class MCPConnection:
    """A single stdio MCP server connection."""

    def __init__(self, name: str, module: str, args: list[str] | None = None, env: dict | None = None):
        self.name = name
        base_env = {**os.environ, **{"PYTHONIOENCODING": "utf-8"}, **(env or {})}
        self.params = StdioServerParameters(
            command=sys.executable,
            args=["-m", module] + (args or []),
            env=base_env,
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
            print(f"  [mcp] connected -> {self.name}")
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

    async def call(self, tool: str, **kwargs) -> Any:
        if not self.session:
            await self.connect()
        result = await self.session.call_tool(tool, arguments=kwargs or None)
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
        print("[JEPAAgent] MCP-orchestrated agent initialized")

    async def _ensure_server(self, name: str):
        if name not in self._servers:
            cfg = SERVER_DEFS[name]
            conn = MCPConnection(name=name, module=cfg["module"], args=cfg.get("args", []), env=cfg.get("env"))
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

    async def _plan_actions(self, code: str, task: str, k: int = 5) -> list[dict]:
        result = await self._call("cloud_execution", "plan_actions", code_context=code, task=task, k=k)
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
            return await self._call("code_understanding", "parse_code", code=code, language=language, detail="overview")
        except (Exception, asyncio.CancelledError) as e:
            print(f"  [index] ! parse failed for {file_path}: {e}")
            return None

    async def _update_index_entry(self, file_path: str, language: str = "python") -> dict | None:
        """Parse a file and update its entry in the index. Returns old entry if changed."""
        idx = await self._ensure_code_index()
        rel_path = os.path.relpath(file_path, self._project_root) if self._project_root else file_path
        old_entry = idx.get("files", {}).get(rel_path)

        code = read_file(file_path)
        if not code:
            return old_entry
        ast_data = await self._parse_ast(file_path, language)
        if ast_data and not ast_data.get("error"):
            syms = extract_from_ast(ast_data, source_code=code)
            new_entry = {
                "mtime": os.path.getmtime(file_path),
                "language": ast_data.get("language", language),
                "size": os.path.getsize(file_path),
                "symbols": syms,
            }
            idx.setdefault("files", {})[rel_path] = new_entry
            save_index(idx)
            return old_entry
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
                    ast_data = await self._parse_ast(full_path, language=lang)
                    if ast_data and not ast_data.get("error"):
                        syms = extract_from_ast(ast_data, source_code=file_code)
                        idx = await self._ensure_code_index()
                        idx.setdefault("files", {})[rel_path] = {
                            "mtime": os.path.getmtime(full_path),
                            "language": ast_data.get("language", lang),
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
            print(f"  [step] encoded current state -> dim={len(S_t)}")
        except Exception as e:
            return {"error": f"Encoding failed: {e}"}

        # ── AST context from tree-sitter index ──
        ast_context = ""
        try:
            idx = await self._ensure_code_index()
            rel_path = os.path.relpath(file_path, self._project_root) if self._project_root else file_path
            entry = idx.get("files", {}).get(rel_path)

            if needs_reparse(file_path, entry):
                old_entry = await self._update_index_entry(file_path)
                entry = idx.get("files", {}).get(rel_path)
                diff = compute_diff(
                    (old_entry or {}).get("symbols", {}),
                    (entry or {}).get("symbols", {}),
                )
                diff_str = format_diff_markdown(diff, rel_path)
                if diff_str:
                    ast_context = diff_str + "\n"
                    print(f"  [step] AST diff detected")
            elif entry:
                summary = format_symbol_summary(entry)
                if summary:
                    ast_context = f"**Current symbols in `{rel_path}`:** {summary}\n"

        except Exception as e:
            print(f"  [step] ! AST context skipped: {e}")

        task_with_ast = f"{task}\n\n{ast_context}" if ast_context else task

        try:
            candidates = await self._plan_actions(code, task_with_ast, k=k)
            print(f"  [step] generated {len(candidates)} candidates")
        except Exception as e:
            return {"error": f"Planning failed: {e}"}

        if not candidates:
            return {"error": "No candidates generated", "state_embedding": S_t}

        scored_inputs = []
        for cand in candidates:
            desc = cand.get("change_description", cand.get("description", ""))
            exp_code = cand.get("expected_code", "")
            if desc and exp_code:
                scored_inputs.append({"change_description": desc, "expected_code": exp_code, "description": cand.get("description", "")[:60]})

        if not scored_inputs:
            for cand in candidates:
                scored_inputs.append({"change_description": task, "expected_code": cand.get("expected_code", code), "description": cand.get("description", "")[:60]})

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
            print(f"  [step] ranking server failed ({e}), scoring locally...")
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
            print(f"  [step]   candidate {i}: {desc}...{loss_str}")

        best_candidate = candidates[best_idx]
        best_code = best_candidate.get("expected_code", "")
        print(f"  [step] selected candidate {best_idx} (loss={best_loss:.4f})")

        syntax = await self._validate_syntax(best_code)
        if not syntax.get("valid", False):
            print(f"  [step] ! best candidate has syntax errors")

        success = apply_patch(file_path, best_code)

        # ── Update code index after patch ──
        if success:
            try:
                lang = _ext_to_lang.get(os.path.splitext(file_path)[1].lower(), "python")
                old_entry = await self._update_index_entry(file_path, language=lang)
                if old_entry:
                    print(f"  [step] updated code index for {os.path.basename(file_path)}")
            except Exception as e:
                print(f"  [step] ! index update skipped: {e}")

        try:
            vcontent = f"# JEPA Step {len(self.history) + 1}\n\n**Task:** {task}\n**File:** {file_path}\n**Selected:** candidate {best_idx}\n**JEPA Loss:** {best_loss:.4f}\n"
            await self._vault_write(f"lessons/step-{len(self.history) + 1}.md", vcontent, overwrite=True)
        except Exception:
            pass

        result = {
            "step": len(self.history) + 1, "task": task, "file": file_path,
            "best_idx": best_idx, "best_description": best_candidate.get("description", ""),
            "jepa_loss": float(best_loss), "all_losses": [float(l) for l in losses],
            "success": success, "num_candidates": len(candidates),
            "syntax_valid": syntax.get("valid", False),
        }

        # ── Run tests after patch ──
        if success:
            _agent_dir = os.path.dirname(os.path.abspath(__file__))
            _stdout, _stderr, _ec = run_command(f'"{sys.executable}" -m pytest tests/ -q --tb=short', cwd=_agent_dir)
            if _ec == 0:
                print(f"  [step] all tests passed after patch")
            else:
                print(f"  [step] ! tests FAILED (exit={_ec})")
                print(f"  [step] ! {_stderr[:200]}")
            result["tests_exit"] = _ec
            result["tests_stderr"] = _stderr[:200] if _stderr else ""

        self.history.append(result)
        return result

    async def onboard(self, file_path: str, install_parsers: bool = False) -> dict:
        """Run project onboarding: detect root, languages, check parsers, persist profile.

        Idempotent — re-runs update the stored profile. Call before run() or standalone.
        """
        print("  [onboard] detecting project root...")
        root = detect_project_root(file_path)
        if not root:
            print("  [onboard] ! could not determine project root")
            return {"error": "No project root detected"}

        # Load existing profile
        existing = load_project_profile(root)
        if existing:
            print(f"  [onboard] found existing profile for {existing.get('project_name', root)}")
            self._project_root = root
            self._project_profile = existing
            self._onboarded = True
            # Build index if missing or stale
            try:
                idx = await self._ensure_code_index()
                if not idx.get("files"):
                    exts = existing.get("all_extensions", [])
                    if exts:
                        scanned = await self._build_code_index(exts)
                        print(f"  [onboard] indexed {scanned} source file(s)")
            except (Exception, asyncio.CancelledError) as e:
                print(f"  [onboard] ! index build skipped: {e}")
            return {"status": "loaded", "profile": existing}

        print(f"  [onboard] project root: {root}")
        languages = detect_project_languages(root)
        print(f"  [onboard] detected languages: {[l['language'] for l in languages]}")

        profile = generate_project_profile(root, languages)
        self._project_root = root
        self._project_profile = profile

        # Save to .jepa-project.json
        save_project_profile(profile)
        print(f"  [onboard] saved profile to {root}/.jepa-project.json")

        # Write to vault
        md = format_profile_markdown(profile)
        try:
            await self._ensure_server("obsidian_brain")
            await self._vault_write("rules/project-profile.md", md, overwrite=True)
            print("  [onboard] wrote profile to vault: rules/project-profile.md")
        except Exception as e:
            print(f"  [onboard] ! vault write skipped: {e}")

        # Check parsers for detected languages
        missing = []
        for lang_info in languages:
            lang = lang_info["language"]
            if lang == "unknown":
                continue
            if lang != "python":
                missing.append(lang)

        if missing:
            print(f"  [onboard] ! missing parsers: {', '.join(missing)}")
            if install_parsers:
                for lang in missing:
                    try:
                        await self._ensure_server("code_understanding")
                        r = await self._call("code_understanding", "install_language", language=lang)
                        if isinstance(r, dict) and r.get("success"):
                            print(f"  [onboard] installed parser: {lang}")
                        else:
                            print(f"  [onboard] ! failed to install {lang}: {r}")
                    except Exception as e:
                        print(f"  [onboard] ! install error for {lang}: {e}")

        # ── Build code index (tree-sitter AST cache) ──
        try:
            exts = profile.get("all_extensions", [])
            if exts:
                scanned = await self._build_code_index(exts)
                print(f"  [onboard] indexed {scanned} source file(s)")
        except (Exception, asyncio.CancelledError) as e:
            print(f"  [onboard] ! code index build skipped: {e}")

        self._onboarded = True
        return {"status": "created", "profile": profile, "missing_parsers": missing}

    async def run(self, task: str, file_path: str, max_steps: int = MAX_STEPS) -> list[dict]:
        """Multi-step JEPA loop. Auto-runs onboard on first call if no profile."""
        if not self._onboarded:
            onboard_result = await self.onboard(file_path)
            if onboard_result.get("error"):
                print(f"  [onboard] warning: {onboard_result['error']}")

        results = []
        for step_num in range(1, max_steps + 1):
            print(f"\n{'='*50}\n  JEPA Step {step_num}/{max_steps}\n{'='*50}")
            result = await self.step(task, file_path)
            results.append(result)
            if result.get("jepa_loss", 1.0) < 0.01:
                print("[JEPAAgent] Converged (loss near zero).")
                break
            if not result.get("success"):
                print("[JEPAAgent] Step failed, stopping.")
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
