# JEPA Pipeline Specification

## Overview

The JEPA (Joint Embedding Predictive Architecture) coding agent operates a closed-loop pipeline:
observe current code → encode to embedding → predict candidate changes → score candidates in embedding space → select best → apply → store lesson.

## Step Detail

### 1. Observe
- Agent reads the target file via `core.executor.read_file()`
- Returns raw source code string `S_t`

### 2. Encode (Z_t)
- **Server**: `semantic_search` → `encode_code(code)`
- **Model**: `Salesforce/codet5p-220m` (T5EncoderModel, 768-dim)
- Takes file contents, returns float embedding vector `Z_t`
- Used as reference point for the "current state" of the code

### 3. Predict (k candidates)
- **Server**: `cloud_execution` → `plan_actions(code_context, task, k)`
- **Model**: `deepseek-chat` (DeepSeek Flash API)
- Prompt instructs DeepSeek to output JSON array of objects, each with:
  - `change_description`: what the change does (encoded as Z_hat — the *predicted* embedding)
  - `expected_code`: the resulting code after change (encoded as Z_actual — the *observed* embedding)
  - `description`: human-readable summary

### 4. Score (JEPA loss)
- **Server**: `validators` → `rank_candidates(candidates, loss_type)`
- For each candidate:
  - Encode `change_description` → `Z_hat`
  - Encode `expected_code` → `Z_actual`
  - Compute `loss = cosine_distance(Z_hat, Z_actual)` (bounded [0, 2])
- Lower loss → prediction better matches observation → candidate more likely correct

### 5. Select
- Pick candidate with lowest JEPA loss
- **Fallback**: If `rank_candidates` fails, score each candidate individually via `_score_pair()` and use `numpy.argmin()`

### 6. Validate
- **Server**: `validators` → `validate_syntax(code, language)`
- Uses Python AST module (`ast.parse()`) to check for syntax errors
- Result is advisory only — code is applied regardless of syntax result

### 7. Execute
- **Module**: `core.executor.apply_patch(file_path, code)`
- Writes the selected `expected_code` to the target file
- Returns boolean success flag

### 8. Store
- **Server**: `obsidian_brain` → `write_vault(path, content, overwrite)`
- Logs step summary to `vault/lessons/step-N.md`
- Non-critical — failures are silently caught

## Multi-Step Loop

When `steps > 1`:
1. Run steps 1-8 above
2. If `jepa_loss < 0.01` → converged, stop early
3. If step failed (`success == False`) → stop
4. Otherwise refine task: "Continue fixing. Previous attempt: {best_description}"
5. Loop reads the updated file (post-patch) as new state `S_t`

## Convergence Criteria
- **Loss threshold**: `jepa_loss < 0.01` → converged
- **Failure**: `success == False` → stop
- **Max steps**: configurable via `--steps` / `MAX_STEPS` (default 3)

## Entry Points
- **Async**: `await JEPAAgent().run(task, file_path, max_steps)` → `list[dict]`
- **Sync**: `run_agent(task, file_path, k, steps)` → `list[dict]` (wraps async in `asyncio.run()`)
- **CLI**: `python main.py --task "..." --file "..." --candidates 5 --steps 3`
