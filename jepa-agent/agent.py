"""JEPA agent loop — core pipeline.

Flow per step:
1. Read file → CodeT5+ encode → S_t (current embedding)
2. DeepSeek generates N candidate patches + expected_code
3. For each candidate:
   a. Encode expected_code → Z_hat (predicted embedding)
   b. Encode actual patched code → Z_actual
   c. JEPA loss = distance(Z_hat, Z_actual)
4. Select lowest-loss candidate
5. Apply best patch to file
6. Return result
"""

import numpy as np

from encoder import CodeEncoder
from predictor import DeepSeekPredictor
from scorer import jepa_loss, rank_candidates
from executor import Workspace, read_file, apply_patch
from config import NUM_CANDIDATES, MAX_STEPS, JEPA_LOSS_TYPE


class JEPAAgent:
    """JEPA-style coding agent."""

    def __init__(self):
        print("[JEPAAgent] initializing encoder + predictor...")
        self.encoder = CodeEncoder()
        self.predictor = DeepSeekPredictor()
        self.workspace = Workspace()
        self.history: list[dict] = []

    def step(
        self,
        task: str,
        file_path: str,
        k: int = NUM_CANDIDATES,
    ) -> dict:
        """Single JEPA step: observe → predict → score → execute."""
        # 1. Observe current state
        code = read_file(file_path)
        if not code:
            return {"error": f"File not found or empty: {file_path}"}

        S_t = self.encoder.encode(code)  # current embedding
        print(f"  [step] encoded current state → dim={S_t.shape}")

        # 2. DeepSeek proposes candidates
        candidates = self.predictor.plan_actions(code, task, k=k)
        print(f"  [step] generated {len(candidates)} candidates")

        if not candidates:
            return {"error": "No candidates generated", "state_embedding": S_t}

        # 3. Score each candidate via JEPA loss
        predicted_embs: list[np.ndarray] = []
        actual_embs: list[np.ndarray] = []

        for i, cand in enumerate(candidates):
            expected_code = cand.get("expected_code", "")
            change_desc = cand.get("change_description", "")
            description = cand.get("description", "")

            # Predicted embedding Z_hat — from DeepSeek's semantic description
            # This is the "latent prediction": what should the code state feel like?
            Z_hat = self.encoder.encode(change_desc if change_desc else description)

            # Actual embedding Z — from the actual code after applying the patch
            # In prototype, candidate.expected_code IS the resulting code
            Z_actual = self.encoder.encode(expected_code)

            predicted_embs.append(Z_hat)
            actual_embs.append(Z_actual)

            loss = jepa_loss(Z_hat, Z_actual, JEPA_LOSS_TYPE)
            print(f"  [step]   candidate {i}: {description[:50]}... loss={loss:.4f}")

        # 4. Rank by JEPA loss
        if len(candidates) > 1:
            # Cross-evaluation: compute loss between each candidate's predicted
            # embedding (Z_hat from change_description) and each candidate's
            # actual embedding (Z from expected_code).
            # Best candidate = whose prediction best aligns with reality.
            n = len(candidates)
            loss_matrix = np.zeros((n, n))
            for i in range(n):
                for j in range(n):
                    loss_matrix[i, j] = jepa_loss(
                        predicted_embs[i], actual_embs[j], JEPA_LOSS_TYPE
                    )

            # Self-consistency: diagonal of matrix
            # (how well each candidate's prediction matches its own actual output)
            self_scores = [loss_matrix[i, i] for i in range(n)]

            # Cross-consistency: how well each candidate's prediction matches
            # the AVERAGE actual encoding across all candidates
            avg_actual = np.mean(actual_embs, axis=0)
            cross_scores = [
                jepa_loss(predicted_embs[i], avg_actual, JEPA_LOSS_TYPE)
                for i in range(n)
            ]

            # Combine: use self-consistency as primary score
            scores = self_scores
            best_idx = int(np.argmin(scores))
        else:
            best_idx = 0
            scores = [0.0]

        best_candidate = candidates[best_idx]
        best_code = best_candidate.get("expected_code", "")

        print(f"  [step] selected candidate {best_idx} (loss={scores[best_idx]:.4f})")

        # 5. Execute: apply best patch
        success = apply_patch(file_path, best_code)

        result = {
            "step": len(self.history) + 1,
            "task": task,
            "file": file_path,
            "best_idx": best_idx,
            "best_description": best_candidate.get("description", ""),
            "jepa_loss": float(scores[best_idx]),
            "all_losses": [float(s) for s in scores],
            "success": success,
            "num_candidates": len(candidates),
        }
        self.history.append(result)
        return result

    def run(
        self,
        task: str,
        file_path: str,
        max_steps: int = MAX_STEPS,
    ) -> list[dict]:
        """Multi-step JEPA loop."""
        results = []
        for _ in range(max_steps):
            result = self.step(task, file_path)
            results.append(result)
            if result.get("jepa_loss", 1.0) < 0.01:
                # Near-zero loss → prediction matched reality perfectly → done
                print("[JEPAAgent] Converged (loss near zero).")
                break
            if not result.get("success"):
                print("[JEPAAgent] Step failed, stopping.")
                break
            # Update task for next step if needed
            task = f"Continue fixing. Previous attempt: {result.get('best_description', '')}"
        return results
