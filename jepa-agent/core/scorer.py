"""JEPA loss functions — score candidate actions in embedding space.

Compares predicted embedding (Z_hat) vs actual embedding (Z) to rank candidates.
"""

import numpy as np

from .config import JEPA_LOSS_TYPE, JEPA_TEMPERATURE


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance: 1 - cos(a, b). Range [0, 2]. Lower = more similar."""
    a_norm = a / (np.linalg.norm(a) + 1e-12)
    b_norm = b / (np.linalg.norm(b) + 1e-12)
    return float(1.0 - np.dot(a_norm, b_norm))


def l2_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance. Lower = more similar."""
    return float(np.linalg.norm(a - b))


def contrastive_loss(
    anchor: np.ndarray,
    positive: np.ndarray,
    negatives: list[np.ndarray],
    temperature: float = JEPA_TEMPERATURE,
) -> float:
    """InfoNCE-style contrastive loss.

    Args:
        anchor: predicted embedding Z_hat
        positive: actual embedding Z (positive pair)
        negatives: list of distractor/negative embeddings
        temperature: scaling factor
    Returns:
        Loss value (lower = better alignment)
    """
    pos_sim = np.dot(anchor, positive) / temperature
    neg_sims = [np.dot(anchor, n) / temperature for n in negatives]
    all_sims = np.array([pos_sim] + neg_sims)
    exp_sims = np.exp(all_sims - np.max(all_sims))
    return float(-np.log(exp_sims[0] / exp_sims.sum()))


def jepa_loss(
    predicted: np.ndarray,
    actual: np.ndarray,
    loss_type: str = JEPA_LOSS_TYPE,
) -> float:
    """Compute JEPA-style loss between predicted and actual embeddings.

    Args:
        predicted: predicted next embedding Z_hat
        actual: actual next embedding Z
        loss_type: "cosine" | "l2" | "contrastive"
    Returns:
        Scalar loss (lower = better prediction)
    """
    if loss_type == "cosine":
        return cosine_distance(predicted, actual)
    elif loss_type == "l2":
        return l2_distance(predicted, actual)
    elif loss_type == "contrastive":
        # For contrastive, caller should use contrastive_loss() directly
        return cosine_distance(predicted, actual)
    else:
        raise ValueError(f"Unknown loss_type: {loss_type}")


def rank_candidates(
    predicted_embeddings: list[np.ndarray],
    actual_embeddings: list[np.ndarray],
    loss_type: str = JEPA_LOSS_TYPE,
) -> list[int]:
    """Rank candidate indices by JEPA loss (ascending = best first)."""
    scores = [
        jepa_loss(pred, act, loss_type)
        for pred, act in zip(predicted_embeddings, actual_embeddings)
    ]
    return sorted(range(len(scores)), key=lambda i: scores[i])
