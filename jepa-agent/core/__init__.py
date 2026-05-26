"""JEPA Core — encoder, predictor, scorer, executor, config."""

from .config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    CODET5_MODEL,
    EMBEDDING_DIM,
    MAX_CODE_TOKENS,
    NUM_CANDIDATES,
    MAX_STEPS,
    JEPA_LOSS_TYPE,
    JEPA_TEMPERATURE,
)
from .encoder import CodeEncoder
from .predictor import DeepSeekPredictor
from .scorer import jepa_loss, cosine_distance, l2_distance, rank_candidates
from .executor import Workspace, read_file, write_file, apply_patch, run_command

__all__ = [
    "DEEPSEEK_API_KEY", "DEEPSEEK_BASE_URL", "DEEPSEEK_MODEL",
    "CODET5_MODEL", "EMBEDDING_DIM", "MAX_CODE_TOKENS",
    "NUM_CANDIDATES", "MAX_STEPS", "JEPA_LOSS_TYPE", "JEPA_TEMPERATURE",
    "CodeEncoder", "DeepSeekPredictor",
    "jepa_loss", "cosine_distance", "l2_distance", "rank_candidates",
    "Workspace", "read_file", "write_file", "apply_patch", "run_command",
]
