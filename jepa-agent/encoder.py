"""CodeT5+ encoder — maps code text to semantic embedding vectors."""

import os
import shutil
from pathlib import Path

import torch
import numpy as np
from transformers import T5EncoderModel, AutoTokenizer

from config import CODET5_MODEL, MAX_CODE_TOKENS, EMBEDDING_DIM

# Store model cache inside project folder so it doesn't pollute user home
_MODEL_CACHE = Path(__file__).parent / "models"
_MODEL_CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("HF_HOME", str(_MODEL_CACHE))
os.environ.setdefault("HF_HUB_CACHE", str(_MODEL_CACHE / "hub"))


class CodeEncoder:
    """Wrap CodeT5+ encoder. Produces mean-pooled embedding for code text."""

    def __init__(self, model_name: str = CODET5_MODEL):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = T5EncoderModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        print(f"[CodeEncoder] loaded {model_name} on {self.device}")

    @torch.no_grad()
    def encode(self, code: str) -> np.ndarray:
        """Encode code string → 1-D embedding vector."""
        inputs = self.tokenizer(
            code,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_CODE_TOKENS,
        ).to(self.device)

        outputs = self.model(**inputs)
        # mean-pool over non-padding tokens
        attn = inputs["attention_mask"]
        hidden = outputs.last_hidden_state * attn.unsqueeze(-1)
        emb = hidden.sum(dim=1) / attn.sum(dim=1, keepdim=True)
        return emb.cpu().numpy().flatten()

    def encode_diff(self, original: str, patched: str) -> np.ndarray:
        """Delta embedding: patched - original (direction of code change)."""
        return self.encode(patched) - self.encode(original)

    @torch.no_grad()
    def encode_batch(self, codes: list[str]) -> np.ndarray:
        """Encode multiple code strings → 2-D array (batch, dim)."""
        inputs = self.tokenizer(
            codes,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_CODE_TOKENS,
            padding=True,
        ).to(self.device)

        outputs = self.model(**inputs)
        attn = inputs["attention_mask"]
        hidden = outputs.last_hidden_state * attn.unsqueeze(-1)
        emb = hidden.sum(dim=1) / attn.sum(dim=1, keepdim=True)
        return emb.cpu().numpy()

    @property
    def dim(self) -> int:
        return EMBEDDING_DIM
