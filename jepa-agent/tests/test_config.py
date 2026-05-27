"""Tests for core.config — configuration defaults and types."""

import os

import pytest

from core import config


class TestConfigValues:
    """Verify all config values exist with sensible defaults."""

    def test_deepseek_api_key_exists(self):
        # Should always be a string (empty if env var not set)
        assert isinstance(config.DEEPSEEK_API_KEY, str)

    def test_deepseek_base_url(self):
        assert config.DEEPSEEK_BASE_URL == "https://api.deepseek.com"

    def test_deepseek_model(self):
        assert config.DEEPSEEK_MODEL == "deepseek-chat"

    def test_codet5_model(self):
        assert config.CODET5_MODEL == "Salesforce/codet5p-220m"

    def test_embedding_dim(self):
        assert config.EMBEDDING_DIM == 768

    def test_jepa_loss_type_valid(self):
        assert config.JEPA_LOSS_TYPE in ("cosine", "l2", "contrastive")

    def test_jepa_temperature(self):
        assert isinstance(config.JEPA_TEMPERATURE, float)
        assert config.JEPA_TEMPERATURE > 0

    def test_num_candidates(self):
        assert isinstance(config.NUM_CANDIDATES, int)
        assert config.NUM_CANDIDATES > 0

    def test_max_steps(self):
        assert isinstance(config.MAX_STEPS, int)
        assert config.MAX_STEPS > 0

    def test_max_code_tokens(self):
        assert isinstance(config.MAX_CODE_TOKENS, int)
        assert config.MAX_CODE_TOKENS > 0

    def test_embedding_dim_matches_model(self):
        # 220M model has 768-dim hidden
        if config.CODET5_MODEL == "Salesforce/codet5p-220m":
            assert config.EMBEDDING_DIM == 768
