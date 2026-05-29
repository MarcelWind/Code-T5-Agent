"""JEPA Coding Agent - Configuration.

Version: 0.1.0
"""

import os

__version__ = '0.1.0'

# ── DeepSeek Flash API ──
DEEPSEEK_API_KEY = os.environ.get("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

# ── CodeT5+ Encoder ──
CODET5_MODEL = "Salesforce/codet5p-220m"     # fast (220M params)
# CODET5_MODEL = "Salesforce/codet5p-770m"   # better embeddings (1.5GB)
EMBEDDING_DIM = 768                            # codet5p-220m hidden dim

# ── JEPA Scoring ──
JEPA_LOSS_TYPE = "cosine"   # cosine | l2 | contrastive
JEPA_TEMPERATURE = 0.07     # for contrastive loss

# ── Agent Loop ──
NUM_CANDIDATES = 5
MAX_STEPS = 5
MAX_CODE_TOKENS = 512       # max tokens for CodeT5+ encoding

# ── Context Builder ──
CONTEXT_BUDGET_TOKENS = 4096   # max tokens for context package sent to LLM
CONTEXT_EXPANSION_HOPS = 2     # max import hop depth for dependency expansion
CONTEXT_MAX_SEED_FILES = 5     # max semantic search seeds to expand
CONTEXT_INCLUDE_MEMORY = True  # include vault memory hits in context
