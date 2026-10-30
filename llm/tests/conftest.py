import pytest
import torch

from aurora_llm.config import ModelConfig


@pytest.fixture
def small_cfg():
    torch.manual_seed(0)
    return ModelConfig(vocab_size=97, d_model=64, n_layers=2, n_heads=4, n_kv_heads=2, max_seq_len=32)
