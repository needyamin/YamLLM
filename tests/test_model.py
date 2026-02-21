import os
import sys
import pytest
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import LLMConfig
from src.model.attention import CausalSelfAttention
from src.model.layers import Block, FeedForward
from src.model.transformer import Transformer

@pytest.fixture
def tiny_config():
    return LLMConfig(
        vocab_size=100,
        max_seq_len=64,
        d_model=32,
        n_layers=2,
        n_heads=4,
        batch_size=2
    )

def test_causal_self_attention_shape(tiny_config):
    attn = CausalSelfAttention(tiny_config)
    
    # Input tensor shape: (batch_size, sequence_length, d_model)
    x = torch.randn(tiny_config.batch_size, 16, tiny_config.d_model)
    
    # Output tensor shape should be the same
    y = attn(x)
    assert y.shape == (tiny_config.batch_size, 16, tiny_config.d_model)

def test_feed_forward_shape(tiny_config):
    ff = FeedForward(tiny_config)
    x = torch.randn(tiny_config.batch_size, 16, tiny_config.d_model)
    y = ff(x)
    assert y.shape == (tiny_config.batch_size, 16, tiny_config.d_model)

def test_block_shape(tiny_config):
    block = Block(tiny_config)
    x = torch.randn(tiny_config.batch_size, 16, tiny_config.d_model)
    y = block(x)
    assert y.shape == (tiny_config.batch_size, 16, tiny_config.d_model)
    
def test_transformer_forward_pass(tiny_config):
    model = Transformer(tiny_config)
    
    # Random integers simulating tokens
    idx = torch.randint(0, tiny_config.vocab_size, (tiny_config.batch_size, 16))
    
    # Inference mode forward pass (no targets)
    logits, loss = model(idx)
    
    assert loss is None
    # Only the last token's logits are returned during inference optimization
    assert logits.shape == (tiny_config.batch_size, 1, tiny_config.vocab_size)
    
    # Training mode forward pass (with targets)
    targets = torch.randint(0, tiny_config.vocab_size, (tiny_config.batch_size, 16))
    logits, loss = model(idx, targets)
    
    assert loss is not None
    assert loss.item() > 0
    # All tokens' logits are returned
    assert logits.shape == (tiny_config.batch_size, 16, tiny_config.vocab_size)
