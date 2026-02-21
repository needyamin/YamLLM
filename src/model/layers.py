import torch
import torch.nn as nn
from src.utils.config import LLMConfig
from src.model.attention import CausalSelfAttention

class FeedForward(nn.Module):
    """
    Standard Feed-Forward Network (MLP) as used in GPT models.
    Expands the dimensionality by 4x and then projects back down.
    """
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.c_fc    = nn.Linear(config.d_model, 4 * config.d_model, bias=False)
        self.act     = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.d_model, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class Block(nn.Module):
    """
    A single Transformer decoding block combining Self-Attention and Feed-Forward networks
    with layer normalization and residual connections.
    """
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.d_model, eps=config.epsilon)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.d_model, eps=config.epsilon)
        self.mlp = FeedForward(config)

    def forward(self, x):
        # We use pre-norm formulation (LayerNorm is applied BEFORE attention/mlp)
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
