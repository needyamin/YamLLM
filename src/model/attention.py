import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from src.utils.config import LLMConfig

class CausalSelfAttention(nn.Module):
    """
    Multi-Head Causal Self-Attention module.
    It guarantees that the model can only look at previous tokens.
    """
    def __init__(self, config: LLMConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads
        self.dropout = config.dropout
        
        # Key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        
        # Output projection
        self.c_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        
        # Regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        
        # Causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.max_seq_len, config.max_seq_len))
                                     .view(1, 1, config.max_seq_len, config.max_seq_len))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (d_model)
        
        # Calculate query, key, values for all heads in batch and move head forward to be the batch dim
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.d_model, dim=2)
        
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) # (B, nh, T, hs)
        
        # Causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        # Using flash attention via memory-efficient pytorch functional scaled_dot_product_attention 
        # is highly recommended for modern PyTorch (>= 2.0).
        y = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=None, 
            dropout_p=self.dropout if self.training else 0.0, 
            is_causal=True
        )
        
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
        
        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y
