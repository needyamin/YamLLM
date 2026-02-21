from dataclasses import dataclass
from typing import Optional

@dataclass
class LLMConfig:
    """
    Configuration for the Large Language Model.
    Defaults are set for a small, easily trainable GPT-style model.
    """
    # Tokenizer settings
    vocab_size: int = 50257
    max_seq_len: int = 1024
    
    # Transformer architecture settings
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    dropout: float = 0.1
    epsilon: float = 1e-5 # For LayerNorm
    
    # Training settings
    batch_size: int = 32
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    max_iters: int = 100000
    warmup_iters: int = 2000
    eval_interval: int = 500
    eval_iters: int = 200
    
    # System
    device: str = "cuda"
    dtype: str = "float32" # or bfloat16/float16
    
    @classmethod
    def create_nano(cls) -> "LLMConfig":
        """Creates a tiny config for testing/debugging."""
        return cls(
            d_model=128,
            n_layers=4,
            n_heads=4,
            max_seq_len=256,
            batch_size=16
        )
