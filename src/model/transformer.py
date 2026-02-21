import math
import torch
import torch.nn as nn
from src.utils.config import LLMConfig
from src.model.layers import Block

class Transformer(nn.Module):
    """
    The full GPT-style decoder-only Transformer Large Language Model.
    """
    def __init__(self, config: LLMConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.d_model),
            wpe = nn.Embedding(config.max_seq_len, config.d_model),
            drop = nn.Dropout(config.dropout),
            h = nn.ModuleList([Block(config) for _ in range(config.n_layers)]),
            ln_f = nn.LayerNorm(config.d_model, eps=config.epsilon),
        ))
        
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        
        # Weight tying: The embedding weights and LM head weights are shared
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initializes weights according to typical Transformer initialization."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.max_seq_len, f"Cannot forward sequence of length {t}, block size is only {self.config.max_seq_len}"
        
        pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

        # Forward the transformer model
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, d_model)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, d_model)
        x = self.transformer.drop(tok_emb + pos_emb)
        
        for block in self.transformer.h:
            x = block(x)
            
        x = self.transformer.ln_f(x)

        if targets is not None:
            # If we are training, compute the loss
            logits = self.lm_head(x)
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
            return logits, loss
        else:
            # Inference-time optimization
            # Only calculate logits for the final token in the sequence to save compute
            logits = self.lm_head(x[:, [-1], :]) # shape (b, 1, vocab_size)
            return logits, None
            
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Takes a conditioning sequence of indices idx (LongTensor of shape (b,t)) and completes
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        """
        self.eval()
        for _ in range(max_new_tokens):
            # Crop length to the block size max_seq_len
            idx_cond = idx if idx.size(1) <= self.config.max_seq_len else idx[:, -self.config.max_seq_len:]
            
            # Forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            
            # Pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            
            # Optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
            # Apply softmax to convert logits to (normalized) probabilities
            probs = torch.nn.functional.softmax(logits, dim=-1)
            
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
            
        self.train()
        return idx
