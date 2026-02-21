import torch
import torch.nn as nn
from typing import List

@torch.no_grad()
def evaluate_model(model: nn.Module, val_loader, eval_iters: int, device: str) -> float:
    """
    Evaluates the model on the validation dataset.
    Returns the average loss across the evaluation iterations.
    """
    model.eval()
    val_iter = iter(val_loader)
    losses = torch.zeros(eval_iters)
    
    for k in range(eval_iters):
        try:
            x, y = next(val_iter)
        except StopIteration:
            # If the loader runs out of data, re-initialize iterator
            val_iter = iter(val_loader)
            x, y = next(val_iter)
            
        x, y = x.to(device), y.to(device)
        logits, loss = model(x, y)
        losses[k] = loss.item()
        
    avg_loss = losses.mean().item()
    return avg_loss

def calculate_perplexity(loss: float) -> float:
    """
    Calculates the perplexity from the cross-entropy loss.
    Perplexity is a standard metric for language models (lower is better).
    """
    import math
    return math.exp(loss)
