import math
import torch
import torch.nn as nn
from tqdm import tqdm
from typing import Optional
from src.utils.config import LLMConfig
from src.evaluation.evaluator import evaluate_model # We will define this next

class Trainer:
    """
    Handles the training loop for the Large Language Model.
    """
    def __init__(self, model: nn.Module, config: LLMConfig, train_loader, val_loader=None):
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = config.device
        
        # Move model to the correct device
        self.model.to(self.device)
        
        # Setup the optimizer (AdamW is standard for Transformers)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95)
        )
        
        # Setup learning rate scheduler (Cosine Annealing with Warmup)
        # Using a simple LambdaLR for learning rate schedule:
        def get_lr_multiplier(step):
            if step < config.warmup_iters:
                return float(step) / float(max(1, config.warmup_iters))
            # Cosine decay down to 10% of learning rate
            progress = float(step - config.warmup_iters) / float(max(1, config.max_iters - config.warmup_iters))
            return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))
            
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, get_lr_multiplier)
        
    def train(self):
        """Runs the training loop."""
        self.model.train()
        
        # We use an iterator that continues infinitely
        def cycle(loader):
            while True:
                for data in loader:
                    yield data
                    
        train_iter = iter(cycle(self.train_loader))
        
        pbar = tqdm(range(1, self.config.max_iters + 1), desc="Training IT")
        for step in pbar:
            # 1. Get batch
            x, y = next(train_iter)
            x, y = x.to(self.device), y.to(self.device)
            
            # 2. Forward pass
            logits, loss = self.model(x, y)
            
            # 3. Backward pass
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            
            # Gradient clipping (standard practice to avoid exploding gradients)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # 4. Update weights
            self.optimizer.step()
            self.scheduler.step()
            
            # Evaluate periodically
            if self.val_loader is not None and step % self.config.eval_interval == 0:
                val_loss = evaluate_model(self.model, self.val_loader, self.config.eval_iters, self.device)
                pbar.write(f"Step {step} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f}")
                self.model.train()
            
            # Update progress bar every 10 steps
            if step % 10 == 0:
                lr = self.scheduler.get_last_lr()[0]
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr:.2e}"})
                
    def save_checkpoint(self, path: str):
        """Saves a model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, path)
