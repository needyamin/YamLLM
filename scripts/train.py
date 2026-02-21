import os
import sys
import torch

# Add project root to python path to allow src imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.config import LLMConfig
from src.tokenizer.bpe import BPETokenizer
from src.model.transformer import Transformer
from src.data.dataset import create_dataloader
from src.training.trainer import Trainer

def main():
    print("Initializing LLM Training...")
    
    # 1. Setup Configuration
    # We use a nano config for quick testing
    config = LLMConfig.create_nano()
    config.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {config.device}")
    
    # 2. Setup Tokenizer (Train a tiny one on dummy data for demonstration)
    print("Setting up Tokenizer...")
    dummy_text = "The quick brown fox jumps over the lazy dog. " * 1000
    tokenizer = BPETokenizer()
    # Vocabulary size of 256 (base bytes) + 50 merges = 306
    tokenizer.train(dummy_text, vocab_size=306)
    config.vocab_size = 306 
    
    # 3. Create Dataset and DataLoaders
    print("Preparing Data...")
    encoded_data = tokenizer.encode(dummy_text)
    
    # Split 90% train, 10% validation
    split_idx = int(len(encoded_data) * 0.9)
    train_data = encoded_data[:split_idx]
    val_data = encoded_data[split_idx:]
    
    train_loader = create_dataloader(train_data, config.max_seq_len, config.batch_size)
    val_loader = create_dataloader(val_data, config.max_seq_len, config.batch_size)
    
    # 4. Initialize Model
    print("Initializing Transformer Model...")
    model = Transformer(config)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {num_params / 1e6:.2f}M")
    
    # 5. Start Training
    print("Starting Training Loop...")
    trainer = Trainer(model, config, train_loader, val_loader)
    
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving checkpoint...")
        
    # Save final model
    os.makedirs("checkpoints", exist_ok=True)
    trainer.save_checkpoint("checkpoints/model_final.pt")
    tokenizer.save("checkpoints/tokenizer.json")
    print("Done! Model and tokenizer saved to /checkpoints")

if __name__ == "__main__":
    main()
