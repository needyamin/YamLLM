import os
import sys
import torch

# Add project root to python path to allow src imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.transformer import Transformer
from src.tokenizer.bpe import BPETokenizer
from src.utils.config import LLMConfig

def main():
    print("Loading Trained LLM...")
    
    # Hardcoded to the paths saved by train.py
    checkpoint_path = "checkpoints/model_final.pt"
    tokenizer_path = "checkpoints/tokenizer.json"
    
    if not os.path.exists(checkpoint_path) or not os.path.exists(tokenizer_path):
        print("Error: Checkpoints not found. Please run 'python scripts/train.py' first.")
        return
        
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Load Tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)
    
    # 2. Load Checkpoint Data (Config and Weights)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # 3. Initialize Model and Load Weights
    model = Transformer(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("Model loaded successfully.")
    
    # 4. Interactive Generation Loop
    print("\n--- LLM Interactive Generation ---")
    print("Type 'quit' to exit.")
    
    while True:
        prompt = input("\nPrompt: ")
        if prompt.lower() in ['quit', 'exit']:
            break
            
        print("Generating...", end="", flush=True)
        
        # Encode prompt
        input_ids = tokenizer.encode(prompt)
        x = torch.tensor([input_ids], dtype=torch.long, device=device)
        
        # Generate completion
        max_new_tokens = 50
        y = model.generate(x, max_new_tokens, temperature=0.8, top_k=10)
        
        # Decode and print
        output_tokens = y[0].tolist()
        completion = tokenizer.decode(output_tokens)
        
        print("\n\n" + completion)

if __name__ == "__main__":
    main()
