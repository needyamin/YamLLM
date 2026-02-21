import os
import sys
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

# Add project root to python path to allow src imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.transformer import Transformer
from src.tokenizer.bpe import BPETokenizer
from src.utils.config import LLMConfig

# Globals for model and tokenizer
model = None
tokenizer = None
device = "cuda" if torch.cuda.is_available() else "cpu"

app = FastAPI(
    title="Custom PyTorch LLM API",
    description="A minimalist LLM inference API similar to Ollama / vLLM",
    version="1.0.0"
)

class CompletionRequest(BaseModel):
    prompt: str
    max_tokens: int = Field(default=50, ge=1, le=512)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    top_k: int = Field(default=10, ge=1)

class CompletionResponse(BaseModel):
    generated_text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int

@app.on_event("startup")
async def load_model():
    """Loads the compiled transformer model and tokenizer into memory on startup."""
    global model, tokenizer
    
    checkpoint_path = "checkpoints/model_final.pt"
    tokenizer_path = "checkpoints/tokenizer.json"
    
    if not os.path.exists(checkpoint_path) or not os.path.exists(tokenizer_path):
        raise RuntimeError(
            "Checkpoints not found. Please train the model first by running:\n"
            "python scripts/train.py"
        )
        
    print(f"Loading Model onto {device}...")
    
    # 1. Load Tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load(tokenizer_path)
    
    # 2. Load Weight Checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    
    # 3. Initialize PyTorch Module
    model = Transformer(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print("API Server Ready.")

@app.post("/v1/completions", response_model=CompletionResponse)
async def generate_completion(request: CompletionRequest):
    """
    Generates text completions corresponding to the provided prompt.
    """
    if model is None or tokenizer is None:
        raise HTTPException(status_code=503, detail="Model currently loading or unavailable.")
        
    try:
        # Encode
        input_ids = tokenizer.encode(request.prompt)
        x = torch.tensor([input_ids], dtype=torch.long, device=device)
        
        # Inference Generation
        y = model.generate(
            x, 
            request.max_tokens, 
            temperature=request.temperature, 
            top_k=request.top_k
        )
        
        # Decode Output (trimming the original prompt length)
        output_tokens = y[0].tolist()
        new_tokens = output_tokens[len(input_ids):]
        completion = tokenizer.decode(new_tokens)
        
        return CompletionResponse(
            generated_text=completion,
            prompt_tokens=len(input_ids),
            completion_tokens=len(new_tokens),
            total_tokens=len(output_tokens)
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "device": device}

if __name__ == "__main__":
    print("Starting LLM Interface API via target loop...")
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
