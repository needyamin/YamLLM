import torch
from torch.utils.data import Dataset, DataLoader
from src.tokenizer.bpe import BPETokenizer

class LLMDataset(Dataset):
    """
    Dataset for next-token prediction language modeling.
    Iterates through sequences of max_seq_len tokens.
    """
    def __init__(self, data: list[int], max_seq_len: int):
        self.data = torch.tensor(data, dtype=torch.long)
        self.max_seq_len = max_seq_len

    def __len__(self):
        # We need pairs of length `max_seq_len` for both input and target (offset by 1)
        # So we divide the total length by the length of a single sequence block
        return len(self.data) - self.max_seq_len

    def __getitem__(self, idx):
        # Chunk of sequence
        x = self.data[idx : idx + self.max_seq_len]
        
        # Target is the next token for each token in the sequence
        y = self.data[idx + 1 : idx + self.max_seq_len + 1]
        
        return x, y

def create_dataloader(data: list[int], max_seq_len: int, batch_size: int, shuffle: bool = True):
    """Returns a PyTorch DataLoader for the given data."""
    dataset = LLMDataset(data, max_seq_len)
    
    # In distributed training, we would use DistributedSampler here
    # sampler = DistributedSampler(dataset)
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=0, # keep at 0 for simplicity/Windows compatibility
        pin_memory=True
    )
