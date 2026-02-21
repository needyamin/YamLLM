import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

class BPETokenizer:
    """
    Byte-Pair Encoding (BPE) Tokenizer implementation from scratch.
    """
    def __init__(self):
        # The vocab maps from an integer ID to a bytes object
        self.vocab: Dict[int, bytes] = {}
        
        # The merges define the BPE pair merges
        self.merges: Dict[Tuple[int, int], int] = {}
        
        # We start by initializing the vocabulary with 256 raw byte tokens (0-255)
        for i in range(256):
            self.vocab[i] = bytes([i])
            
    def _get_stats(self, ids: List[int]) -> Dict[Tuple[int, int], int]:
        """Counts the frequencies of adjacent pairs of token IDs."""
        counts = defaultdict(int)
        for pair in zip(ids, ids[1:]):
            counts[pair] += 1
        return counts
        
    def _merge(self, ids: List[int], pair: Tuple[int, int], idx: int) -> List[int]:
        """Replaces all consecutive occurrences of `pair` in `ids` with `idx`."""
        new_ids = []
        i = 0
        while i < len(ids):
            # If we are not at the final ID and we found the pair...
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(idx)
                i += 2
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids

    def train(self, text: str, vocab_size: int, verbose: bool = False):
        """
        Trains the tokenizer on a given text sequence until the vocab_size is reached.
        """
        assert vocab_size >= 256, "vocab_size must be >= 256 (the base byte size)"
        
        num_merges = vocab_size - 256
        
        # Convert text to raw UTF-8 bytes then to a list of integers [0-255]
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        
        for i in range(num_merges):
            stats = self._get_stats(ids)
            if not stats: 
                break # Nothing more to merge
            
            # Find the most frequent pair
            best_pair = max(stats, key=stats.get)
            
            # Form the new token ID
            new_idx = 256 + i
            
            # Add to our merges and vocab
            self.merges[best_pair] = new_idx
            self.vocab[new_idx] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            
            # Actually perform the merge on our ids
            ids = self._merge(ids, best_pair, new_idx)
            
            if verbose:
                print(f"Merge {i+1}/{num_merges}: {best_pair} -> {new_idx} ({self.vocab[new_idx]})")
                
    def encode(self, text: str) -> List[int]:
        """Encodes a string into a list of token IDs."""
        text_bytes = text.encode("utf-8")
        ids = list(text_bytes)
        
        # Keep applying merges until there is nothing left to merge
        while len(ids) >= 2:
            stats = self._get_stats(ids)
            # Find the pair in current IDs that also exists in our known merges with the lowest ID (first merged)
            pair = min(stats.keys(), key=lambda p: self.merges.get(p, float("inf")))
            
            # If this pair is not even in our merges, we are done
            if pair not in self.merges:
                break
                
            idx = self.merges[pair]
            ids = self._merge(ids, pair, idx)
            
        return ids
        
    def decode(self, ids: List[int]) -> str:
        """Decodes a list of token IDs back into a string."""
        tokens = b"".join(self.vocab[i] for i in ids)
        # Using "replace" handles potential invalid utf-8 sequences that could arise 
        # from incomplete token generation during inference
        return tokens.decode("utf-8", errors="replace")
        
    def save(self, filepath: str):
        """Saves the tokenizer merges to a file."""
        # Convert tuples to string keys for JSON serialization
        save_merges = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(save_merges, f)
            
    def load(self, filepath: str):
        """Loads tokenizer merges from a file."""
        with open(filepath, "r", encoding="utf-8") as f:
            load_merges = json.load(f)
            
        self.merges = {tuple(map(int, k.split(","))): v for k, v in load_merges.items()}
        
        # Reconstruct vocabulary
        for i in range(256):
            self.vocab[i] = bytes([i])
            
        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0] + self.vocab[p1]
