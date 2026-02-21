import os
import sys
import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tokenizer.bpe import BPETokenizer

def test_tokenizer_train_and_encode():
    tokenizer = BPETokenizer()
    text = "hello world python is amazing! I love language models. hello world!"
    
    # Base vocab is 256. We want to add 10 merges.
    tokenizer.train(text, vocab_size=266)
    
    # Tokenizer should have learned some merges
    assert len(tokenizer.vocab) == 266
    
    # Encode standard text
    encoded = tokenizer.encode("hello world python")
    
    # Ensure it's not mostly individual bytes if merges happened
    assert len(encoded) < len("hello world python".encode("utf-8"))
    
    # Decode back to test lossless reconstruction
    decoded = tokenizer.decode(encoded)
    assert decoded == "hello world python"
    
def test_tokenizer_save_load(tmp_path):
    tokenizer = BPETokenizer()
    text = "testing save and load functionality"
    tokenizer.train(text, vocab_size=270)
    
    save_path = tmp_path / "tokenizer.json"
    tokenizer.save(str(save_path))
    
    # Create new tokenizer and load
    new_tokenizer = BPETokenizer()
    new_tokenizer.load(str(save_path))
    
    # Verify vocab size and merges are identical
    assert len(tokenizer.vocab) == len(new_tokenizer.vocab)
    assert tokenizer.merges == new_tokenizer.merges
    
    # Verify encoding is identical
    assert tokenizer.encode(text) == new_tokenizer.encode(text)
