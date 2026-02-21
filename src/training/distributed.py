import os
import torch
import torch.distributed as dist

def setup_distributed():
    """
    Initializes PyTorch Distributed Data Parallel environment variables 
    if training across multiple GPUs/nodes.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        
        return True, rank, local_rank, world_size
    return False, 0, 0, 1

def cleanup_distributed():
    """Cleans up the distributed environment."""
    if dist.is_initialized():
        dist.destroy_process_group()
