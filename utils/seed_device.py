# utils/seed_device.py
"""Get devices and set seed for reproducibility."""
import os
import string
import torch
import random
import numpy as np
from typing import List, Union
import torch.distributed as dist

def generate_random_string():
    characters = string.ascii_letters + string.digits
    print(''.join(random.choice(characters) for i in range(128)))

def get_device() -> List[torch.device]:
    """
    Get the computing devices (CUDA or CPU).

    Returns:
        List[torch.device]: A list of devices to use for computations.
    """
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"CUDA is available with {num_gpus} GPUs")
        devices = [torch.device(f"cuda:{i}") for i in range(num_gpus)]
    else:
        print("CUDA is not available")
        devices = [torch.device("cpu")]
    
    return devices

def setup_seed_reproducibility(seed: int, cuda_deterministic: bool = True) -> List[torch.device]:
    """
    优化后的种子设置函数，避免冗余操作
    """
    print("Setting up seed reproducibility")
    devices = get_device()
    print("Using devices:", devices)
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    cuda_devices = [d for d in devices if (isinstance(d, str) and d == "cuda") or 
                    (isinstance(d, torch.device) and d.type == "cuda")]
    
    if cuda_devices:
        torch.cuda.manual_seed_all(seed)
        print("Set CUDA seed:", seed)
        
        if cuda_deterministic:
            print("Setting CUDA to deterministic mode")
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        else:
            print("Setting CUDA to non-deterministic mode")
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
    
    return devices

def setup_multinodes(world_size) -> int:
    """
    Set up the environment for multi-node training using Distributed Data Parallel.
    
    Returns:
        int: The local rank of the process.
    """
    ip = os.environ["MASTER_ADDR"]
    port = os.environ["MASTER_PORT"]
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', 
                            init_method='tcp://{}:{}'.format(ip, port),
                            rank=int(os.environ['RANK']), 
                            world_size=world_size)
    print(f"Initialized process group with backend 'nccl' on local rank {local_rank}")
    return local_rank

def cleanup_multinodes():
    dist.destroy_process_group()