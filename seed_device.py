# seed_device.py
"""Get devices and set seed for reproducibility."""
import os
import torch
import random
import numpy as np
from typing import List, Union
import torch.distributed as dist

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

def set_seed(device: Union[str, torch.device], seed: int, cuda_deterministic: bool) -> None:
    """
    Set the seed for random number generators and configure CUDA settings for reproducibility.

    Args:
        device (Union[str, torch.device]): The device to set the seed for. Can be 'cuda' or 'cpu'.
        seed (int): The seed value to set.
        cuda_deterministic (bool): Whether to use deterministic settings for CUDA.
    """
    print("Setting seed:", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if isinstance(device, str) and device == "cuda" or isinstance(device, torch.device) and device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
        print("Set CUDA seed:", seed)

        if cuda_deterministic:
            # Set CUDA to deterministic mode
            print("Setting CUDA to deterministic mode")
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
        else:
            # Set CUDA to non-deterministic mode for potentially better performance
            print("Setting CUDA to non-deterministic mode")
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False

def setup_seed_reproducibility(seed: int, cuda_deterministic: bool = True) -> List[torch.device]:
    """
    Set up the seed and device configuration for reproducibility.

    Args:
        seed (int): The seed value to set.
        cuda_deterministic (bool): Whether to use deterministic settings for CUDA.

    Returns:
        List[torch.device]: A list of devices used for computations.
    """
    print("Setting up seed reproducibility")
    devices = get_device()
    print("Using devices:", devices)
    for device in devices:
        set_seed(device, seed, cuda_deterministic)
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