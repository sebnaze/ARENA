# Distributed training basics

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Literal

import einops
import numpy as np
import torch as t
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import wandb
import itertools
from IPython.core.display import HTML
from IPython.display import display
from pathlib import Path
from jaxtyping import Float, Int
from torch import Tensor, optim
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from tqdm import tqdm

# -----------------   SETUP  -------------------------

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part3_optimization"
#root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
root_dir = Path("/Users/sebastin/Documents/perso/ARENA_training/ARENA_3.0")
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

MAIN = __name__ == "__main__"

import part3_optimization.tests as tests
from part2_cnns.solutions import Linear, ResNet34, get_resnet_for_feature_extraction
from part3_optimization.utils import plot_fn, plot_fn_with_points
from plotly_utils import bar, imshow, line

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

# ------------------------------------------------------------------

WORLD_SIZE = min(t.cuda.device_count(), 3)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12345"


def broadcast(tensor: Tensor, rank: int, world_size: int, src: int = 0):
    """
    Broadcast averaged gradients from rank 0 to all other ranks.
    """
    device = t.device("cpu")

    if rank == src:
        for other_rank in range(world_size):
            if other_rank != src:
                dist.send(tensor=tensor, dst=other_rank)
    else:
        received_tensor = t.zeros_like(tensor)
        dist.recv(received_tensor, src=src)
        tensor.copy_(received_tensor)
    
    



def run_broadcast(rank: int, world_size: int, broadcast):
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    #t.cuda.set_device(rank) <-- not needed for CPU tensors

    # Create a tensor for each rank with its rank as the value
    tensor = t.tensor([float(rank)], dtype=t.float32)#.cuda()
    print(f"Rank {rank} created tensor with value: {tensor.item()}")

    # Run broadcast operation (tensor is broadcasted from rank 0 to all ranks)
    broadcast(tensor, rank, world_size, src=0)

    # Check and print results on all ranks
    print(f"Rank {rank} broadcasted tensor: expected 0.0 (from rank 0), got {tensor}")
    t.testing.assert_close(tensor, t.full_like(tensor, 0.0))

    dist.destroy_process_group()


def test_broadcast(broadcast, world_size):
    world_size = world_size  # Number of processes (simulated ranks)
    mp.spawn(run_broadcast, args=(world_size, broadcast), nprocs=world_size, join=True)
    print("All tests in `test_broadcast` passed!")


if MAIN:
    WORLD_SIZE = 4  # simulate X processes
    test_broadcast(broadcast, WORLD_SIZE)
