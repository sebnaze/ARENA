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

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12345"

def broadcast(tensor: Tensor, rank: int, world_size: int, src: int = 0):
    """
    Broadcast averaged gradients from rank 0 to all other ranks.
    """

    if rank == src:
        for other_rank in range(world_size):
            if other_rank != src:
                dist.send(tensor=tensor, dst=other_rank)
    else:
        received_tensor = t.zeros_like(tensor)
        dist.recv(received_tensor, src=src)
        tensor.copy_(received_tensor)
    

def reduce(tensor, rank, world_size, dst=0, op: Literal["sum", "mean"] = "sum"):
    """
    Reduces gradients to rank `dst`, so this process contains the sum or mean of all tensors across
    processes.
    """
    if rank==dst:
        ts = [tensor]
        for other_rank in range(world_size):
            if other_rank!=rank:
                received_tensor = t.zeros_like(tensor)
                dist.recv(received_tensor, src=other_rank)
                ts.append(received_tensor)
                tensor += received_tensor
        if op=="mean":
            tensor /= world_size
        
    else:
        dist.send(tensor, dst=dst)
        #received_tensor = t.zeros_like(tensor)
        #dist.recv(received_tensor, src=dst)
        #tensor.copy_(received_tensor)
        
            


def all_reduce(tensor, rank, world_size, op: Literal["sum", "mean"] = "sum"):
    """
    Allreduce the tensor across all ranks, using 0 as the initial gathering rank.
    """
    reduce(tensor, rank,  world_size, op=op)
    broadcast(tensor, rank, world_size)



def run_reduce(rank: int, world_size: int, reduce):
    #dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)
    #t.cuda.set_device(rank)

    tensor_list = [
        t.tensor([0, 0], dtype=t.float32),
        t.tensor([1, 2], dtype=t.float32),
        t.tensor([10, 20], dtype=t.float32),
    ]

    for op in ["sum", "mean"]:
        tensor = tensor_list[rank].clone()#.cuda() <-- clone to avoid in-place modification of the original tensor_list

        # Run reduce operation
        reduce(tensor, rank, world_size, dst=0, op=op)

        # Check and print results on all ranks
        expected = (
            (sum(tensor_list[:world_size]) / (1 if op == "sum" else world_size))
            if rank == 0
            else tensor.cpu()
        )
        print(
            f"Rank {rank}, {op=}, expected {'' if rank == 0 else 'non-'}reduced {expected}, got {tensor.cpu()}"
        )
        t.testing.assert_close(tensor.cpu(), expected)

    dist.destroy_process_group()


def test_reduce(reduce, world_size):
    world_size = world_size  # Number of processes (simulated ranks)
    print("Running reduce on dst=0, with initial tensors: [0, 0], [1, 2], [10, 20]")
    mp.spawn(run_reduce, args=(world_size, reduce), nprocs=world_size, join=True)
    print("All tests in `test_reduce` passed!\n")


def run_all_reduce(rank: int, world_size: int, all_reduce):
    #dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    #t.cuda.set_device(rank)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    tensor_list = [
        t.tensor([0, 0], dtype=t.float32),
        t.tensor([1, 2], dtype=t.float32),
        t.tensor([10, 20], dtype=t.float32),
    ]

    for op in ["sum", "mean"]:
        tensor = tensor_list[rank].clone()#.cuda()

        # Run all_reduce operation
        all_reduce(tensor, rank, world_size, op=op)

        # Check and print results on all ranks
        expected = sum(tensor_list[:world_size]) / (1 if op == "sum" else world_size)
        print(f"Rank {rank}, {op=}, expected non-reduced {expected}, got {tensor.cpu()}")
        t.testing.assert_close(tensor.cpu(), expected)

    dist.destroy_process_group()


def test_all_reduce(all_reduce, world_size):
    world_size = world_size  # Number of processes (simulated ranks)
    print("Running all_reduce, with initial tensors: [0, 0], [1, 2], [10, 20]")
    mp.spawn(run_all_reduce, args=(world_size, all_reduce), nprocs=world_size, join=True)
    print("All tests in `test_all_reduce` passed!")


class SimpleModel(t.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.param = t.nn.Parameter(t.tensor([2.0]))

    def forward(self, x: Tensor):
        return x - self.param


def run_simple_model(rank, world_size):
    #dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    device = t.device("cpu") #(f"cuda:{rank}")
    model = SimpleModel().to(device)  # Move the model to the device corresponding to this process
    optimizer = t.optim.SGD(model.parameters(), lr=0.1)

    input = t.tensor([rank], dtype=t.float32, device=device)
    output = model(input)
    loss = output.pow(2).sum()
    loss.backward()  # Each rank has separate gradients at this point

    print(f"Rank {rank}, before all_reduce, grads: {model.param.grad=}")
    all_reduce(model.param.grad, rank, world_size)  # Synchronize gradients
    print(f"Rank {rank}, after all_reduce, synced grads (summed over processes): {model.param.grad=}")

    optimizer.step()  # Step with the optimizer (this will update all models the same way)
    print(f"Rank {rank}, new param: {model.param.data}")

    dist.destroy_process_group()


if MAIN:
    world_size = 2
    mp.spawn(
        run_simple_model,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )
