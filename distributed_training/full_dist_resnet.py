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
root_dir = Path("/home/sebastin/Documents/ARENA/ARENA_3.0")
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
            


def all_reduce(tensor, rank, world_size, op: Literal["sum", "mean"] = "sum"):
    """
    Allreduce the tensor across all ranks, using 0 as the initial gathering rank.
    """
    reduce(tensor, rank,  world_size, op=op)
    broadcast(tensor, rank, world_size)


# -----------------   Optimization  ------------------------- #

class AdamW:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float = 0.001,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-08,
        weight_decay: float = 0.0,
    ):
        """Implements Adam.

        Like the PyTorch version, but assumes amsgrad=False and maximize=False
            https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html
        """
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.lmda = weight_decay
        self.t = 1

        self.m = [t.zeros_like(p) for p in self.params]
        self.v = [t.zeros_like(p) for p in self.params]

    def zero_grad(self) -> None:
        for p in self.params:
            p.grad = None

    @t.inference_mode()
    def step(self) -> None:
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue  # No gradient for this parameter, skip it
            else:
                grad = param.grad

            # Apply weight decay (L2 regularization) directly to the parameters (decoupled weight decay)
            if self.lmda != 0:
                param -= self.lr * self.lmda * param
                
            # Update first (m, mean) and second (v, variance) momentums by moving average
            self.m[i] = self.beta1*self.m[i] + (1-self.beta1)*grad
            self.v[i] = self.beta2*self.v[i] + (1-self.beta2)*grad**2
            
            # Bias correction
            m_corrected = self.m[i] / (1 - self.beta1**self.t)
            v_corrected = self.v[i] / (1 - self.beta2**self.t)

            param -= self.lr*m_corrected/(t.sqrt(v_corrected)+self.eps)
        
        self.t += 1

    def __repr__(self) -> str:
        return f"AdamW(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, eps={self.eps}, weight_decay={self.lmda})"



# ----------------------   Dataset  ------------------------- #
def get_cifar() -> tuple[datasets.CIFAR10, datasets.CIFAR10]:
    """Returns CIFAR-10 train and test sets."""
    cifar_trainset = datasets.CIFAR10(exercises_dir / "data", train=True, download=True, transform=IMAGENET_TRANSFORM)
    cifar_testset = datasets.CIFAR10(exercises_dir / "data", train=False, download=True, transform=IMAGENET_TRANSFORM)
    return cifar_trainset, cifar_testset


IMAGE_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

IMAGENET_TRANSFORM = transforms.Compose(
    [
        transforms.ToTensor(),
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
)


# -----------------   ResNet Finetuning  ------------------------- #

def get_untrained_resnet(n_classes: int) -> ResNet34:
    """
    Gets untrained resnet using code from part2_cnns.solutions (you can replace this with your
    implementation).
    """
    resnet = ResNet34()
    resnet.out_layers[-1] = Linear(resnet.out_features_per_group[-1], n_classes)
    return resnet


@dataclass
class ResNetFinetuningArgs:
    n_classes: int = 10
    batch_size: int = 128
    epochs: int = 3
    learning_rate: float = 1e-3
    weight_decay: float = 0.0


@dataclass
class WandbResNetFinetuningArgs(ResNetFinetuningArgs):
    """Contains new params for use in wandb.init, as well as all the ResNetFinetuningArgs params."""

    wandb_project: str | None = "day3-resnet"
    wandb_name: str | None = None


@dataclass
class DistResNetTrainingArgs(WandbResNetFinetuningArgs):
    world_size: int = 1
    wandb_project: str | None = "day3-resnet-dist-training"


class DistResNetTrainer:
    args: DistResNetTrainingArgs

    def __init__(self, args: DistResNetTrainingArgs, rank: int):
        self.args = args
        self.rank = rank
        self.device = t.device("cpu")#(f"cuda:{rank}")
        self.examples_seen: int = 0  # tracking examples seen (used as step for wandb)
        self.best_accuracy: float = 0.0

    def pre_training_setup(self):
        # Get data
        cifar_trainset, cifar_testset = get_cifar()
        self.train_sampler = DistributedSampler(cifar_trainset, num_replicas=self.args.world_size, rank=self.rank)
        self.test_sampler = DistributedSampler(cifar_testset, num_replicas=self.args.world_size, rank=self.rank)
        self.train_loader = DataLoader(cifar_trainset, batch_size=self.args.batch_size, sampler=self.train_sampler)
        self.test_loader = DataLoader(cifar_testset, batch_size=self.args.batch_size, sampler=self.test_sampler)

        # Get model and optimizer
        self.model = get_untrained_resnet(self.args.n_classes).to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)
        
        # Initialize wandb on rank 0 only to avoid duplicate runs
        if self.rank == 0:
            wandb.init(project=self.args.wandb_project, name=self.args.wandb_name, config=self.args.__dict__)
            wandb.watch(self.model, log="all", log_freq=1000)

    def training_step(self, imgs: Tensor, labels: Tensor) -> Tensor:
        """
        Runs a training step on this batch of data, returning the loss.
        """
        if self.rank == 0:
            t0 = time.perf_counter()
        
        self.model.train()
        imgs, labels = imgs.to(self.device), labels.to(self.device)
        self.examples_seen += imgs.shape[0]*self.args.world_size  # Update examples seen (accounting for all ranks)
        
        logits = self.model(imgs)
        
        if self.rank == 0:
            t1 = time.perf_counter()
            print(f"Forward pass time: {t1-t0:.4f} seconds")
            wandb.log({"fwd_time": t1-t0}, step=self.examples_seen)

        loss = F.cross_entropy(logits, labels)
        loss.backward()

        if self.rank == 0:
            t2 = time.perf_counter()
            print(f"Backward pass time: {t2-t1:.4f} seconds")
            wandb.log({"bwd_time": t2-t1}, step=self.examples_seen)

        for param in self.model.parameters():
            if param.grad is not None:
                all_reduce(param.grad, self.rank, self.args.world_size, op="mean")

        if self.rank == 0:
            t3 = time.perf_counter()
            print(f"Gradient synchronization time: {t3-t2:.4f} seconds")
            wandb.log({"grad_sync_time": t3-t2}, step=self.examples_seen)
        
        t.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0) # <-- suggested by claude
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        if self.rank == 0:
            print(f"Step {self.examples_seen}, loss: {loss.item():.4f}")
            wandb.log({"train_loss": loss.item()}, step=self.examples_seen)
        
        return loss.detach().cpu()  # Return loss as a CPU tensor for logging

    @t.inference_mode()
    def evaluate(self) -> float:
        """
        Evaluates the model on the test set, returning the accuracy.
        """
        self.model.eval()
        correct, total = 0, 0
        for imgs, labels in self.test_loader:
            imgs, labels = imgs.to(self.device), labels.to(self.device)
            logits = self.model(imgs)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        # Aggregate correct and total counts across ranks
        correct_tensor = t.tensor(correct, device=self.device)
        total_tensor = t.tensor(total, device=self.device)
        all_reduce(correct_tensor, self.rank, self.args.world_size, op="sum")
        all_reduce(total_tensor, self.rank, self.args.world_size, op="sum")

        accuracy = correct_tensor.item() / total_tensor.item()
        
        if self.rank == 0:
            print(f"Test Accuracy: {accuracy:.4f}")
            wandb.log({"test_accuracy": accuracy}, step=self.examples_seen)

            if accuracy > self.best_accuracy:
                self.best_accuracy = accuracy
                print(f"New best accuracy: {accuracy:.4f}")
                #wandb.run.summary["best_accuracy"] = accuracy
                t.save(self.model.state_dict(), f"best_model.pt")  # Save model checkpoint

        return accuracy

    def train(self):
        self.pre_training_setup()
        for epoch in range(self.args.epochs):
            if self.rank == 0:
                print(f"Starting epoch {epoch+1}/{self.args.epochs}")
                t0 = time.perf_counter()

            self.train_sampler.set_epoch(epoch)  # Shuffle data differently each epoch
            for imgs, labels in tqdm(self.train_loader, desc=f"Rank {self.rank} Epoch {epoch+1}/{self.args.epochs}"):
                self.training_step(imgs, labels)
            
            if self.rank == 0:
                t1 = time.perf_counter()
                print(f"Epoch {epoch+1} completed in {t1-t0:.2f} seconds")
                wandb.log({"epoch_train_time": t1-t0}, step=self.examples_seen)
            
            self.evaluate()

            if self.rank == 0:
                t2 = time.perf_counter()
                print(f"Epoch {epoch+1} evaluation time: {t2-t1:.2f} seconds")
                wandb.log({"epoch_eval_time": t2-t1}, step=self.examples_seen)
                print(f"Finished epoch {epoch+1}/{self.args.epochs}")
                print("-"*50)


def dist_train_resnet_from_scratch(rank, world_size):
    #dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    total_cores = int(np.floor(os.cpu_count() * 0.8)) # <- use 80% of available CPU cores to avoid overloading 
    world_size = dist.get_world_size()
    threads_per_process = max(1, total_cores // world_size)
    
    # 2. Force PyTorch to limit its internal thread pool
    t.set_num_threads(threads_per_process)
    t.set_num_interop_threads(threads_per_process)

    args = DistResNetTrainingArgs(world_size=world_size)
    trainer = DistResNetTrainer(args, rank)
    trainer.train()
    dist.destroy_process_group()


if MAIN:
    world_size = 8#t.cuda.device_count()
    mp.spawn(
        dist_train_resnet_from_scratch,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )
