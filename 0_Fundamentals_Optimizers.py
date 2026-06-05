#%%
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

# %%
def pathological_curve_loss(x: Tensor, y: Tensor):
    # Example of a pathological curvature. There are many more possible, feel free to experiment here!
    x_loss = t.tanh(x) ** 2 + 0.01 * t.abs(x)
    y_loss = t.sigmoid(y)
    return x_loss + y_loss


plot_fn(pathological_curve_loss, min_points=[(0, "y_min")])


# %%
def opt_fn_with_sgd(
    fn: Callable, xy: Float[Tensor, "2"], lr=0.001, momentum=0.98, n_iters: int = 100
) -> Float[Tensor, "n_iters 2"]:
    """
    Optimize the a given function starting from the specified point.

    xy: shape (2,). The (x, y) starting point.
    n_iters: number of steps.
    lr, momentum: parameters passed to the torch.optim.SGD optimizer.

    Return: (n_iters+1, 2). The (x, y) values, from initial values to values after step `n_iters`.
    """
    # Make sure tensor has requires_grad=True, otherwise it can't be optimized (more on this tomorrow!)
    assert xy.requires_grad

    m = 0 #momentum
    out = t.zeros((n_iters + 1, 2))
    out[0] = xy.detach().clone()
    for i in range(n_iters):
        loss = fn(xy[0], xy[1])
        loss.backward()
        grad = xy.grad
        m = momentum * m + grad
        with t.no_grad():
            xy -= lr * m
        xy.grad.zero_()  # Clear the gradient for the next iteration
        out[i+1] = xy.detach().clone()
    return out
    


points = []

optimizer_list = [
    (optim.SGD, {"lr": 0.1, "momentum": 0.0}),
    (optim.SGD, {"lr": 0.02, "momentum": 0.99}),
]

for optimizer_class, params in optimizer_list:
    xy = t.tensor([2.5, 2.5], requires_grad=True)
    xys = opt_fn_with_sgd(pathological_curve_loss, xy=xy, lr=params["lr"], momentum=params["momentum"])
    points.append((xys, optimizer_class, params))
    print(f"{params=}, last point={xys[-1]}")

plot_fn_with_points(pathological_curve_loss, points=points, min_points=[(0, "y_min")])

# %% SGD
class SGD:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ):
        """Implements SGD with momentum.

        Like the PyTorch version, but assume nesterov=False, maximize=False, and dampening=0
            https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
        """
        self.params = list(params)  # turn params into a list (it might be a generator, so iterating over it empties it)
        self.lr = lr
        self.mu = momentum
        self.lmda = weight_decay

        self.b = [t.zeros_like(p) for p in self.params]

    def zero_grad(self) -> None:
        """Zeros all gradients of the parameters in `self.params`."""
        for param in self.params:
            param.grad = None

    @t.inference_mode()
    def step(self) -> None:
        """Performs a single optimization step of the SGD algorithm."""
        for i, param in enumerate(self.params):
            if param.grad is None:
                continue  # No gradient for this parameter, skip it

            # Apply weight decay (L2 regularization) if specified
            if self.lmda != 0:
                grad = param.grad + self.lmda * param
            else:
                grad = param.grad

            # Update the momentum buffer
            self.b[i] = self.mu * self.b[i] + grad

            # Update the parameter
            param -= self.lr * self.b[i]

    def __repr__(self) -> str:
        return f"SGD(lr={self.lr}, momentum={self.mu}, weight_decay={self.lmda})"


tests.test_sgd(SGD)

# %% RMSprop
class RMSprop:
    def __init__(
        self,
        params: Iterable[t.nn.parameter.Parameter],
        lr: float = 0.01,
        alpha: float = 0.99,
        eps: float = 1e-08,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
    ):
        """Implements RMSprop.

        Like the PyTorch version, but assumes centered=False
            https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html
        """
        self.params = list(params)  # turn params into a list (because it might be a generator)
        self.lr = lr
        self.eps = eps
        self.mu = momentum
        self.lmda = weight_decay
        self.alpha = alpha

        self.b = [t.zeros_like(p) for p in self.params]
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

            # Apply weight decay (L2 regularization) if specified
            if self.lmda != 0:
                grad = param.grad + self.lmda * param
                
            # Track variance of past gradients 
            self.v[i] = self.alpha*self.v[i] + (1-self.alpha)*grad**2
            
            # Update the momentum buffer
            if self.mu != 0:
                self.b[i] = self.mu * self.b[i] + grad/(t.sqrt(self.v[i])+self.eps)
                # Update the parameter
                param -= self.lr * self.b[i]
            else:
                param -= self.lr*grad/(t.sqrt(self.v[i])+self.eps)
            

    def __repr__(self) -> str:
        return (
            f"RMSprop(lr={self.lr}, eps={self.eps}, momentum={self.mu}, weight_decay={self.lmda}, alpha={self.alpha})"
        )


tests.test_rmsprop(RMSprop)

# %% Adam
class Adam:
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
            https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
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

            # Apply weight decay (L2 regularization) if specified
            if self.lmda != 0:
                grad = param.grad + self.lmda * param
                
            # Update first (m) and second (v) momentums
            self.m[i] = self.beta1*self.m[i] + (1-self.beta1)*grad
            self.v[i] = self.beta2*self.v[i] + (1-self.beta2)*grad**2
            
            # Bias correction
            m_corrected = self.m[i] / (1 - self.beta1**self.t)
            v_corrected = self.v[i] / (1 - self.beta2**self.t)

            param -= self.lr*m_corrected/(t.sqrt(v_corrected)+self.eps)
        
        self.t += 1

    def __repr__(self) -> str:
        return f"Adam(lr={self.lr}, beta1={self.beta1}, beta2={self.beta2}, eps={self.eps}, weight_decay={self.lmda})"


tests.test_adam(Adam)

# %%
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


tests.test_adamw(AdamW)

# %%
def opt_fn(
    fn: Callable,
    xy: Tensor,
    optimizer_class,
    optimizer_hyperparams: dict,
    n_iters: int = 100,
) -> Tensor:
    """Optimize the a given function starting from the specified point.

    optimizer_class: one of the optimizers you've defined, either SGD, RMSprop, or Adam
    optimzer_kwargs: keyword arguments passed to your optimiser (e.g. lr and weight_decay)
    """
    assert xy.requires_grad

    optimizer = optimizer_class([xy], **optimizer_hyperparams)

    xy_list = [xy.detach().clone()]  # so that we don't unintentionally modify past values in `xy_list`

    for i in range(n_iters):
        fn(xy[0], xy[1]).backward()
        optimizer.step()
        optimizer.zero_grad()
        xy_list.append(xy.detach().clone())

    return t.stack(xy_list)


points = []

optimizer_list = [
    (SGD, {"lr": 0.03, "momentum": 0.99}),
    (RMSprop, {"lr": 0.02, "alpha": 0.99, "momentum": 0.8}),
    (Adam, {"lr": 0.2, "betas": (0.99, 0.99), "weight_decay": 0.005}),
    (AdamW, {"lr": 0.2, "betas": (0.99, 0.99), "weight_decay": 0.005}),
]

for optimizer_class, params in optimizer_list:
    xy = t.tensor([2.5, 2.5], requires_grad=True)
    xys = opt_fn(
        pathological_curve_loss,
        xy=xy,
        optimizer_class=optimizer_class,
        optimizer_hyperparams=params,
    )
    points.append((xys, optimizer_class, params))

plot_fn_with_points(pathological_curve_loss, min_points=[(0, "y_min")], points=points)


# %%
def bivariate_gaussian(x, y, x_mean=0.0, y_mean=0.0, x_sig=1.0, y_sig=1.0):
    norm = 1 / (2 * np.pi * x_sig * y_sig)
    x_exp = 0.5 * ((x - x_mean) ** 2) / (x_sig**2)
    y_exp = 0.5 * ((y - y_mean) ** 2) / (y_sig**2)
    return norm * t.exp(-x_exp - y_exp)


means = [(1.0, -0.5), (-1.0, 0.5), (-0.5, -0.8)]


def neg_trimodal_func(x, y):
    """
    This function has 3 global minima, at `means`. Unstable methods can overshoot these minima, and
    non-adaptive methods can fail to converge to them in the first place given how shallow the
    gradients are everywhere except in the close vicinity of the minima.
    """
    z = -bivariate_gaussian(x, y, x_mean=means[0][0], y_mean=means[0][1], x_sig=0.2, y_sig=0.2)
    z -= bivariate_gaussian(x, y, x_mean=means[1][0], y_mean=means[1][1], x_sig=0.2, y_sig=0.2)
    z -= bivariate_gaussian(x, y, x_mean=means[2][0], y_mean=means[2][1], x_sig=0.2, y_sig=0.2)
    return z


plot_fn(neg_trimodal_func, x_range=(-2, 2), y_range=(-2, 2), min_points=means)
#%%
def rosenbrocks_banana_func(x: Tensor, y: Tensor, a=1, b=100) -> Tensor:
    """
    This function has a global minimum at `(a, a)` so in this case `(1, 1)`. It's characterized by a
    long, narrow, parabolic valley (parameterized by `y = x**2`). Various gradient descent methods
    have trouble navigating this valley because they often oscillate unstably (gradients from the
    `b`-term dwarf the gradients from the `a`-term).

    See more on this function: https://en.wikipedia.org/wiki/Rosenbrock_function.
    """
    return (a - x) ** 2 + b * (y - x**2) ** 2 + 1


plot_fn(
    rosenbrocks_banana_func,
    x_range=(-2.5, 2.5),
    y_range=(-2, 4),
    z_range=(0, 100),
    min_points=[(1, 1)],
)

# %%
optimizer_list = [
    (SGD, {"lr": 0.1, "momentum": 0.5}),
    (RMSprop, {"lr": 0.1, "alpha": 0.99, "momentum": 0.5}),
    (Adam, {"lr": 0.1, "betas": (0.9, 0.999)}),
]

points = []
for optimizer_class, params in optimizer_list:
    xy = t.tensor([1.0, 1.0], requires_grad=True)
    xys = opt_fn(neg_trimodal_func, xy=xy, optimizer_class=optimizer_class, optimizer_hyperparams=params)
    points.append((xys, optimizer_class, params))

plot_fn_with_points(neg_trimodal_func, points=points, x_range=(-2, 2), y_range=(-2, 2), min_points=means)

#%%
optimizer_list = [
    (SGD, {"lr": 0.001, "momentum": 0.99}),
    (Adam, {"lr": 0.1, "betas": (0.9, 0.999)}),
]

points = []
for optimizer_class, params in optimizer_list:
    xy = t.tensor([-1.5, 2.5], requires_grad=True)
    xys = opt_fn(
        rosenbrocks_banana_func, xy=xy, optimizer_class=optimizer_class, optimizer_hyperparams=params, n_iters=500
    )
    points.append((xys, optimizer_class, params))

plot_fn_with_points(
    rosenbrocks_banana_func, x_range=(-2.5, 2.5), y_range=(-2, 4), z_range=(0, 100), min_points=[(1, 1)], points=points
)

# %% Multiple parameter groups in SGD
class SGD:
    def __init__(self, params, **kwargs):
        """Implements SGD with momentum.

        Accepts parameters in groups, or an iterable.

        Like the PyTorch version, but assume nesterov=False, maximize=False, and dampening=0
            https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
        """
        # Deal with case where we didn't supply groups, so we just make it into a single dictionary
        if not isinstance(params, (list, tuple)):
            params = [{"params": params}]

        # Make sure each group["params"] is a list of params not a generator (so we don't iterate
        # over & destroy it!)
        for p in params:
            p["params"] = list(p["params"])

        self.param_groups = []

        # YOUR CODE HERE - fill in `self.param_groups`
        for group in params:
            group_dict = {
                "params": group["params"],
                "lr": kwargs.get("lr", 0.001),
                "momentum": kwargs.get("momentum", 0.0),
                "weight_decay": kwargs.get("weight_decay", 0.0),
            }
            self.param_groups.append(group_dict)

    def zero_grad(self) -> None:
        """Zeros all gradients of the parameters in `self.params`."""
        for group in self.param_groups:
            for param in group["params"]:
                param.grad = None

    @t.inference_mode()
    def step(self) -> None:
        """Performs a single optimization step of the SGD algorithm."""
        for group in self.param_groups:
            lr = group["lr"]
            momentum = group["momentum"]
            weight_decay = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue  # No gradient for this parameter, skip it

                # Apply weight decay (L2 regularization) if specified
                if weight_decay != 0:
                    grad = param.grad + weight_decay * param
                else:
                    grad = param.grad

                # Update the parameter using SGD with momentum
                if momentum != 0:
                    if "momentum_buffer" not in group:
                        group["momentum_buffer"] = {}
                    buf = group["momentum_buffer"].get(param, t.zeros_like(param))
                    buf.mul_(momentum).add_(grad) # <- this is the same as `buf = momentum * buf + grad`, but done in-place since we need to update the buffer value in `self.param_groups`
                    group["momentum_buffer"][param] = buf
                    param -= lr * buf
                else:
                    param -= lr * grad


tests.test_sgd_param_groups(SGD)

# %%. SOLUTION
class SGD:
    def __init__(self, params, **kwargs):
        """Implements SGD with momentum.

        Accepts parameters in groups, or an iterable.

        Like the PyTorch version, but assume nesterov=False, maximize=False, and dampening=0
            https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#torch.optim.SGD
        """
        # Deal with case where we didn't supply groups, so we just make it into a single dictionary
        if not isinstance(params, (list, tuple)):
            params = [{"params": params}]

        # Make sure each group["params"] is a list of params not a generator (so we don't iterate
        # over & destroy it!)
        for p in params:
            p["params"] = list(p["params"])

        self.param_groups = []

        for param_group in params:
            # Set hyperparameters hierarchically: specified for group > specified as a keyword
            # argument > default value. We do this via dict merge (right takes precedence over left).
            param_group = {"momentum": 0.0, "weight_decay": 0.0, **kwargs, **param_group}

            # Check that "lr" is supplied
            assert "lr" in param_group, "Error: one of the param groups didn't specify 'lr'"

            # Set "params" and "b" in our group
            param_group["b"] = [t.zeros_like(p) for p in param_group["params"]]

            self.param_groups.append(param_group)

    def zero_grad(self) -> None:
        for param_group in self.param_groups:
            for p in param_group["params"]:
                p.grad = None

    @t.inference_mode()
    def step(self) -> None:
        # loop through each param group

        for param_group in self.param_groups:
            # Get hparams for this group
            lmda = param_group["weight_decay"]
            mu = param_group["momentum"]
            lr = param_group["lr"]

            # Same code as for SGD implementation before, but using group-specific hparams
            for b, theta in zip(param_group["b"], param_group["params"]):
                g = theta.grad
                if lmda != 0:
                    g = g + lmda * theta  # not inplace, since we're not changing theta.grad
                if mu != 0:
                    b.copy_(mu * b + g)  # needs to be inplace, since we're changing `self.b` value
                    g = b
                theta -= lr * g  # inplace operation, to modify params


tests.test_sgd_param_groups(SGD)

















# %% --------------------------------
#     Weights and biases
# -----------------------------------
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


cifar_trainset, cifar_testset = get_cifar()

imshow(
    cifar_trainset.data[:15],
    facet_col=0,
    facet_col_wrap=5,
    facet_labels=[cifar_trainset.classes[i] for i in cifar_trainset.targets[:15]],
    title="CIFAR-10 images",
    height=600,
    width=1000,
)

# %% Resnet Finetuning
@dataclass
class ResNetFinetuningArgs:
    n_classes: int = 10
    batch_size: int = 128
    epochs: int = 3
    learning_rate: float = 1e-3
    weight_decay: float = 0.0


class ResNetFinetuner:
    def __init__(self, args: ResNetFinetuningArgs):
        self.args = args

    def pre_training_setup(self):
        self.model = get_resnet_for_feature_extraction(self.args.n_classes).to(device)
        self.optimizer = AdamW(
            self.model.out_layers[-1].parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        self.trainset, self.testset = get_cifar()
        self.train_loader = DataLoader(self.trainset, batch_size=self.args.batch_size, shuffle=True)
        self.test_loader = DataLoader(self.testset, batch_size=self.args.batch_size, shuffle=False)
        self.logged_variables = {"loss": [], "accuracy": []}
        self.examples_seen = 0

    def training_step(
        self,
        imgs: Float[Tensor, "batch channels height width"],
        labels: Int[Tensor, " batch"],
    ) -> Float[Tensor, ""]:
        """Perform a gradient update step on a single batch of data."""
        imgs, labels = imgs.to(device), labels.to(device)

        logits = self.model(imgs)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.examples_seen += imgs.shape[0]
        self.logged_variables["loss"].append(loss.item())
        return loss

    @t.inference_mode()
    def evaluate(self) -> float:
        """Evaluate the model on the test set and return the accuracy."""
        self.model.eval()
        total_correct, total_samples = 0, 0

        for imgs, labels in tqdm(self.test_loader, desc="Evaluating"):
            imgs, labels = imgs.to(device), labels.to(device)
            logits = self.model(imgs)
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += len(imgs)

        accuracy = total_correct / total_samples
        self.logged_variables["accuracy"].append(accuracy)
        return accuracy

    def train(self) -> dict[str, list[float]]:
        self.pre_training_setup()

        accuracy = self.evaluate()

        for epoch in range(self.args.epochs):
            self.model.train()

            pbar = tqdm(self.train_loader, desc="Training")
            for imgs, labels in pbar:
                loss = self.training_step(imgs, labels)
                pbar.set_postfix(loss=f"{loss:.3f}", ex_seen=f"{self.examples_seen:06}")

            accuracy = self.evaluate()
            pbar.set_postfix(loss=f"{loss:.3f}", accuracy=f"{accuracy:.2f}", ex_seen=f"{self.examples_seen:06}")

        return self.logged_variables

# %%
args = ResNetFinetuningArgs()
trainer = ResNetFinetuner(args)
logged_variables = trainer.train()


line(
    y=[logged_variables["loss"][: 391 * 3 + 1], logged_variables["accuracy"][:4]],
    x_max=len(logged_variables["loss"][: 391 * 3 + 1] * args.batch_size),
    yaxis2_range=[0, 1],
    use_secondary_yaxis=True,
    labels={"x": "Examples seen", "y1": "Cross entropy loss", "y2": "Test Accuracy"},
    title="Feature extraction with ResNet34",
    width=600,
)

# %% Same finetuning using wandb 
@dataclass
class WandbResNetFinetuningArgs(ResNetFinetuningArgs):
    """Contains new params for use in wandb.init, as well as all the ResNetFinetuningArgs params."""

    wandb_project: str | None = "day3-resnet"
    wandb_name: str | None = None


class WandbResNetFinetuner(ResNetFinetuner):
    args: WandbResNetFinetuningArgs  # adding this line helps with typechecker!
    examples_seen: int = 0  # tracking examples seen (used as step for wandb)

    def pre_training_setup(self):
        """Initializes the wandb run using `wandb.init` and `wandb.watch`."""
        super().pre_training_setup()
        wandb.init(project=self.args.wandb_project, name=self.args.wandb_name, config=self.args.__dict__)
        wandb.watch(self.model, log="all", log_freq=1000)

    def training_step(
        self,
        imgs: Float[Tensor, "batch channels height width"],
        labels: Int[Tensor, " batch"],
    ) -> Float[Tensor, ""]:
        """Equivalent to ResNetFinetuner.training_step, but logging the loss to wandb."""
        loss = super().training_step(imgs, labels)
        wandb.log({"train_loss": loss.item()}, step=self.examples_seen)
        return loss

    @t.inference_mode()
    def evaluate(self) -> float:
        """Equivalent to ResNetFinetuner.evaluate, but logging the accuracy to wandb."""
        accuracy = super().evaluate()
        wandb.log({"test_accuracy": accuracy}, step=self.examples_seen)
        return accuracy

    def train(self) -> None:
        """Equivalent to ResNetFinetuner.train, but with wandb integration."""
        self.pre_training_setup()
        accuracy = self.evaluate()

        for epoch in range(self.args.epochs):
            self.model.train()

            pbar = tqdm(self.train_loader, desc="Training")
            for imgs, labels in pbar:
                loss = self.training_step(imgs, labels)
                pbar.set_postfix(loss=f"{loss:.3f}", ex_seen=f"{self.examples_seen:06}")

            accuracy = self.evaluate()
            pbar.set_postfix(loss=f"{loss:.3f}", accuracy=f"{accuracy:.2f}", ex_seen=f"{self.examples_seen:06}")
        wandb.finish()


args = WandbResNetFinetuningArgs()
trainer = WandbResNetFinetuner(args)
trainer.train()

# %% Sweep config
# YOUR CODE HERE - fill `sweep_config` so it has the requested behaviour
sweep_config = dict(
    method = 'random',
    metric = {
        'name': 'accuracy',
        'goal': 'maximize'
    },
    parameters = {
        'learning_rate': {'min': 1e-4, 'max': 1e-1, 'distribution': 'log_uniform_values'},
        'weight_decay': {'values': [True, False]},
        'weight_decay_lambda': {'min': 1e-4, 'max': 1e-2, 'distribution': 'log_uniform_values'},
        'batch_size': {'values': [32, 64, 128, 256]},
    }
)


def update_args(args: WandbResNetFinetuningArgs, sampled_parameters: dict) -> WandbResNetFinetuningArgs:
    """
    Returns a new args object with modified values. The dictionary `sampled_parameters` will have
    the same keys as your `sweep_config["parameters"]` dict, and values equal to the sampled values
    of those hyperparameters.
    """
    assert set(sampled_parameters.keys()) == set(sweep_config["parameters"].keys())

    # YOUR CODE HERE - update `args` based on `sampled_parameters`
    args.learning_rate = sampled_parameters['learning_rate']
    args.weight_decay = sampled_parameters['weight_decay_lambda'] if sampled_parameters['weight_decay'] else 0.0
    args.batch_size = sampled_parameters['batch_size']  
    return args

tests.test_sweep_config(sweep_config)
tests.test_update_args(update_args, sweep_config)

# %%
def train():
    # Define args & initialize wandb
    args = WandbResNetFinetuningArgs()
    wandb.init(project=args.wandb_project, name=args.wandb_name, reinit=False)

    # After initializing wandb, we can update args using `wandb.config`
    args = update_args(args, dict(wandb.config))

    # Train the model with these new hyperparameters (the second `wandb.init` call will be ignored)
    trainer = WandbResNetFinetuner(args)
    trainer.train()


sweep_id = wandb.sweep(sweep=sweep_config, project="day3-resnet-sweep")
wandb.agent(sweep_id=sweep_id, function=train, count=3)
wandb.finish()















# %% ------------------------------------------------------
#               Distributed training
#    ------------------------------------------------------
WORLD_SIZE = min(t.cuda.device_count(), 3)

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12345"


def send_receive(rank, world_size):
    dist.init_process_group(backend="gloo", rank=rank, world_size=world_size)

    if rank == 0:
        # Send tensor to rank 1
        sending_tensor = t.zeros(1)
        print(f"{rank=}, sending {sending_tensor=}")
        dist.send(tensor=sending_tensor, dst=1)
    elif rank == 1:
        # Receive tensor from rank 0
        received_tensor = t.ones(1)
        print(f"{rank=}, creating {received_tensor=}")
        dist.recv(received_tensor, src=0)  # this line overwrites the tensor's data with our `sending_tensor`
        print(f"{rank=}, received {received_tensor=}")

    dist.destroy_process_group()


if MAIN:
    world_size = 2  # simulate 2 processes
    mp.spawn(
        send_receive,
        args=(world_size,),
        nprocs=world_size,
        join=True,
    )
# %% implementing broadcast (check spearate file in distributed_training)
def broadcast(tensor: Tensor, rank: int, world_size: int, src: int = 0):
    """
    Broadcast averaged gradients from rank 0 to all other ranks.
    """
    raise NotImplementedError()


if MAIN:
    tests.test_broadcast(broadcast, WORLD_SIZE)

# %% Ring allreduce

WORLD_SIZE = 1

os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = "12345"

def ring_all_reduce(tensor: Tensor, rank, world_size, op: Literal["sum", "mean"] = "sum") -> None:
    """
    Ring all_reduce implementation using non-blocking send/recv to avoid deadlock.
    """
    chunk_size = tensor.numel() // world_size
    left = (rank - 1) % world_size
    right = (rank + 1) % world_size 
    
    # Step 1: Reduce-scatter phase
    for i in range(world_size - 1):
        send_chunk = (rank - i) % world_size
        recv_chunk = (rank - i - 1) % world_size

        send_tensor = tensor[send_chunk * chunk_size : (send_chunk + 1) * chunk_size].clone()
        recv_tensor = t.zeros_like(send_tensor)

        send_req = dist.isend(tensor=send_tensor, dst=right)
        recv_req = dist.irecv(tensor=recv_tensor, src=left)

        send_req.wait()
        recv_req.wait()

        tensor[recv_chunk * chunk_size : (recv_chunk + 1) * chunk_size] += recv_tensor
    
    # Step 2: Allgather phase
    for i in range(world_size - 1):
        send_chunk = (rank - i) % world_size
        recv_chunk = (rank - i - 1) % world_size

        send_tensor = tensor[send_chunk * chunk_size : (send_chunk + 1) * chunk_size].clone()
        recv_tensor = t.zeros_like(send_tensor)

        send_req = dist.isend(tensor=send_tensor, dst=right)
        recv_req = dist.irecv(tensor=recv_tensor, src=left)

        send_req.wait()
        recv_req.wait()

        tensor[recv_chunk * chunk_size : (recv_chunk + 1) * chunk_size] = recv_tensor

    if op == "mean":
        tensor /= world_size
        


if MAIN:
    tests.test_all_reduce(ring_all_reduce)

# %%
