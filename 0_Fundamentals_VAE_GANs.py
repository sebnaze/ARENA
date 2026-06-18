# %% 
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import einops
import torch as t
import torchinfo
import wandb
from datasets import load_dataset
from einops.layers.torch import Rearrange
from jaxtyping import Float
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

# Make sure exercises are in the path
chapter = "chapter0_fundamentals"
section = "part5_vaes_and_gans"
#root_dir = next(p for p in Path.cwd().parents if (p / chapter).exists())
root_dir = Path("/Users/sebastin/Documents/perso/ARENA_training/ARENA_3.0")
exercises_dir = root_dir / chapter / "exercises"
section_dir = exercises_dir / section
if str(exercises_dir) not in sys.path:
    sys.path.append(str(exercises_dir))

MAIN = __name__ == "__main__"

import part5_vaes_and_gans.tests as tests
import part5_vaes_and_gans.utils as utils
from plotly_utils import imshow

device = t.device("mps" if t.backends.mps.is_available() else "cuda" if t.cuda.is_available() else "cpu")

# %% download the datasets
celeb_data_dir = section_dir / "data/celeba"
celeb_image_dir = celeb_data_dir / "img_align_celeba"

os.makedirs(celeb_image_dir, exist_ok=True)

if len(list(celeb_image_dir.glob("*.jpg"))) > 0:
    print("Dataset already loaded.")
else:
    dataset = load_dataset("nielsr/CelebA-faces")
    print("Dataset loaded.")

    for idx, item in tqdm(enumerate(dataset["train"]), total=len(dataset["train"]), desc="Saving imgs...", ascii=True):
        # The image is already a JpegImageFile, so we can directly save it
        item["image"].save(celeb_image_dir / f"{idx:06}.jpg")

    print("All images have been saved.")

# %% Define a function to load the datasets
def get_dataset(dataset: Literal["MNIST", "CELEB"], train: bool = True) -> Dataset:
    assert dataset in ["MNIST", "CELEB"]

    if dataset == "CELEB":
        image_size = 64
        assert train, "CelebA dataset only has a training set"
        transform = transforms.Compose(
            [
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
        trainset = datasets.ImageFolder(root=exercises_dir / "part5_vaes_and_gans/data/celeba", transform=transform)

    elif dataset == "MNIST":
        img_size = 28
        transform = transforms.Compose(
            [
                transforms.Resize(img_size),
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )
        trainset = datasets.MNIST(
            root=exercises_dir / "part5_vaes_and_gans/data",
            transform=transform,
            download=True,
            train=train,
        )

    return trainset

# %% Ensure datasets are correctly loaded
def display_data(x: Tensor, nrows: int, title: str):
    """Displays a batch of data, using plotly."""
    ncols = x.shape[0] // nrows
    # Reshape into the right shape for plotting (make it 2D if image is monochrome)
    y = einops.rearrange(x, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=nrows).squeeze()
    # Normalize in the 0-1 range, then map to integer type
    y = (y - y.min()) / (y.max() - y.min())
    y = (y * 255).to(dtype=t.uint8)
    # Display data
    imshow(
        y,
        binary_string=(y.ndim == 2),
        height=50 * (nrows + 4),
        width=50 * (ncols + 5),
        title=f"{title}<br>single input shape = {x[0].shape}",
    )


trainset_mnist = get_dataset("MNIST")
trainset_celeb = get_dataset("CELEB")

# Display MNIST
x = next(iter(DataLoader(trainset_mnist, batch_size=25)))[0]
display_data(x, nrows=5, title="MNIST data")

# Display CelebA
x = next(iter(DataLoader(trainset_celeb, batch_size=25)))[0]
display_data(x, nrows=5, title="CelebA data")

# %% hold out data
testset = get_dataset("MNIST", train=False)
HOLDOUT_DATA = dict()
for data, target in DataLoader(testset, batch_size=1):
    if target.item() not in HOLDOUT_DATA:
        HOLDOUT_DATA[target.item()] = data.squeeze()
        if len(HOLDOUT_DATA) == 10:
            break
HOLDOUT_DATA = t.stack([HOLDOUT_DATA[i] for i in range(10)]).to(dtype=t.float, device=device).unsqueeze(1)

display_data(HOLDOUT_DATA, nrows=1, title="MNIST holdout data")

# %% building autoencoder
from part2_cnns.solutions import BatchNorm2d, Conv2d, Linear, ReLU, Sequential
from part5_vaes_and_gans.solutions import ConvTranspose2d
# %%

class Autoencoder(nn.Module):
    def __init__(self, latent_dim_size: int, hidden_dim_size: int):
        """Creates the encoder & decoder modules."""
        super().__init__()
        self.encoder = Sequential(
            Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=2, padding=1),
            ReLU(),
            Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.Flatten(start_dim=1, end_dim=-1),
            Linear(in_features=32*7*7, out_features=hidden_dim_size, bias=True),
            ReLU(),
            Linear(in_features=hidden_dim_size, out_features=latent_dim_size, bias=True)
        )
        self.decoder = Sequential(
            Linear(in_features=latent_dim_size, out_features=hidden_dim_size, bias=True),
            ReLU(),
            Linear(in_features=hidden_dim_size, out_features=32*7*7, bias=True),
            ReLU(),
            Rearrange('b (c h w) -> b c h w', c=32, h=7, w=7),
            ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1),
            ReLU(),
            ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=4, stride=2, padding=1)
        )
        self.latent_dim_size = latent_dim_size
        self.hidden_dim_size = hidden_dim_size

    def forward(self, x: Tensor) -> Tensor:
        """Returns the reconstruction of the input, after mapping through encoder & decoder."""
        latent = self.encoder(x)
        return self.decoder(latent)


tests.test_autoencoder(Autoencoder)


# %%
@dataclass
class AutoencoderArgs:
    # architecture
    latent_dim_size: int = 5
    hidden_dim_size: int = 128

    # data / training
    dataset: Literal["MNIST", "CELEB"] = "MNIST"
    batch_size: int = 512
    epochs: int = 10
    lr: float = 1e-3
    betas: tuple[float, float] = (0.5, 0.999)

    # logging
    use_wandb: bool = True
    wandb_project: str | None = "day5-autoencoder"
    wandb_name: str | None = None
    log_every_n_steps: int = 250


class AutoencoderTrainer:
    def __init__(self, args: AutoencoderArgs):
        self.args = args
        self.trainset = get_dataset(args.dataset)
        self.trainloader = DataLoader(self.trainset, batch_size=args.batch_size, shuffle=True)
        self.model = Autoencoder(
            latent_dim_size=args.latent_dim_size,
            hidden_dim_size=args.hidden_dim_size,
        ).to(device)
        self.optimizer = t.optim.Adam(self.model.parameters(), lr=args.lr, betas=args.betas)


    def training_step(self, img: Tensor) -> Tensor:
        """
        Performs a training step on the batch of images in `img`. Returns the loss. Logs to wandb
        if enabled.
        """
        img = img.to(device)
        recon_img = self.model(img)
        loss = t.nn.functional.mse_loss(img, recon_img)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.step += 1
        if self.args.use_wandb:
            if self.step % self.args.log_every_n_steps == 0:
                wandb.log({"train_loss": loss.item()}, step=self.step)
        return loss

    @t.inference_mode()
    def log_samples(self) -> None:
        """
        Evaluates model on holdout data, either logging to weights & biases or displaying output.
        """
        assert self.step > 0, "First call should come after a training step. Remember to increment `self.step`."
        output = self.model(HOLDOUT_DATA)
        if self.args.use_wandb:
            output = (output - output.min()) / (output.max() - output.min())  # Normalize to [0, 1]
            output = (output * 255).to(dtype=t.uint8)  # Convert to uint8 for logging
            wandb.log({"images": [wandb.Image(arr) for arr in output.cpu().numpy()]}, step=self.step)
        else:
            display_data(t.concat([HOLDOUT_DATA, output]), nrows=2, title="AE reconstructions")

    def train(self) -> Autoencoder:
        """Performs a full training run."""
        self.step = 0
        if self.args.use_wandb:
            wandb.init(project=self.args.wandb_project, name=self.args.wandb_name)
            wandb.watch(self.model)

        # YOUR CODE HERE - iterate over epochs, and train your model
        for epoch in range(self.args.epochs):
            pbar = tqdm(self.trainloader, desc=f"Epoch {epoch}/{self.args.epochs} - Training")
            mean_loss = 0
            for i, (img, label) in enumerate(pbar):
                loss = self.training_step(img)
                mean_loss += loss
                pbar.set_postfix(mean_loss=f"{mean_loss/(i+1):.3f}", n_img_seen=f"{self.step*self.args.batch_size}")

            self.log_samples()

        if self.args.use_wandb:
            wandb.finish()

        return self.model


args = AutoencoderArgs(use_wandb=True)
trainer = AutoencoderTrainer(args)
autoencoder = trainer.train()

# %% latent space of autoencoder
def create_grid_of_latents(
    model, interpolation_range=(-1, 1), n_points=11, dims=(0, 1)
) -> Float[Tensor, "rows_x_cols latent_dims"]:
    """Create a tensor of zeros which varies along the 2 specified dimensions of the latent space."""
    grid_latent = t.zeros(n_points, n_points, model.latent_dim_size, device=device)
    x = t.linspace(*interpolation_range, n_points)
    grid_latent[..., dims[0]] = x.unsqueeze(-1)  # rows vary over dim=0
    grid_latent[..., dims[1]] = x  # cols vary over dim=1
    return grid_latent.flatten(0, 1)  # flatten over (rows, cols) into a single batch dimension


grid_latent = create_grid_of_latents(autoencoder, interpolation_range=(-3, 3))

# Map grid latent through the decoder
output = autoencoder.decoder(grid_latent)

# Visualize the output
utils.visualise_output(output, grid_latent, title="Autoencoder latent space visualization")

# %%
# Get a small dataset with 5000 points
small_dataset = Subset(get_dataset("MNIST"), indices=range(0, 5000))
imgs = t.stack([img for img, label in small_dataset]).to(device)
labels = t.tensor([label for img, label in small_dataset]).to(device).int()

# Get the latent vectors for this data along first 2 dims, plus for the holdout data
latent_vectors = autoencoder.encoder(imgs)[:, :2]
holdout_latent_vectors = autoencoder.encoder(HOLDOUT_DATA)[:, :2]

# Plot the results
utils.visualise_input(latent_vectors.to('cpu'), labels.to('cpu'), holdout_latent_vectors.to('cpu'), HOLDOUT_DATA.to('cpu'))

# %% VAE
class VAE(nn.Module):
    encoder: nn.Module
    decoder: nn.Module

    def __init__(self, latent_dim_size: int, hidden_dim_size: int):
        super().__init__()
        self.encoder = Sequential(
            Conv2d(in_channels=1, out_channels=16, kernel_size=4, stride=2, padding=1),
            ReLU(),
            Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.Flatten(start_dim=1, end_dim=-1),
            Linear(in_features=32*7*7, out_features=hidden_dim_size, bias=True),
            ReLU(),
            Linear(in_features=hidden_dim_size, out_features=2*latent_dim_size, bias=True),
            Rearrange('b (n latent_dim) -> n b latent_dim', n=2) 
        )
        self.decoder = Sequential(
            Linear(in_features=latent_dim_size, out_features=hidden_dim_size, bias=True),
            ReLU(),
            Linear(in_features=hidden_dim_size, out_features=32*7*7, bias=True),
            ReLU(),
            Rearrange('b (c h w) -> b c h w', c=32, h=7, w=7),
            ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1),
            ReLU(),
            ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=4, stride=2, padding=1)
        )
        self.latent_dim_size = latent_dim_size
        self.hidden_dim_size = hidden_dim_size

    def sample_latent_vector(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Passes `x` through the encoder, returns tuple of (sampled latent vector, mean, log std dev).
        This function can be used in `forward`, but also used on its own to generate samples for
        evaluation.
        """
        mu, logsigma = self.encoder(x) # shape (2, b, latent_dim_size)
        eps = t.randn_like(logsigma)
        latent = mu + t.exp(logsigma) * eps
        return latent, mu, logsigma

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        """
        Passes `x` through the encoder and decoder. Returns the reconstructed input, as well as mu
        and logsigma.
        """
        latent, mu, logsigma = self.sample_latent_vector(x)
        return self.decoder(latent), mu, logsigma
    



tests.test_vae(VAE)

# %% Training the VAE
@dataclass
class VAEArgs(AutoencoderArgs):
    wandb_project: str | None = "day5-vae-mnist"
    beta_kl: float = 0.1


class VAETrainer:
    def __init__(self, args: VAEArgs):
        self.args = args
        self.trainset = get_dataset(args.dataset)
        self.trainloader = DataLoader(self.trainset, batch_size=args.batch_size, shuffle=True)#, num_workers=8)
        self.model = VAE(
            latent_dim_size=args.latent_dim_size,
            hidden_dim_size=args.hidden_dim_size,
        ).to(device)
        self.optimizer = t.optim.Adam(self.model.parameters(), lr=args.lr, betas=args.betas)

    def training_step(self, img: Tensor):
        """
        Performs a training step on the batch of images in `img`. Returns the loss. Logs to wandb
        if enabled.
        """
        img = img.to(device)
        recon_img, mu, logsigma = self.model(img)
        D_KL = 0.5*(mu**2 + t.exp(2*logsigma) - 1) - logsigma
        loss = t.nn.functional.mse_loss(img, recon_img) + self.args.beta_kl * D_KL.mean() 
        
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.step += 1
        if self.args.use_wandb:
            if self.step % self.args.log_every_n_steps == 0:
                wandb.log({"train_loss": loss.item()}, step=self.step)
        return loss

    @t.inference_mode()
    def log_samples(self) -> None:
        """
        Evaluates model on holdout data, either logging to wandb or displaying output inline.
        """
        assert self.step > 0, "First call should come after a training step. Remember to increment `self.step`."
        output = self.model(HOLDOUT_DATA)[0]
        if self.args.use_wandb:
            output = (output - output.min()) / (output.max() - output.min())  # Normalize to [0, 1]
            output = (output * 255).to(dtype=t.uint8)  # Convert to uint8 for logging
            wandb.log({"images": [wandb.Image(arr) for arr in output.cpu().numpy()]}, step=self.step)
        else:
            display_data(t.concat([HOLDOUT_DATA, output]), nrows=2, title="VAE reconstructions")

    def train(self) -> VAE:
        """Performs a full training run."""
        self.step = 0
        if self.args.use_wandb:
            wandb.init(project=self.args.wandb_project, name=self.args.wandb_name)
            wandb.watch(self.model)

        # YOUR CODE HERE - iterate over epochs, and train your model
        for epoch in range(self.args.epochs):
            pbar = tqdm(self.trainloader, desc=f"Epoch {epoch}/{self.args.epochs} - Training")
            mean_loss = 0
            for i, (img, label) in enumerate(pbar):
                loss = self.training_step(img)
                mean_loss += loss
                pbar.set_postfix(mean_loss=f"{mean_loss/(i+1):.3f}", n_img_seen=f"{self.step*self.args.batch_size}")

            self.log_samples()
            
        if self.args.use_wandb:
            wandb.finish()

        return self.model


args = VAEArgs(latent_dim_size=5, hidden_dim_size=100, use_wandb=False)
trainer = VAETrainer(args)
vae = trainer.train()

# %%
class Tanh(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return t.tanh(x)


class LeakyReLU(nn.Module):
    def __init__(self, negative_slope: float = 0.01):
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x: Tensor) -> Tensor:
        return t.where(x >= 0, x, self.negative_slope * x)

    def extra_repr(self) -> str:
        return f"negative_slope={self.negative_slope}"


class Sigmoid(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return 1 / (1 + t.exp(-x))


tests.test_Tanh(Tanh)
tests.test_LeakyReLU(LeakyReLU)
tests.test_Sigmoid(Sigmoid)

# %% --------------------     GAN   -----------------------------

class Generator(nn.Module):
    def __init__(
        self,
        latent_dim_size: int = 100,
        img_size: int = 64,
        img_channels: int = 3,
        hidden_channels: list[int] = [128, 256, 512],
    ):
        """
        Implements the generator architecture from the DCGAN paper (the diagram at the top
        of page 4). We assume the size of the activations doubles at each layer (so image
        size has to be divisible by 2 ** len(hidden_channels)).

        Args:
            latent_dim_size:
                the size of the latent dimension, i.e. the input to the generator
            img_size:
                the size of the image, i.e. the output of the generator
            img_channels:
                the number of channels in the image (3 for RGB, 1 for grayscale)
            hidden_channels:
                the number of channels in the hidden layers of the generator (starting closest
                to the middle of the DCGAN and going outward, i.e. in chronological order for
                the generator)
        """
        n_layers = len(hidden_channels)
        assert img_size % (2**n_layers) == 0, "activation size must double at each layer"

        super().__init__()

        self.project_and_reshape = Sequential(
            Linear(in_features=latent_dim_size, out_features=hidden_channels[-1]*(img_size//(2**n_layers))**2, bias=False),
            Rearrange('b (c h w) -> b c h w', c=hidden_channels[-1], h=img_size//(2**n_layers), w=img_size//(2**n_layers)),
            BatchNorm2d(num_features=hidden_channels[-1]),
        )
        self.hidden_layers = Sequential(
            *[nn.Sequential(
                ConvTranspose2d(
                    in_channels=hidden_channels[-i-1],
                    out_channels=hidden_channels[-i-2],
                    kernel_size=4,
                    stride=2,
                    padding=1
                ),
                BatchNorm2d(num_features=hidden_channels[-i-2]),
                ReLU()
            ) for i in range(n_layers-1)],
            ConvTranspose2d(
                in_channels=hidden_channels[0],
                out_channels=img_channels,
                kernel_size=4,
                stride=2,
                padding=1
            ),
            Tanh()
        )   

    def forward(self, x: Tensor) -> Tensor:
        x = self.project_and_reshape(x)
        x = self.hidden_layers(x)
        return x


class Discriminator(nn.Module):
    def __init__(
        self,
        img_size: int = 64,
        img_channels: int = 3,
        hidden_channels: list[int] = [128, 256, 512],
    ):
        """
        Implements the discriminator architecture from the DCGAN paper (the mirror image of
        the diagram at the top of page 4). We assume the size of the activations doubles at
        each layer (so image size has to be divisible by 2 ** len(hidden_channels)).

        Args:
            img_size:
                the size of the image, i.e. the input of the discriminator
            img_channels:
                the number of channels in the image (3 for RGB, 1 for grayscale)
            hidden_channels:
                the number of channels in the hidden layers of the discriminator (starting
                closest to the middle of the DCGAN and going outward, i.e. in reverse-
                chronological order for the discriminator)
        """
        n_layers = len(hidden_channels)
        assert img_size % (2**n_layers) == 0, "activation size must double at each layer"

        super().__init__()

        self.hidden_layers = Sequential(
            *[Sequential(
                Conv2d(
                    in_channels=img_channels,
                    out_channels=hidden_channels[0],
                    kernel_size=4,
                    stride=2,
                    padding=1
                ),
                LeakyReLU(negative_slope=0.2),
            ),
            *[nn.Sequential(
                Conv2d(
                    in_channels=hidden_channels[i],
                    out_channels=hidden_channels[i+1],
                    kernel_size=4,
                    stride=2,
                    padding=1
                ),
                BatchNorm2d(num_features=hidden_channels[i+1]),
                LeakyReLU(negative_slope=0.2)
            ) for i in range(n_layers-1)]
            ]
        )   
        
        self.classifier = Sequential(
            nn.Flatten(start_dim=1, end_dim=-1),
            Linear(in_features=hidden_channels[-1]*(img_size//(2**n_layers))**2, out_features=1, bias=False),
            Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.hidden_layers(x)
        x = self.classifier(x)
        return x.squeeze()  # remove dummy `out_channels` dimension


class DCGAN(nn.Module):
    netD: Discriminator
    netG: Generator

    def __init__(
        self,
        latent_dim_size: int = 100,
        img_size: int = 64,
        img_channels: int = 3,
        hidden_channels: list[int] = [128, 256, 512],
    ):
        super().__init__()
        self.latent_dim_size = latent_dim_size
        self.img_size = img_size
        self.img_channels = img_channels
        self.hidden_channels = hidden_channels
        self.netD = Discriminator(img_size, img_channels, hidden_channels)
        self.netG = Generator(latent_dim_size, img_size, img_channels, hidden_channels)


# %% Check that the number of parameters in your implementation matches the reference implementation
from part2_cnns.utils import print_param_count
from part5_vaes_and_gans import solutions

print_param_count(Generator(), solutions.DCGAN().netG)
print_param_count(Discriminator(), solutions.DCGAN().netD)

# %% Check that the shapes of the activations in your implementation match the reference implementation
model = DCGAN().to(device)
x = t.randn(3, 100).to(device)
print(torchinfo.summary(model.netG, input_data=x), end="\n\n")
print(torchinfo.summary(model.netD, input_data=model.netG(x)))

















# %% Bonus : Transposed convolutions
from part2_cnns.solutions import (
    IntOrPair,
    conv1d_minimal,
    conv2d_minimal,
    force_pair,
    pad1d,
    pad2d,
)


def conv_transpose1d_minimal(
    x: Float[Tensor, "batch in_channels width"],
    weights: Float[Tensor, "in_channels out_channels kernel_width"],
) -> Float[Tensor, "batch out_channels output_width"]:
    """Like torch's conv_transpose1d using bias=False and all other keyword arguments left at their default values."""
    out_width = x.shape[-1] + weights.shape[-1] - 1
    n_pad = weights.shape[-1] - 1
    x_padded = pad1d(x, left=n_pad, right=n_pad, pad_value=0)
    weights = einops.rearrange(weights, "in_channels out_channels kernel_width -> out_channels in_channels kernel_width")
    return conv1d_minimal(x_padded, weights.flip(-1))


tests.test_conv_transpose1d_minimal(conv_transpose1d_minimal)


# %% fractional stride (a.k.a. "spaced out") convolutions
def fractional_stride_1d(
    x: Float[Tensor, "batch in_channels width"], stride: int = 1
) -> Float[Tensor, "batch in_channels output_width"]:
    """
    Returns a version of x suitable for transposed convolutions, i.e. "spaced out" with zeros
    between its values. This spacing only happens along the last dimension.

    x: shape (batch, in_channels, width)

    Example:
        x = [[[1, 2, 3], [4, 5, 6]]]
        stride = 2
        output = [[[1, 0, 2, 0, 3], [4, 0, 5, 0, 6]]]
    """
    if stride == 1:
        return x
    batch, in_channels, width = x.shape
    out_width = width + (width - 1) * (stride - 1)
    output = t.zeros((batch, in_channels, out_width), dtype=x.dtype)
    output[..., ::stride] = x
    return output


tests.test_fractional_stride_1d(fractional_stride_1d)

# %% Putting it all together: transposed convolution using fractional stride and conv1d
def conv_transpose1d(
    x: Float[Tensor, "batch in_channels width"],
    weights: Float[Tensor, "in_channels out_channels kernel_width"],
    stride: int = 1,
    padding: int = 0,
) -> Float[Tensor, "batch out_channels output_width"]:
    """
    Like torch's conv_transpose1d using bias=False and all other keyword arguments left at their
    default values.
    """
    x_strided = fractional_stride_1d(x, stride=stride)
    x_padded = pad1d(x_strided, left=weights.shape[-1] - 1 - padding, right=weights.shape[-1] - 1 - padding, pad_value=0)
    weights = einops.rearrange(weights, "in_channels out_channels kernel_width -> out_channels in_channels kernel_width")   
    return conv1d_minimal(x_padded, weights.flip(-1))


tests.test_conv_transpose1d(conv_transpose1d)

# %% Now extend the fractional stride and transposed convolution to 2D, applying the same logic along both the height and width dimensions.
def fractional_stride_2d(
    x: Float[Tensor, "batch in_channels height width"], stride_h: int, stride_w: int
) -> Float[Tensor, "batch in_channels output_height output_width"]:
    """
    Same as fractional_stride_1d, except we apply it along the last 2 dims of x (height and width).
    """
    if stride_h == stride_w == 1:
        return x
    batch, in_channels, height, width = x.shape
    out_height = height + (height - 1) * (stride_h - 1)
    out_width = width + (width - 1) * (stride_w - 1)
    output = t.zeros((batch, in_channels, out_height, out_width), dtype=x.dtype)
    output[..., ::stride_h, ::stride_w] = x
    return output


def conv_transpose2d(x, weights, stride: IntOrPair = 1, padding: IntOrPair = 0) -> Tensor:
    """Like torch's conv_transpose2d using bias=False
    x: shape (batch, in_channels, height, width)
    weights: shape (out_channels, in_channels, kernel_height, kernel_width)
    Returns: shape (batch, out_channels, output_height, output_width)
    """
    stride_h, stride_w = force_pair(stride)
    padding_h, padding_w = force_pair(padding)

    x_strided = fractional_stride_2d(x, stride_h=stride_h, stride_w=stride_w)
    x_padded = pad2d(x_strided, left=weights.shape[-1] - 1 - padding_w, right=weights.shape[-1] - 1 - padding_w, 
                     top=weights.shape[-2] - 1 - padding_h, bottom=weights.shape[-2] - 1 - padding_h,
                     pad_value=0)
    weights = einops.rearrange(weights, "i o h w -> o i h w")   
    return conv2d_minimal(x_padded, weights.flip(-2).flip(-1))


tests.test_fractional_stride_2d(fractional_stride_2d)
tests.test_conv_transpose2d(conv_transpose2d)

# %% 
class ConvTranspose2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: IntOrPair,
        stride: IntOrPair = 1,
        padding: IntOrPair = 0,
    ):
        """
        Same as torch.nn.ConvTranspose2d with bias=False.
        Name your weight field `self.weight` for compatibility with the tests.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = force_pair(kernel_size)
        self.stride = stride
        self.padding = padding

        sqrt_k = (1 / (out_channels * self.kernel_size[0] * self.kernel_size[1]))**0.5
        self.weight = nn.Parameter(t.rand(in_channels, out_channels, *self.kernel_size) * 2 * sqrt_k - sqrt_k) 


    def forward(
        self, x: Float[Tensor, "batch in_channels height width"]
    ) -> Float[Tensor, "batch out_channels output_height output_width"]:
        return conv_transpose2d(x, self.weight, stride=self.stride, padding=self.padding)

    def extra_repr(self) -> str:
        keys = ["in_channels", "out_channels", "kernel_size", "stride", "padding"]
        return ", ".join([f"{key}={getattr(self, key)}" for key in keys])


tests.test_ConvTranspose2d(ConvTranspose2d)

# %%
