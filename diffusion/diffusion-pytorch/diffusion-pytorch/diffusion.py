"""
Implementation of diffusion model in pytorch to generate MNIST images.

Based on https://huggingface.co/blog/annotated-diffusion,
which implements the DDPM paper https://arxiv.org/abs/2006.11239.
"""

import os
import functools
import pathlib
import inspect
import numpy as np
import einops
from einops.layers.torch import Rearrange
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm.auto import tqdm

import torch
import torchvision

# Set device
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(f"Using {device} device")
print()
results_folder = pathlib.Path("./results")
results_folder.mkdir(exist_ok = True)
#---------------------------------------------
# Download the MNIST dataset using torchvision
# 60k+10k 28x28 black-and-white images of handwritten digits
#---------------------------------------------
print('------------------ Dataset ------------------')

# Download the pre-existing Dataset and convert to tensors
print('Loading MNIST digit dataset...')
train_dataset = torchvision.datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True,
                                           transform=torchvision.transforms.ToTensor())

# We want to create a dataset of:
#   - (x_t, t, label) for each t in [0,50]
#   - For each training instance, we will:
#       - Sample a random t
#       - Sample a noise term epsilon from a gaussian
#       - Diffuse the image to the t'th step
#       - Train our model to learn the noise mapping the diffused image to the original image, for the given t

# Construct a dataloader
batch_size = 100
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Get an example image
imgs, labels = next(iter(train_dataloader))
img0 = imgs[0][0]
label0 = labels[0]
image_shape = img0.numpy().shape
image_dim = image_shape[0]
print(f'  Total samples: {len(train_dataset)} (train)')
print(f'  Shape of images: {image_shape} (dim={image_dim}x{image_dim})')
print('Done.')
print('--------------------------------------------')

#---------------------------------------------
# Define the diffusion pipeline.
#
# Starting with a number of time steps [0,...,T] and a noise schedule beta_0<...<beta_T:
#
# We first compute the forward diffusion process using a reparameterization of the beta's:
#   alpha_t=1-beta_t, \bar{alpha}=\prod_s=1^t{alpha_s}.
# This allows us to directly diffuse to a time step t:
#   q(x_t|x_0) = N(sqrt(\bar{alpha}_t) x_0, 1-\bar{alpha}_t I), which is equivalent to
#   x_t = sqrt(1-beta_t) x_{t-1} + sqrt(beta_t) epsilon, where epsilon ~N(0,I).
#
# We then want to compute the reverse diffusion process, p(x_{t-1} | x_t).
# This is analytically intractable (the image space is too large), so we use a learned model to approximate it.
# We assume that p is Gaussian, and assume a fixed variance sigma_t^2=beta_t for each t. 
# We then reparameterize the mean and can instead learn a noise eps_theta:
#   Loss = MSE[eps, eps_theta(x_t,t)], where eps is the generated noise from the forward diffusion process
#                                      and eps_theta is the learned function.      
# 
# That is, the algorithm is:
#  1. Take a noiseless image x_0
#  2. Sample a time step t
#  3. Sample a noise term epsilon~N(0,I), and forward diffuse
#  4. Perform gradient descent to learn the noise: optimize MSE[eps, eps_theta(x_t,t)]
#
# The NN takes a noisy image as input, and outputs an image of the noise.
# 
# We can then construct noise samples, and use the NN to denoise them and produce target images. 
#---------------------------------------------

# Define beta schedule, and define related parameters: alpha, alpha_bar
T = 300
beta = torch.linspace(0.0001, 0.02, T)
alpha = 1. - beta
alphabar = torch.cumprod(alpha, axis=0)
alphabar_prev = torch.nn.functional.pad(alphabar[:-1], (1, 0), value=1.0)
sqrt_1_alpha = torch.sqrt(1.0 / alpha)

# Quantities needed for diffusion q(x_t | x_{t-1})
sqrt_alphabar = torch.sqrt(alphabar)
sqrt_one_minus_alphabar = torch.sqrt(1. - alphabar)

# Quantities needed for inversion q(x_{t-1} | x_t, x_0)
posterior_variance = beta * (1. - alphabar_prev) / (1. - alphabar)

# Define function allowing us to extract t index for a batch
def extract(a, t, x_shape):
    '''
    a: tensor of shape (T,)
    t: time step t
    x_shape: shape of x_0
    '''
    batch_size = t.shape[0]
    out = a.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

# Forward diffusion
def q(x_0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_0)

    sqrt_alphabar_t = extract(sqrt_alphabar, t, x_0.shape)
    sqrt_one_minus_alphabar_t = extract(sqrt_one_minus_alphabar, t, x_0.shape)

    return sqrt_alphabar_t * x_0 + sqrt_one_minus_alphabar_t * noise

# Forward diffusion example
t = torch.tensor([T-1])
difuze = q(img0, t)
plt.imshow(img0.numpy(), cmap='gray', vmin=0, vmax=1)
plt.savefig(str(results_folder / f'img0.png'))
plt.imshow(difuze.numpy(), cmap='gray', vmin=0, vmax=1)
plt.savefig(str(results_folder / f'img0_diffused.png'))

#---------------------------------------------
# Define the model: https://huggingface.co/blog/annotated-diffusion
#---------------------------------------------
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if inspect.isfunction(d) else d

def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


class Residual(torch.nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return torch.nn.Sequential(
        torch.nn.Upsample(scale_factor=2, mode="nearest"),
        torch.nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    # No More Strided Convolutions or Pooling
    return torch.nn.Sequential(
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        torch.nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )

class SinusoidalPositionEmbeddings(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class WeightStandardizedConv2d(torch.nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = einops.reduce(weight, "o ... -> o 1 1 1", "mean")
        var = einops.reduce(weight, "o ... -> o 1 1 1", functools.partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return torch.nn.functional.conv2d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

class Block(torch.nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = torch.nn.GroupNorm(groups, dim_out)
        self.act = torch.nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(torch.nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            torch.nn.Sequential(torch.nn.SiLU(), torch.nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = torch.nn.Conv2d(dim, dim_out, 1) if dim != dim_out else torch.nn.Identity()

    def forward(self, x, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = einops.rearrange(time_emb, "b c -> b c 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + self.res_conv(x)

class Attention(torch.nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = torch.nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = torch.nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: einops.rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )
        q = q * self.scale

        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = torch.einsum("b h i j, b h d j -> b h i d", attn, v)
        out = einops.rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)

class LinearAttention(torch.nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = torch.nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = torch.nn.Sequential(torch.nn.Conv2d(hidden_dim, dim, 1), 
                                    torch.nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: einops.rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        )

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = einops.rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)

class PreNorm(torch.nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = torch.nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)
    
class Unet(torch.nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        self_condition=False,
        resnet_block_groups=4,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels
        self.self_condition = self_condition
        input_channels = channels * (2 if self_condition else 1)

        init_dim = default(init_dim, dim)
        self.init_conv = torch.nn.Conv2d(input_channels, init_dim, 1, padding=0) # changed to 1 and 0 from 7,3

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = functools.partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings
        time_dim = dim * 4

        self.time_mlp = torch.nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            torch.nn.Linear(dim, time_dim),
            torch.nn.GELU(),
            torch.nn.Linear(time_dim, time_dim),
        )

        # layers
        self.downs = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                torch.nn.ModuleList(
                    [
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else torch.nn.Conv2d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                torch.nn.ModuleList(
                    [
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else torch.nn.Conv2d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.out_dim = default(out_dim, channels)

        self.final_res_block = block_klass(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = torch.nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, x_self_cond=None):
        if self.self_condition:
            x_self_cond = default(x_self_cond, lambda: torch.zeros_like(x))
            x = torch.cat((x_self_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t)
            h.append(x)

            x = block2(x, t)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t)
        return self.final_conv(x)

#---------------------------------------------
# Sampling
# With a trained model, we can now subtract the noise
#---------------------------------------------
@torch.no_grad()
def p_sample(model, x, t, t_index):
    betas_t = extract(beta, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = extract(sqrt_one_minus_alphabar, t, x.shape)
    sqrt_recip_alphas_t = extract(sqrt_1_alpha, t, x.shape)
    
    # Equation 11 in the paper
    # Use our model (noise predictor) to predict the mean
    model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)

    if t_index == 0:
        return model_mean
    else:
        posterior_variance_t = extract(posterior_variance, t, x.shape)
        noise = torch.randn_like(x)
        # Algorithm 2 line 4:
        return model_mean + torch.sqrt(posterior_variance_t) * noise 

# Algorithm 2 (including returning all images)
@torch.no_grad()
def p_sample_loop(model, shape):
    device = next(model.parameters()).device

    b = shape[0]
    # start from pure noise (for each example in the batch)
    img = torch.randn(shape, device=device)
    imgs = []

    for i in tqdm(reversed(range(0, T)), desc='sampling loop time step', total=T):
        img = p_sample(model, img, torch.full((b,), i, device=device, dtype=torch.long), i)
        imgs.append(img.cpu().numpy())
    return imgs

@torch.no_grad()
def sample(model, image_size, batch_size=16, channels=1):
    return p_sample_loop(model, shape=(batch_size, channels, image_size, image_size))

#---------------------------------------------
# Training the denoising model
#---------------------------------------------
print()
print('------------------- Model -------------------')

# Create an instance of the network
model = Unet(
    dim=image_dim,
    channels=1,
    dim_mults=(1, 2, 4,)
)
model.to(device)

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {total_trainable_params}")
count_parameters(model)

print()
print('------------------ Training ------------------')
# Hyperparameters
learning_rate = 1e-3
n_epochs = 10

# Defining the loss function
def p_losses(denoise_model, x_0, t, noise=None):
    if noise is None:
        noise = torch.randn_like(x_0)

    x_noisy = q(x_0=x_0, t=t, noise=noise)
    predicted_noise = denoise_model(x_noisy, t)

    loss = torch.nn.functional.mse_loss(noise, predicted_noise)

    return loss

# Define optimizer
loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
training_loss = []

model_outputfile = str(results_folder / 'model.pkl')
if os.path.exists(model_outputfile):
    model.load_state_dict(torch.load(model_outputfile))
    print(f"Loaded trained model from: {model_outputfile} (delete and re-run if you'd like to re-train)")
else:
    for epoch in range(n_epochs):
        print(f'Epoch {epoch}')
        for step, batch in enumerate(train_dataloader):
        
            batch = batch[0].to(device)

            # Algorithm 1 line 3: sample t uniformally for every example in the batch
            t = torch.randint(0, T, (batch_size,), device=device).long()

            loss = p_losses(model, batch, t)
            training_loss.append(loss.cpu().detach().numpy().item())

            if step % 100 == 0:
                print(f"  Loss (step {step}):", loss.item())

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    print('Done training!')
    torch.save(model.state_dict(), model_outputfile)
    print(f'Saved model: {model_outputfile}')
    plt.plot(training_loss)
    plt.xlabel('Training step')
    plt.ylabel('Loss')
    plt.savefig(str(results_folder / f'loss.png'))
    plt.clf()

#---------------------------------------------
# Sampling from the trained model
#---------------------------------------------
print()
print('------------------ Sampling ------------------')
print('--------------------------------------------')
# sample 64 images
samples = sample(model, image_size=image_dim, batch_size=batch_size, channels=1)
for random_index in range(10):

    # Plot
    plt.imshow(samples[-1][random_index].reshape(image_dim, image_dim, 1), cmap="gray")
    plt.savefig(str(results_folder / f'{random_index}_generated.png'))
    plt.clf()

    # Generate a gif of denoising
    fig = plt.figure()
    ims = []
    for i in range(T):
        im = plt.imshow(samples[i][random_index].reshape(image_dim, image_dim, 1), cmap="gray", animated=True)
        ims.append([im])
    animate = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    animate.save(str(results_folder / f'{random_index}_generated.gif'))