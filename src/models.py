from torch.nn import functional as F 
import torch.nn as nn
from utils import *
import math
from typing import Union, Tuple, List, Optional
from flow_matching.utils import ModelWrapper
from tqdm import tqdm

__all__ = ["SRCNN",
           "FCNN",
           "Diffusion",
           "UNet",
           "WrappedModel"]

class SRCNN(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, hidden_dim=64, num_blocks=3, scale_factor=2):
        super(SRCNN, self).__init__()
        self.initial_conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=5, padding=2)
        self.res_blocks = nn.Sequential(
            *[ResBlock(hidden_dim, hidden_dim) for _ in range(num_blocks)]
        )
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear', align_corners=False)
        self.final_res = ResBlock(hidden_dim,hidden_dim)
        self.final_conv = nn.Conv2d(hidden_dim, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        x = self.initial_conv(x)
        x = self.res_blocks(x)
        x = self.upsample(x)
        x = self.final_res(x)
        x = self.final_conv(x)
        return x
    
    
class FCNN(nn.Module):
    def __init__(self, 
                 dimensions : list, 
                 activation :str = 'relu',
                 indim : int = 2,
                 outdim : int = 1,
                 init : str = None,
                 positive_output=True,
                 residual=False):
        """
        dimensions: neural network hidden dimensions
        if last_layer_relu is True, constrain the outputs to be non-negative
        """
        super().__init__()
        in_dims = [indim] + dimensions
        out_dims = dimensions + [outdim]
        layerList = []
        for dims in zip(in_dims, out_dims):
            in_dim, out_dim = dims
            layerList.append(nn.Linear(in_dim, out_dim))
        self.layers = nn.ModuleList(layerList)
        self.act = choose_act(activation)

        # initialize linear layer weights
        if init is not None:
            assert type(init) is str
            init_linear(self, init)

        self.positive_output = positive_output
        self.res = residual

    def forward(self, x):
        for _, l in enumerate(self.layers[:-1]):
            if self.res:
                x = x + self.act(l(x))
            else:
                x = self.act(l(x))
            # x = nn.BatchNorm1d(x.size(1))(x)
                
        x = self.layers[-1](x)

        # truncate to positve values to make outputs physically meaningful
        if self.positive_output:
            x = F.relu(x)
        
        return x

        
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
#         self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.GELU()#ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding)
#         self.bn2 = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        identity = x
        out = self.conv1(x)
#         out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
#         out = self.bn2(out)
        out += identity  # Residual connection
        return self.act(out)
    
class Diffusion(nn.Module):
    def __init__(self, model, image_resolution=[32, 32, 3], n_times=1000, beta_minmax=[1e-4, 2e-2], device='cuda'):
    
        super(Diffusion, self).__init__()
    
        self.n_times = n_times
        self.img_H, self.img_W, self.img_C = image_resolution

        self.model = model
        
        # define linear variance schedule(betas)
        beta_1, beta_T = beta_minmax
        betas = torch.linspace(start=beta_1, end=beta_T, steps=n_times).to(device) # follows DDPM paper
        self.sqrt_betas = torch.sqrt(betas)
                                     
        # define alpha for forward diffusion kernel
        self.alphas = 1 - betas
        self.sqrt_alphas = torch.sqrt(self.alphas)
        alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1-alpha_bars)
        self.sqrt_alpha_bars = torch.sqrt(alpha_bars)
        
        self.device = device
    
    def extract(self, a, t, x_shape):
        """
            from lucidrains' implementation
                https://github.com/lucidrains/denoising-diffusion-pytorch/blob/beb2f2d8dd9b4f2bd5be4719f37082fe061ee450/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L376
        """
        b, *_ = t.shape
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))
    
    # def scale_to_minus_one_to_one(self, x):
    #     # according to the DDPMs paper, normalization seems to be crucial to train reverse process network
    #     return x * 2 - 1
    
    # def reverse_scale_to_zero_to_one(self, x):
    #     return (x + 1) * 0.5
    
    def make_noisy(self, x_zeros, t): 
        # perturb x_0 into x_t (i.e., take x_0 samples into forward diffusion kernels)
        epsilon = torch.randn_like(x_zeros).to(self.device)
        
        sqrt_alpha_bar = self.extract(self.sqrt_alpha_bars, t, x_zeros.shape)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, t, x_zeros.shape)
        
        # Let's make noisy sample!: i.e., Forward process with fixed variance schedule
        #      i.e., sqrt(alpha_bar_t) * x_zero + sqrt(1-alpha_bar_t) * epsilon
        noisy_sample = x_zeros * sqrt_alpha_bar + epsilon * sqrt_one_minus_alpha_bar
    
        return noisy_sample.detach(), epsilon
    
    
    def forward(self, x_zeros):
        # x_zeros = self.scale_to_minus_one_to_one(x_zeros)
        
        B, _, _, _ = x_zeros.shape
        
        # (1) randomly choose diffusion time-step
        t = torch.randint(low=0, high=self.n_times, size=(B,)).long().to(self.device)
        
        # (2) forward diffusion process: perturb x_zeros with fixed variance schedule
        perturbed_images, epsilon = self.make_noisy(x_zeros, t)
        
        # (3) predict epsilon(noise) given perturbed data at diffusion-timestep t.
        pred_epsilon = self.model(perturbed_images, t)
        
        return perturbed_images, epsilon, pred_epsilon
    
    
    def denoise_at_t(self, x_t, timestep, t):
        B, _, _, _ = x_t.shape
        if t > 1:
            z = torch.randn_like(x_t).to(self.device)
        else:
            z = torch.zeros_like(x_t).to(self.device)
        
        # at inference, we use predicted noise(epsilon) to restore perturbed data sample.
        epsilon_pred = self.model(x_t, timestep)
        
        alpha = self.extract(self.alphas, timestep, x_t.shape)
        sqrt_alpha = self.extract(self.sqrt_alphas, timestep, x_t.shape)
        sqrt_one_minus_alpha_bar = self.extract(self.sqrt_one_minus_alpha_bars, timestep, x_t.shape)
        sqrt_beta = self.extract(self.sqrt_betas, timestep, x_t.shape)
        
        # denoise at time t, utilizing predicted noise
        x_t_minus_1 = 1 / sqrt_alpha * (x_t - (1-alpha)/sqrt_one_minus_alpha_bar*epsilon_pred) + sqrt_beta*z
        
        return x_t_minus_1#.clamp(min=0.0)
                
    def sample(self, N=5, num_steps=0, batch_size=1):
        # DDPM sampling
        
        # autoregressively denoise from x_T to x_0
        #     i.e., generate image from noise, x_T
        images = torch.zeros(1, self.img_C, self.img_H, self.img_W).to(self.device)
        nbatch = N//batch_size if N%batch_size==0 else N//batch_size + 1
        for b in range(nbatch):
            if b==nbatch-1:
                nimage = N - batch_size*(nbatch-1)
            else:
                nimage = batch_size
            x_t = torch.randn((nimage, self.img_C, self.img_H, self.img_W)).to(self.device)
            for t in tqdm(range(self.n_times-1, -1, -1)):
                timestep = torch.tensor([t]).repeat_interleave(nimage, dim=0).long().to(self.device)
                x_t = self.denoise_at_t(x_t, timestep, t)

            images = torch.cat((images, x_t), dim=0)

        return images[1:]


###### ---- UNet ---- #######
class TimeEmbedding(nn.Module):
    """
    ### Embeddings for time steps
    Generates sinusoidal positional embeddings and processes them through a simple MLP.
    """

    def __init__(self, n_channels: int):
        """
        * `n_channels` is the number of dimensions in the embedding
        """
        super().__init__()
        self.n_channels = n_channels

        # Define the activation function
        self.act = nn.SiLU()

        # MLP layers
        self.lin1 = nn.Linear(self.n_channels // 4, self.n_channels)
        self.lin2 = nn.Linear(self.n_channels, self.n_channels)

    def forward(self, t: torch.Tensor):
        """
        * `t` has shape `[batch_size]`
        """
        # Create sinusoidal position embeddings
        half_dim = self.n_channels // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=1)  # Shape: [batch_size, half_dim * 2]

        # Transform with the MLP
        emb = self.act(self.lin1(emb))  # Shape: [batch_size, n_channels]
        emb = self.lin2(emb)            # Shape: [batch_size, n_channels]

        return emb


class ResidualBlock(nn.Module):
    """
    ### Residual block
    A residual block with two convolutional layers, group normalization, and a residual connection.
    It also incorporates time step embeddings.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int,
                 n_groups: int = 32, dropout: float = 0.1):
        """
        * `in_channels` is the number of input channels
        * `out_channels` is the number of output channels
        * `time_channels` is the number of channels in the time step embeddings
        * `n_groups` is the number of groups for group normalization
        * `dropout` is the dropout rate
        """
        super().__init__()
        # self.norm1 = nn.GroupNorm(n_groups, in_channels)
        self.act1 = nn.SiLU()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # self.norm2 = nn.GroupNorm(n_groups, out_channels)
        self.act2 = nn.SiLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Shortcut connection
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()

        # Time embedding
        self.time_emb = nn.Linear(time_channels, out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        h = self.conv1(self.act1(x))  # First convolution
        h += self.time_emb(t)[:, :, None, None]   # Add time embeddings
        h = self.conv2(self.dropout(self.act2(h)))  # Second convolution
        return h + self.shortcut(x)  # Residual connection


class AttentionBlock(nn.Module):
    """
    ### Attention block
    Implements multi-head self-attention similar to transformer architectures.
    """

    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None, n_groups: int = 32):
        """
        * `n_channels` is the number of channels in the input
        * `n_heads` is the number of heads in multi-head attention
        * `d_k` is the number of dimensions in each head
        * `n_groups` is the number of groups for group normalization
        """
        super().__init__()
        if d_k is None:
            d_k = n_channels

        # self.norm = nn.GroupNorm(n_groups, n_channels)
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)  # Query, Key, Value
        self.output = nn.Linear(n_heads * d_k, n_channels)
        self.scale = d_k ** -0.5
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` is unused but kept for compatibility
        """
        batch_size, n_channels, height, width = x.shape
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)  # [batch, seq, channels]

        qkv = self.projection(x)  # [batch, seq, n_heads * d_k * 3]
        qkv = qkv.view(batch_size, -1, self.n_heads, 3 * self.d_k)  # [batch, seq, n_heads, 3*d_k]
        q, k, v = qkv.chunk(3, dim=-1)  # Each: [batch, seq, n_heads, d_k]

        # Scaled dot-product attention
        attn = torch.einsum('bshd,bsHd->bhsH', q, k) * self.scale  # [batch, n_heads, seq, seq]
        attn = attn.softmax(dim=-1)
        res = torch.einsum('bhsH,bshd->bshd', attn, v)  # [batch, n_heads, seq, d_k]
        res = res.reshape(batch_size, -1, self.n_heads * self.d_k)  # [batch, seq, n_heads*d_k]
        res = self.output(res)  # [batch, seq, n_channels]
        res += x  # Residual connection

        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)  # [batch, channels, height, width]
        return res

class DownBlock(nn.Module):
    """
    ### Down block
    Combines ResidualBlock and AttentionBlock for the downsampling path.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        self.res = ResidualBlock(in_channels, out_channels, time_channels)
        self.attn = AttentionBlock(out_channels) if has_attn else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class UpBlock(nn.Module):
    """
    ### Up block
    Combines ResidualBlock and AttentionBlock for the upsampling path.
    """

    def __init__(self, in_channels: int, out_channels: int, time_channels: int, has_attn: bool):
        super().__init__()
        # The input has `in_channels + out_channels` due to concatenation with skip connections
        self.res = ResidualBlock(in_channels + out_channels, out_channels, time_channels)
        self.attn = AttentionBlock(out_channels) if has_attn else nn.Identity()

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res(x, t)
        x = self.attn(x)
        return x


class MiddleBlock(nn.Module):
    """
    ### Middle block
    The bottleneck of the U-Net, combining Residual and Attention blocks.
    """

    def __init__(self, n_channels: int, time_channels: int):
        super().__init__()
        self.res1 = ResidualBlock(n_channels, n_channels, time_channels)
        self.attn = AttentionBlock(n_channels)
        self.res2 = ResidualBlock(n_channels, n_channels, time_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x


class Upsample(nn.Module):
    """
    ### Upsample
    Scales up the feature map by a factor of 2 using transposed convolution.
    """

    def __init__(self, n_channels: int):
        super().__init__()
        self.conv = nn.ConvTranspose2d(n_channels, n_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # `t` is unused but kept for compatibility
        return self.conv(x)


class Downsample(nn.Module):
    """
    ### Downsample
    Scales down the feature map by a factor of 2 using convolution.
    """

    def __init__(self, n_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        # `t` is unused but kept for compatibility
        return self.conv(x)


class UNet(nn.Module):
    """
    ## U-Net Architecture
    A flexible U-Net implementation with residual and attention blocks.
    """

    def __init__(self, image_channels: int = 1, n_channels: int = 64,
                 ch_mults: Union[Tuple[int, ...], List[int]] = (1, 2, 3, 4),
                 is_attn: Union[Tuple[bool, ...], List[bool]] = (False, False, True, True),
                 n_blocks: int = 2):
        """
        * `image_channels` is the number of channels in the input image (e.g., 3 for RGB)
        * `n_channels` is the number of channels in the initial feature map
        * `ch_mults` is a list specifying channel multipliers at each resolution
        * `is_attn` is a list indicating whether to use attention at each resolution
        * `n_blocks` is the number of Down/Up blocks at each resolution
        """
        super().__init__()

        assert len(ch_mults) == len(is_attn), "ch_mults and is_attn must have the same length"
        self.n_resolutions = len(ch_mults)

        # Initial convolution to project the image to the feature map
        self.image_proj = nn.Conv2d(image_channels, n_channels, kernel_size=3, padding=1)

        # Time embedding
        self.time_emb = TimeEmbedding(n_channels * 4)

        # Downsampling path
        down_blocks = []
        in_channels = n_channels
        for i in range(self.n_resolutions):
            out_channels = in_channels * ch_mults[i]
            for _ in range(n_blocks):
                down_blocks.append(DownBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
                in_channels = out_channels
            if i < self.n_resolutions - 1:
                down_blocks.append(Downsample(in_channels))
        self.down = nn.ModuleList(down_blocks)

        # Middle block
        self.middle = MiddleBlock(out_channels, n_channels * 4)

        # Upsampling path
        up_blocks = []
        for i in reversed(range(self.n_resolutions)):
            out_channels = in_channels
            for _ in range(n_blocks):
                up_blocks.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
            if i > 0:
                out_channels = in_channels // ch_mults[i]
                up_blocks.append(UpBlock(in_channels, out_channels, n_channels * 4, is_attn[i]))
                up_blocks.append(Upsample(out_channels))
                in_channels = out_channels
        self.up = nn.ModuleList(up_blocks)

        # Final normalization and convolution
        self.norm = nn.GroupNorm(8, in_channels)
        self.act = nn.SiLU()
        self.final = nn.Conv2d(in_channels, image_channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        * `x` has shape `[batch_size, image_channels, height, width]`
        * `t` has shape `[batch_size]` representing time steps
        """
        # Generate time embeddings
        t_emb = self.time_emb(t)  # Shape: [batch_size, n_channels * 4]

        # Initial projection
        x = self.image_proj(x)  # Shape: [batch_size, n_channels, height, width]

        # Downsampling path with skip connections
        skip_connections = [x]
        for layer in self.down:
            x = layer(x, t_emb)
            skip_connections.append(x)

        # Middle block
        x = self.middle(x, t_emb)

        # Upsampling path with skip connections
        for layer in self.up:
            if isinstance(layer, Upsample):
                x = layer(x, t_emb)
            else:
                skip = skip_connections.pop()
                x = torch.cat([x, skip], dim=1)  # Concatenate along the channel dimension
                x = layer(x, t_emb)

        # Final normalization and activation
        x = self.act(x)
        x = self.final(x)
        return x


# Velocity Field Wrapper
class WrappedModel(ModelWrapper):
    
    def __init__(self, model: nn.Module):
        super().__init__(model)
        self.device = next(model.parameters()).device
        
    def forward(self, x: torch.Tensor, t: torch.Tensor):
        if t.dim() == 0:  # t is a scalar
            t = t.repeat(x.shape[0])
        return self.model(x, t)