import torch
import einops
import numpy as np
import torch.nn as nn
from torch import Tensor
from functools import partial
from torchdiffeq import odeint

from unet import UNetModel
from diffusers import AutoencoderKL


def exists(val):
    return val is not None


class DepthFM(nn.Module):
    def __init__(self, ckpt_path: str):
        super().__init__()
        vae_id = "runwayml/stable-diffusion-v1-5"
        self.vae = AutoencoderKL.from_pretrained(vae_id, subfolder="vae")
        self.scale_factor = 0.18215

        # set with checkpoint
        ckpt = torch.load(ckpt_path, map_location="cpu")
        self.noising_step = ckpt['noising_step']
        self.empty_text_embed = ckpt['empty_text_embedding']
        self.model = UNetModel(**ckpt['ldm_hparams'])
        self.model.load_state_dict(ckpt['state_dict'])
    
    def ode_fn(self, t: Tensor, x: Tensor, **kwargs):
        if t.numel() == 1:
            t = t.expand(x.size(0))
        return self.model(x=x, t=t, **kwargs)
    
    def generate(self, z: Tensor, num_steps: int = 4, n_intermediates: int = 0, **kwargs):
        """
        ODE solving from z0 (ims) to z1 (depth).
        """
        ode_kwargs = dict(method="euler", rtol=1e-5, atol=1e-5, options=dict(step_size=1.0 / num_steps))
        
        # t specifies which intermediate times should the solver return
        # e.g. t = [0, 0.5, 1] means return the solution at t=0, t=0.5 and t=1
        # but it also specifies the number of steps for fixed step size methods
        t = torch.linspace(0, 1, n_intermediates + 2, device=z.device, dtype=z.dtype)
        # t = torch.tensor([0., 1.], device=z.device, dtype=z.dtype)

        # allow conditioning information for model
        ode_fn = partial(self.ode_fn, **kwargs)
        
        ode_results = odeint(ode_fn, z, t, **ode_kwargs)
        
        if n_intermediates > 0:
            return ode_results
        return ode_results[-1]
    
    def forward(self, ims: Tensor, num_steps: int = 4, ensemble_size: int = 1):
        """
        Args:
            ims: Tensor of shape (b, 3, h, w) in range [-1, 1]
        Returns:
            depth: Tensor of shape (b, 1, h, w) in range [0, 1]
        """
        if ensemble_size > 1:
            assert ims.shape[0] == 1, "Ensemble mode only supported with batch size 1"
            ims = ims.repeat(ensemble_size, 1, 1, 1)
        
        bs, dev = ims.shape[0], ims.device

        ims_z = self.encode(ims, sample_posterior=False)

        conditioning = torch.tensor(self.empty_text_embed).to(dev).repeat(bs, 1, 1)
        context = ims_z
        
        x_source = ims_z

        if self.noising_step > 0:
            x_source = q_sample(x_source, self.noising_step)    

        # solve ODE
        depth_z = self.generate(x_source, num_steps=num_steps, context=context, context_ca=conditioning)

        depth = self.decode(depth_z)
        depth = depth.mean(dim=1, keepdim=True)

        if ensemble_size > 1:
            depth = depth.mean(dim=0, keepdim=True)
        
        # normalize depth maps to range [-1, 1]
        depth = per_sample_min_max_normalization(depth.exp())

        return depth
    
    @torch.no_grad()
    def predict_depth(self, ims: Tensor, num_steps: int = 4, ensemble_size: int = 1):
        """ Inference method for DepthFM. """
        return self.forward(ims, num_steps, ensemble_size)
    
    @torch.no_grad()
    def encode(self, x: Tensor, sample_posterior: bool = True):
        posterior = self.vae.encode(x)
        if sample_posterior:
            z = posterior.latent_dist.sample()
        else:
            z = posterior.latent_dist.mode()
        # normalize latent code
        z = z * self.scale_factor
        return z
    
    @torch.no_grad()
    def decode(self, z: Tensor):
        z = 1.0 / self.scale_factor * z
        return self.vae.decode(z).sample


def sigmoid(x):
  return 1 / (1 + np.exp(-x))


def cosine_log_snr(t, eps=0.00001):
    """
    Returns log Signal-to-Noise ratio for time step t and image size 64
    eps: avoid division by zero
    """
    return -2 * np.log(np.tan((np.pi * t) / 2) + eps)


def cosine_alpha_bar(t):
    return sigmoid(cosine_log_snr(t))


def q_sample(x_start: torch.Tensor, t: int, noise: torch.Tensor = None, n_diffusion_timesteps: int = 1000):
    """
    Diffuse the data for a given number of diffusion steps. In other
    words sample from q(x_t | x_0).
    """
    dev = x_start.device
    dtype = x_start.dtype

    if noise is None:
        noise = torch.randn_like(x_start)
    
    alpha_bar_t = cosine_alpha_bar(t / n_diffusion_timesteps)
    alpha_bar_t = torch.tensor(alpha_bar_t).to(dev).to(dtype)

    return torch.sqrt(alpha_bar_t) * x_start + torch.sqrt(1 - alpha_bar_t) * noise


def per_sample_min_max_normalization(x):
    """ Normalize each sample in a batch independently
    with min-max normalization to [0, 1] """
    bs, *shape = x.shape
    x_ = einops.rearrange(x, "b ... -> b (...)")
    min_val = einops.reduce(x_, "b ... -> b", "min")[..., None]
    max_val = einops.reduce(x_, "b ... -> b", "max")[..., None]
    x_ = (x_ - min_val) / (max_val - min_val)
    return x_.reshape(bs, *shape)
