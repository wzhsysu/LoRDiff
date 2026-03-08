# Copyright 2020 by Gongfan Fang, Zhejiang University.
# All rights reserved.
import os
import sys
import warnings
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from transformers import AutoTokenizer, CLIPTextModel
from diffusers import DDPMScheduler
from diffusers.utils.peft_utils import set_weights_and_activate_adapters
from diffusers.utils.import_utils import is_xformers_available
from peft import LoraConfig
from peft.tuners.tuners_utils import onload_layer
from peft.utils import _get_submodules, ModulesToSaveWrapper
from peft.utils.other import transpose

sys.path.append(os.getcwd())
from src.models.autoencoder_kl import AutoencoderKL
from src.models.unet_2d_condition import UNet2DConditionModel
from src.my_utils.vaehook import VAEHook


class L_color(nn.Module):
    def __init__(self):
        super(L_color, self).__init__()

    def forward(self, x):
        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg, mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.pow(mr-mg,2)
        Drb = torch.pow(mr-mb,2)
        Dgb = torch.pow(mb-mg,2)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)
        k = torch.mean(k)
        return k

class L_exp(nn.Module):
    def __init__(self,patch_size=11):
        super(L_exp, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)
    def forward(self, x, mean_val=0.6):
        # exposure control
        x = torch.mean(x,1,keepdim=True)
        mean = self.pool(x)
        d = torch.mean(torch.pow(mean- torch.FloatTensor([mean_val] ).cuda(),2))
        # exposure balance
        return d

class DEtaE2000Loss(nn.Module):
    def __init__(self):
        super(DEtaE2000Loss, self).__init__()

    def forward(self, input, target):
        """
        input, target: shape (B, 3, H, W) in RGB format, assumed normalized between [0, 1]
        """
        input_lab = self.rgb_to_lab(input)
        target_lab = self.rgb_to_lab(target)

        delta_e = self.delta_e2000(input_lab, target_lab)
        loss = delta_e.mean()
        return loss

    def rgb_to_lab(self, x):
        """Differentiable RGB to LAB conversion."""
        # Assume input x in [0, 1]
        eps = 1e-6

        # Linearize sRGB
        mask = (x > 0.04045).float()
        x = (mask * ((x + 0.055) / 1.055) ** 2.4) + ((1 - mask) * (x / 12.92))

        # sRGB to XYZ
        rgb_to_xyz = torch.tensor([[0.412453, 0.357580, 0.180423],
                                   [0.212671, 0.715160, 0.072169],
                                   [0.019334, 0.119193, 0.950227]], device=x.device, dtype=x.dtype)
        x = x.permute(0, 2, 3, 1)  # B x H x W x C
        x = torch.matmul(x, rgb_to_xyz.T)

        # Normalize for D65 white point
        x[..., 0] /= 0.95047
        x[..., 2] /= 1.08883

        # XYZ to LAB
        mask = (x > 0.008856).float()
        x = (mask * x.pow(1/3)) + ((1 - mask) * (7.787 * x + 16/116))

        L = (116 * x[..., 1]) - 16
        a = 500 * (x[..., 0] - x[..., 1])
        b = 200 * (x[..., 1] - x[..., 2])

        lab = torch.stack([L, a, b], dim=-1)  # B x H x W x 3
        lab = lab.permute(0, 3, 1, 2)  # B x 3 x H x W

        return lab

    def delta_e2000(self, lab1, lab2):
        """Calculate Delta E2000 between two LAB images."""
        L1, a1, b1 = lab1[:,0], lab1[:,1], lab1[:,2]
        L2, a2, b2 = lab2[:,0], lab2[:,1], lab2[:,2]

        avg_L = (L1 + L2) / 2
        C1 = torch.sqrt(a1**2 + b1**2)
        C2 = torch.sqrt(a2**2 + b2**2)
        avg_C = (C1 + C2) / 2

        G = 0.5 * (1 - torch.sqrt(avg_C**7 / (avg_C**7 + 25**7 + 1e-8)))

        a1p = (1 + G) * a1
        a2p = (1 + G) * a2

        C1p = torch.sqrt(a1p**2 + b1**2)
        C2p = torch.sqrt(a2p**2 + b2**2)
        avg_Cp = (C1p + C2p) / 2

        h1p = torch.atan2(b1, a1p) % (2 * torch.pi)
        h2p = torch.atan2(b2, a2p) % (2 * torch.pi)

        avg_hp = (h1p + h2p) / 2
        diff_hp = h2p - h1p
        diff_hp = diff_hp - (2 * torch.pi) * (diff_hp > torch.pi).float()
        diff_hp = diff_hp + (2 * torch.pi) * (diff_hp < -torch.pi).float()

        delta_Lp = L2 - L1
        delta_Cp = C2p - C1p
        delta_Hp = 2 * torch.sqrt(C1p * C2p) * torch.sin(diff_hp / 2)

        Sl = 1 + (0.015 * (avg_L - 50)**2) / torch.sqrt(20 + (avg_L - 50)**2)
        Sc = 1 + 0.045 * avg_Cp
        T = 1 - 0.17 * torch.cos(avg_hp - torch.deg2rad(torch.tensor(30.0))) + \
            0.24 * torch.cos(2 * avg_hp) + \
            0.32 * torch.cos(3 * avg_hp + torch.deg2rad(torch.tensor(6.0))) - \
            0.20 * torch.cos(4 * avg_hp - torch.deg2rad(torch.tensor(63.0)))
        Sh = 1 + 0.015 * avg_Cp * T

        delta_ro = 30 * torch.exp(-((avg_hp - torch.deg2rad(torch.tensor(275.0))) / torch.deg2rad(torch.tensor(25.0)))**2)
        Rc = 2 * torch.sqrt(avg_Cp**7 / (avg_Cp**7 + 25**7 + 1e-8))
        Rt = -torch.sin(2 * delta_ro) * Rc

        delta_E = torch.sqrt(
            (delta_Lp / (Sl + 1e-8))**2 +
            (delta_Cp / (Sc + 1e-8))**2 +
            (delta_Hp / (Sh + 1e-8))**2 +
            Rt * (delta_Cp / (Sc + 1e-8)) * (delta_Hp / (Sh + 1e-8))
        )

        return delta_E


class CSDLoss(torch.nn.Module):
    def __init__(self, args, accelerator):
        super().__init__() 

        self.tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model_path_csd, subfolder="tokenizer")
        self.sched = DDPMScheduler.from_pretrained(args.pretrained_model_path_csd, subfolder="scheduler")
        self.args = args

        weight_dtype = torch.float32
        if accelerator.mixed_precision == "fp16":
            weight_dtype = torch.float16
        elif accelerator.mixed_precision == "bf16":
            weight_dtype = torch.bfloat16

        self.unet_fix = UNet2DConditionModel.from_pretrained(args.pretrained_model_path_csd, subfolder="unet")

        if args.enable_xformers_memory_efficient_attention:
            if is_xformers_available():
                self.unet_fix.enable_xformers_memory_efficient_attention()
            else:
                raise ValueError("xformers is not available, please install it by running `pip install xformers`")

        self.unet_fix.to(accelerator.device, dtype=weight_dtype)

        self.unet_fix.requires_grad_(False)
        self.unet_fix.eval()

    def forward_latent(self, model, latents, timestep, prompt_embeds):
        
        noise_pred = model(
        latents,
        timestep=timestep,
        encoder_hidden_states=prompt_embeds,
        ).sample

        return noise_pred

    def eps_to_mu(self, scheduler, model_output, sample, timesteps):
        alphas_cumprod = scheduler.alphas_cumprod.to(device=sample.device, dtype=sample.dtype)
        alpha_prod_t = alphas_cumprod[timesteps]
        while len(alpha_prod_t.shape) < len(sample.shape):
            alpha_prod_t = alpha_prod_t.unsqueeze(-1)
        beta_prod_t = 1 - alpha_prod_t
        pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
        return pred_original_sample

    def cal_csd(
        self,
        latents,
        prompt_embeds,
        negative_prompt_embeds,
        args,
    ):
        bsz = latents.shape[0]
        min_dm_step = int(self.sched.config.num_train_timesteps * args.min_dm_step_ratio)
        max_dm_step = int(self.sched.config.num_train_timesteps * args.max_dm_step_ratio)

        timestep = torch.randint(min_dm_step, max_dm_step, (bsz,), device=latents.device).long()
        noise = torch.randn_like(latents)
        noisy_latents = self.sched.add_noise(latents, noise, timestep)

        with torch.no_grad():
            noisy_latents_input = torch.cat([noisy_latents] * 2)
            timestep_input = torch.cat([timestep] * 2)
            prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds], dim=0)
            noise_pred = self.forward_latent(
                self.unet_fix,
                latents=noisy_latents_input.to(dtype=torch.float16),
                timestep=timestep_input,
                prompt_embeds=prompt_embeds.to(dtype=torch.float16),
            )
            noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + args.cfg_csd * (noise_pred_text - noise_pred_uncond)
            noise_pred.to(dtype=torch.float32)
            noise_pred_uncond.to(dtype=torch.float32)

            pred_real_latents = self.eps_to_mu(self.sched, noise_pred, noisy_latents, timestep)
            pred_fake_latents = self.eps_to_mu(self.sched, noise_pred_uncond, noisy_latents, timestep)
            

        weighting_factor = torch.abs(latents - pred_real_latents).mean(dim=[1, 2, 3], keepdim=True)

        grad = (pred_fake_latents - pred_real_latents) / weighting_factor
        loss = F.mse_loss(latents, self.stopgrad(latents - grad))

        return loss

    def stopgrad(self, x):
        return x.detach()