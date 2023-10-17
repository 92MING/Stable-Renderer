import torch
import os
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else
                      "cpu")
dtype = torch.float16

test_dir = Path("test").absolute()
model_dir = Path("models").absolute()
sd_model_path = model_dir / "Stable-Diffusion/AIDv2.10.safetensors"
vae_model_path = model_dir / "VAE/vae-ft-mse-840000-ema-pruned.ckpt"
vae_approx_model_path = model_dir / "VAE-Approx/vae-approx.pt"
neg_emb_path = model_dir / "embeddings/aid210.pt"
