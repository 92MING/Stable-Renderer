import torch
import os
import sys
try:
    from common_utils.path_utils import PROJECT_DIR
except Exception as e:
    PROJECT_DIR = "."
from pathlib import Path

sd_root = Path(PROJECT_DIR) / "source/sd"
device = torch.device("cuda" if torch.cuda.is_available() else
                      "mps" if torch.backends.mps.is_available() else
                      "cpu")
dtype = torch.float16

test_dir = sd_root / 'test'
model_dir = sd_root / 'models'
sd_model_path = model_dir / "Stable-Diffusion/dreamshaper_8.safetensors"
vae_model_path = model_dir / "VAE/vae-ft-mse-840000-ema-pruned.ckpt"
vae_approx_model_path = model_dir / "VAE-Approx/vae-approx.pt"
neg_emb_path = model_dir / "embeddings/aid210.pt"
