import torch

device = torch.device("cuda" if torch.cuda.is_available() else 
                      "mps" if torch.backends.mps.is_available() else
                      "cpu")
dtype = torch.float16

test_dir = "test"

sd_model_path = "models/Stable-Diffusion/AIDv2.10.safetensors"
vae_model_path = "models/VAE/vae-ft-mse-840000-ema-pruned.ckpt"
vae_approx_model_path = "models/VAE-Approx/vae-approx.pt"
neg_emb_path = "models/embeddings/aid210.pt"
