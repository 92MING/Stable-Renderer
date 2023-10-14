import random
import torch
from diffusers import AutoencoderKL
from PIL import Image
from .modules import config
from .modules.vae import encode, decode

if __name__ == '__main__':
    img_path = config.test_dir / 'groups/group_7/frame_132.png'
    image = Image.open(img_path)

    # Prepare torch generator
    seed = random.randint(0, 9999999999)
    generator = torch.Generator(device="cuda").manual_seed(seed)

    # Load VAE model
    vae = AutoencoderKL.from_single_file(
        config.vae_model_path,
        torch_dtype=config.dtype,
        local_files_only=True
    ).to(config.device)

    # Encode example
    encode_res = encode(vae, image, config.device, config.dtype, generator)

    # Decode example
    decode_res = decode(vae, encode_res, return_pil=False)
