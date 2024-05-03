import torch
import numpy
import cv2
from PIL import Image
from diffusers import AutoencoderKL
from diffusers.utils import (
    PIL_INTERPOLATION,
    numpy_to_pil
)
from typing import Union


def preprocess_image(image: Union[Image.Image, numpy.ndarray, torch.Tensor], batch_size: int = 1):
    """
    Preprocess an image to be compatible with the UNet and VAE model.
    """
    if isinstance(image, Image.Image):
        w, h = image.size
        w, h = (x - x % 8 for x in (w, h))  # resize to integer multiple of 8
        image = image.resize((w, h), resample=PIL_INTERPOLATION["lanczos"])
    elif isinstance(image, numpy.ndarray):
        w, h = image.shape[:2] if len(image.shape) == 3 else image.shape[1:3]
        w, h = (x - x % 8 for x in (w, h))
        image = cv2.resize(image, (w, h), interpolation=cv2.INTER_LANCZOS4)
    elif isinstance(image, torch.Tensor):
        pass
    image = numpy.array(image).astype(numpy.float32) / 255.0
    image = numpy.vstack([image[None].transpose(0, 3, 1, 2)] * batch_size)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


@torch.no_grad()
def encode(image: Union[torch.Tensor, Image.Image, numpy.ndarray], vae: AutoencoderKL, device: torch.device, dtype: torch.dtype, generator: torch.Generator,):
    """
    Encode an image to a latent vector.
    :param vae: VAE model
    :param image: image tensor or PIL image
    :param device: torch device
    :param dtype: torch dtype
    :param generator: torch generator
    :return: latent vector
    """
    image = preprocess_image(image)
    image = image.to(device=device, dtype=dtype)
    latent_dist = vae.encode(image).latent_dist
    latents = latent_dist.sample(generator=generator)
    latents = vae.config.scaling_factor * latents
    return latents


@torch.no_grad()
def decode(latents, vae: AutoencoderKL, return_pil: bool = False) -> list:
    """
    Decode a latent vector to images. Since latents can be a batch, a list of images is returned.
    :param vae: VAE model
    :param latents: latent vector
    :param return_pil: whether to return PIL image or numpy array
    :return: list of decoded images
    """
    latents = 1 / vae.config.scaling_factor * latents
    image = vae.decode(latents).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    # we always cast to float32 as this does not cause significant overhead and is compatible with bfloat16
    image = image.cpu().permute(0, 2, 3, 1).float().numpy()
    if return_pil:
        image = numpy_to_pil(image)
    return image
