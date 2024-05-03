import os
import numpy as np
import torch
from torch import nn
from PIL import Image
from . import config

sd_vae_approx_model = None


@torch.no_grad()
def latents_to_single_pil_approx(latents):
    """
    Fast decode a latent to an pil image for preview.

    Parameters:
    sample : latent tensor with shape (4, h, w), where h and w denote the height and width of latent image respectively.
    """
    x_sample = model()(latents[0].to(config.device, config.dtype).unsqueeze(0))[0].detach() * 0.5 + 0.5
    x_sample = torch.clamp(x_sample, min=0.0, max=1.0)
    x_sample = 255. * np.moveaxis(x_sample.cpu().numpy(), 0, 2)
    x_sample = x_sample.astype(np.uint8)

    return Image.fromarray(x_sample)


class VAEApprox(nn.Module):
    def __init__(self):
        super(VAEApprox, self).__init__()
        self.conv1 = nn.Conv2d(4, 8, (7, 7))
        self.conv2 = nn.Conv2d(8, 16, (5, 5))
        self.conv3 = nn.Conv2d(16, 32, (3, 3))
        self.conv4 = nn.Conv2d(32, 64, (3, 3))
        self.conv5 = nn.Conv2d(64, 32, (3, 3))
        self.conv6 = nn.Conv2d(32, 16, (3, 3))
        self.conv7 = nn.Conv2d(16, 8, (3, 3))
        self.conv8 = nn.Conv2d(8, 3, (3, 3))

    def forward(self, x):
        extra = 11
        x = nn.functional.interpolate(x, (x.shape[2] * 2, x.shape[3] * 2))
        x = nn.functional.pad(x, (extra, extra, extra, extra))

        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7, self.conv8, ]:
            x = layer(x)
            x = nn.functional.leaky_relu(x, 0.1)

        return x


class VAEApproxInverse(nn.Module):
    def __init__(self):
        super(VAEApproxInverse, self).__init__()

        # Note: The input and output channels are swapped for the deconvolution layers.
        self.deconv8 = nn.ConvTranspose2d(3, 8, (3, 3))
        self.deconv7 = nn.ConvTranspose2d(8, 16, (3, 3))
        self.deconv6 = nn.ConvTranspose2d(16, 32, (3, 3))
        self.deconv5 = nn.ConvTranspose2d(32, 64, (3, 3))
        self.deconv4 = nn.ConvTranspose2d(64, 32, (3, 3))
        self.deconv3 = nn.ConvTranspose2d(32, 16, (3, 3))
        self.deconv2 = nn.ConvTranspose2d(16, 8, (5, 5))
        self.deconv1 = nn.ConvTranspose2d(8, 4, (7, 7))

    def forward(self, x):
        extra = 11

        for layer in [self.deconv8, self.deconv7, self.deconv6, self.deconv5, self.deconv4, self.deconv3, self.deconv2, self.deconv1]:
            x = nn.functional.leaky_relu(x, 0.1)
            x = layer(x)

        x = nn.functional.pad(x, (-extra, -extra, -extra, -extra))
        x = nn.functional.interpolate(x, (x.shape[2] // 2, x.shape[3] // 2))

        return x


def model():
    global sd_vae_approx_model

    if sd_vae_approx_model is None:
        model_path = config.vae_approx_model_path
        sd_vae_approx_model = VAEApprox()
        sd_vae_approx_model.load_state_dict(torch.load(model_path, map_location='cpu' if config.device.type != 'cuda' else None))
        sd_vae_approx_model.eval()
        sd_vae_approx_model.to(config.device, dtype=config.dtype)

    return sd_vae_approx_model


def cheap_approximation(sample):
    # https://discuss.huggingface.co/t/decoding-latents-to-rgb-without-upscaling/23204/2

    coefs = torch.tensor([
        [0.298, 0.207, 0.208],
        [0.187, 0.286, 0.173],
        [-0.158, 0.189, 0.264],
        [-0.184, -0.271, -0.473],
    ]).to(sample.device)

    x_sample = torch.einsum("lxy,lr -> rxy", sample, coefs)

    return x_sample
