import huggingface_hub
import torch
import onnxruntime as rt
import numpy as np
import cv2

from comfyUI.types import *
from .node_base import StableRendererNodeBase

def get_mask(img:torch.Tensor, s=1024):
    img = (img / 255).astype(np.float32)
    h, w = h0, w0 = img.shape[:-1]
    h, w = (s, int(s * w / h)) if h > w else (int(s * h / w), s)
    ph, pw = s - h, s - w
    img_input = np.zeros([s, s, 3], dtype=np.float32)
    img_input[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w] = cv2.resize(img, (w, h))
    img_input = np.transpose(img_input, (2, 0, 1))
    img_input = img_input[np.newaxis, :]
    mask = rmbg_model.run(None, {'img': img_input})[0][0]
    mask = np.transpose(mask, (1, 2, 0))
    mask = mask[ph // 2:ph // 2 + h, pw // 2:pw // 2 + w]
    mask = cv2.resize(mask, (w0, h0))[:, :, np.newaxis]
    return mask

# Declare Execution Providers
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']

# Download and host the model
model_path = huggingface_hub.hf_hub_download(
    "skytnt/anime-seg", "isnetis.onnx")
rmbg_model = rt.InferenceSession(model_path, providers=providers)


# Adopted from abg-comfyui
class StableRendererRemoveImageBackground(StableRendererNodeBase):
    Category = "Processing"

    def __call__(self, image:IMAGE) -> IMAGE:
        """
        Remove background.

        Args:
            image (IMAGE): image with shape [batch, height, width, channels]
        
        Returns:
            IMAGE: image with background removed. Identical shape as input
        """
        if image.ndim == 4 and image.shape[0] > 1:
            ret_images = list(map(remove_background, image))
            return (torch.cat(ret_images, dim=0), )
        else:
            img = remove_background(image)
            return (img, )


class StableRendererRGBA2RGB(StableRendererNodeBase):
    Category = "Processing"

    def __call__(self,
                 image: IMAGE,
                 color: STRING(forceInput=False) = "ffffff"  # type: ignore
        ) -> IMAGE:
        """Convert RGBA image to RGB image.

        Args:
            image (torch.Tensor): image with shape [..., height, width, 4]
            color (str): Hex representation of color in string 

        Returns:
            torch.Tensor: image with shape [height, width, 3]
        """
        assert image.ndim >= 3
        assert image.shape[-1] == 4, "Input image must be in RGBA format"
        assert len(color) == 6, "Color must be a hex string"

        try:
            color = tuple(int(color[i:i+2], 16) for i in [0, 2, 4])
        except ValueError:
            raise ValueError(f"Invalid color format {color}, color must be a hex string")
        
        background = torch.tensor(color)
        rgb, alpha = image[..., :3], image[..., 3]

        return (1 - alpha[..., None]) * background + alpha[..., None] * rgb


class StableRendererRGBAThreshold(StableRendererNodeBase):
    Category = "Processing"

    def __call__(self,
                 image: IMAGE,
                 threshold: FLOAT(0.0, 1.0, 0.01, round=0.01) = 0.5  # type: ignore
        ) -> IMAGE:
        """Threshold the alpha channel of an RGBA image.

        Args:
            image (torch.Tensor): image with shape [..., height, width, 4]
            threshold (float): threshold value for alpha channel

        Returns:
            torch.Tensor: image with shape [height, width, 4]
        """
        assert image.ndim >= 3
        assert image.shape[-1] == 4, "Input image must be in RGBA format"

        alpha = image[..., 3]
        mask = alpha > threshold

        return torch.cat([image[..., :3], mask[..., None]], dim=-1)


# region: Remove background
def remove_background(image: torch.Tensor) -> torch.Tensor:
    """remove background

    Args:
        image (torch.Tensor): image with shape [height, width, channels]

    Returns:
        torch.Tensor: image with background removed
    """
    assert image.ndim == 3
    assert image.shape[-1] == 3, "Input image must be in RGB format"

    npa = image2nparray(image)
    rmb = rmbg_fn(npa)
    img = nparray2image(rmb)

    return img


def image2nparray(image:torch.Tensor):
    narray:np.array = np.clip(255. * image.cpu().numpy().squeeze(),0, 255).astype(np.uint8)
    if narray.shape[-1] == 4:
        narray =  narray[..., [2, 1, 0, 3]]  # For RGBA
    else:
        narray = narray[..., [2, 1, 0]]  # For RGB
    return narray


def rmbg_fn(img):
    mask = get_mask(img)
    img = (mask * img + 255 * (1 - mask)).astype(np.uint8)
    mask = (mask * 255).astype(np.uint8)
    img = np.concatenate([img, mask], axis=2, dtype=np.uint8)
    mask = mask.repeat(3, axis=2)
    return img


def nparray2image(narray:np.array):
    print(f"narray shape: {narray.shape}")
    if narray.shape[-1] == 4:
        narray =  narray[..., [2, 1, 0, 3]]
    else:
        narray =  narray[..., [2, 1, 0]] 
    tensor = torch.from_numpy(narray/255.).float().unsqueeze(0)
    return tensor

# endregion


__all__ = ["StableRendererRemoveImageBackground", "StableRendererRGBA2RGB"]