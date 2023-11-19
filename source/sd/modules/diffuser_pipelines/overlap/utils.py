from typing import List, Literal
from torchvision.transforms import ToTensor
import torch.nn.functional as F
from PIL import Image
import torch


def overlap_rate(ovlp_seq: List[torch.Tensor], threshold: float = 0.0):
    """
    Debug function to calculate the overlap rate of a sequence of frames.
    :param ovlp_seq: A list of overlapped frames. Note that this should haven't been overlapped with the original frames.
    """
    num_zeros = torch.sum(torch.stack([torch.sum(abs(latents - threshold) <= 0) for latents in ovlp_seq]))
    num_nonzeros = torch.sum(torch.stack([torch.sum(abs(latents - threshold) > 0) for latents in ovlp_seq]))
    return num_nonzeros / (num_nonzeros + num_zeros)

def value_interpolation(
        x: float,
        start: float,
        end: float, 
        interpolate_function: Literal["constant", "linear", "cosine", "exponential"] = "constant"):
    r"""
    Interpolate value from `start` to `end` with `interpolate_function`, controled by `x`
    :param x: float between 0 to 1 to control interpolation
    :param start: start value
    :param end: end value
    :param interpolate_function: function used for interpolation
    """ 

    if interpolate_function == "constant":
        return start
    elif interpolate_function == "linear":
        return start + (end - start) * x
    elif interpolate_function == "cosine":
        return start + (end - start) * (1 + torch.cos(x * torch.pi)) / 2
    elif interpolate_function == "exponential":
        return start * (end / start) ** x
    else:
        raise NotImplementedError

def build_view_normal_map(normal_images: List[Image.Image], view_vector: torch.Tensor, dtype=torch.float32):
    normal_map = [ToTensor()(normal_image).permute(1, 2, 0) for normal_image in normal_images]
    normal_map = torch.stack(normal_map, dim=0).unsqueeze(3)    # [T, H, W, 1, 3]

    view_vector = F.normalize(view_vector.to(dtype), p=2, dim=0).expand_as(normal_map)   # [T, H, W, 1, 3]

    view_normal_map = torch.abs(torch.einsum("thwcd, thwcd -> thwc", normal_map, view_vector)) # dot product
    return view_normal_map