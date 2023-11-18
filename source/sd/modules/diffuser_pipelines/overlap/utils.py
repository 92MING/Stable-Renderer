from typing import List, Literal
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