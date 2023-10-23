from typing import List
import torch


def overlap_rate(ovlp_seq: List[torch.Tensor], threshold: float = 0.0):
    """
    Debug function to calculate the overlap rate of a sequence of frames.
    :param ovlp_seq: A list of overlapped frames. Note that this should haven't been overlapped with the original frames.
    """
    num_zeros = torch.sum(torch.stack([torch.sum(abs(latents - threshold) <= 0) for latents in ovlp_seq]))
    num_nonzeros = torch.sum(torch.stack([torch.sum(abs(latents - threshold) > 0) for latents in ovlp_seq]))
    return num_nonzeros / (num_nonzeros + num_zeros)