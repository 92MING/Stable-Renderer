import torch
import time
import numpy
import tqdm
from typing import List, Literal, Any
import torch.nn.functional as F
from abc import abstractmethod, ABC
from diffusers import AutoencoderKL
from .overlap_scheduler import Scheduler
from .utils import overlap_rate
from ...data_classes.correspondenceMap import CorrespondenceMap
# from ..multi_frame_stable_diffusion import  StableDiffusionImg2VideoPipeline
from ... import log_utils as logu


def overlap(
    frame_seq: List[torch.Tensor],
    corr_map: CorrespondenceMap,
    pipe: Any,
    interpolate_mode: str = 'nearest',
    weight_option: Literal['frame_distance', 'optical_flow'] = 'frame_distance',
    step: int = None,
    timestep: int = None,
    init_latents_orig_seq: List[torch.Tensor] = None,
    noise_seq: List[torch.Tensor] = None,
    **kwargs
):
    """
    Do overlapping with resizing the latents list to the size of the correspondence map.
    :param latents_seq: A list of frame latents. Each element is a tensor of shape [B, C, H, W].
    :param corr_map: correspondence map
    :param step: current inference step
    :param timestep: current inference timestep
    :return: A list of overlapped frame latents.
    """
    

    # Schedule
    alpha = schedule(step, timestep, 'constant', 1)  # The higher, the more overlap
    # beta = schedule(step, timestep, 'constant', 0.5)  # The higher, the closer to the base color
    # gamma = schedule(step, timestep, 'cosine', alpha_start=0.175, alpha_end=0.05)  # The strength of the extra noise
    beta, gamma = 0, 0
    logu.info(f"Alpha: {alpha} | Beta: {beta} | Gamma: {gamma} | Timestep: {timestep:.2f}")
    one_minus_alpha = 1 - alpha
    one_minus_beta = 1 - beta
    if alpha == 0:
        return frame_seq

    num_frames = len(frame_seq)
    device = frame_seq[0].device
    dtype = frame_seq[0].dtype
    map_w, map_h = corr_map.size
    batch_size, channels, frame_h, frame_w = frame_seq[0].shape
    align_corners = False if interpolate_mode in ['linear', 'bilinear', 'bicubic', 'trilinear'] else None  # Do not change

    # Upscale: (frame_w, frame_h) -> (map_w, map_h)
    up_frame_seq = [F.interpolate(frm_lat, size=(map_h, map_w), mode=interpolate_mode, align_corners=align_corners) for frm_lat in frame_seq]
    # up_frame_seq = [pipe._decode(frm_lat) for frm_lat in tqdm.tqdm(frame_seq, desc='VAE Upscale', leave=True)]
    assert up_frame_seq[0].shape[2:] == (map_h, map_w), f"frame shape {up_frame_seq[0].shape[2:]} does not match corr_map shape {(map_h, map_w)}"
    if beta > 0 and init_latents_orig_seq:
        up_proper_latents_seq = [pipe.scheduler.add_noise(pp_lat, noise, torch.tensor([timestep])) for pp_lat, noise in zip(init_latents_orig_seq, noise_seq)]
        up_proper_latents_seq = [F.interpolate(pp_lat, size=(map_h, map_w), mode=interpolate_mode, align_corners=align_corners).to(device) for pp_lat in up_proper_latents_seq]

    overlap_seq = torch.stack(up_frame_seq, dim=0)  # [T, B, C, H, W]
    mask_seq = torch.zeros((num_frames, batch_size, channels, map_h, map_w), dtype=torch.uint8, device=device)  # [T, 1, H, W]

    logu.info(f"Frame: {frame_seq[0].device} | Overlap: {overlap_seq.device} | Mask: {mask_seq.device}")

    # Progress bar
    tic = time.time()
    progress_total = len(corr_map.Map)
    progress_slice = progress_total // 100
    pbar = tqdm.tqdm(total=100, desc='Overlap', unit='%', leave=False)

    len_1_rate = 0
    for v_i, v_info in enumerate(corr_map.Map.values()):
        len_info = len(v_info)
        if len_info == 1:
            pos, t = v_info[0]
            h, w = pos
            mask_seq[t, :, :, h, w] = 1
        else:
            for i in range(len_info):
                value = 0
                count = 0
                pos_self, t_self = v_info[i]
                h_self, w_self = pos_self
                for j in range(len_info):
                    pos_other, t_other = v_info[j]
                    h_other, w_other = pos_other
                    # print(f"pos_self={pos_self}, t_self={t_self}, pos_other={pos_other}, t_other={t_other}")

                    if weight_option == 'frame_distance':
                        distance = abs(t_self - t_other)
                        # weight = 1  # Average overlapping
                        # weight = numpy.exp(-distance)  # Exponential decay
                        weight = 1 / (distance + 1)  # Linear decay
                        # weight = 1 / (distance ** 2 + 1)  # Quadratic decay
                        value += overlap_seq[t_other, :, :, h_other, w_other] * weight
                        count += weight
                    elif weight_option == 'optical_flow':
                        distance = abs(h_self - h_other) + abs(w_self - w_other)
                        weight = distance
                        value += overlap_seq[t_other, :, :, h_other, w_other] * weight
                        count += weight
                    else:
                        raise NotImplementedError

                ovlp = overlap_seq[t_self, :, :, h_self, w_self]
                if alpha > 0:  # Do overlap
                    ovlp = alpha * (value / count) + one_minus_alpha * ovlp
                if beta > 0 and init_latents_orig_seq:  # Mix base color
                    pos, t = v_info[0]
                    h_bc, w_bc = pos
                    # print(f"t={t}")
                    # print(f"shape{up_proper_latents_seq.shape}")
                    base_color = up_proper_latents_seq[t][:, :, h_bc, w_bc]  # Select the first appearance as the base color
                    ovlp = beta * base_color + one_minus_beta * ovlp

                overlap_seq[t_self, :, :, h_self, w_self] = ovlp
                mask_seq[t_self, :, :, h_self, w_self] = 1

        pbar.update(1) if v_i % progress_slice == 0 else ...

    pbar.close()

    print(len_1_rate / len(corr_map))
    overlap_seq = [ovlp for ovlp in overlap_seq]  # To list

    toc = time.time()
    logu.debug(f"Overlap cost: {toc - tic:.2f}s in total | {(toc-tic)/num_frames:.2f}s per frame")

    # Downscale: (map_w, map_h) -> (frame_w, frame_h)
    overlap_seq = [F.interpolate(ovlp, size=(frame_h, frame_w), mode=interpolate_mode, align_corners=align_corners) for ovlp in overlap_seq]
    # for i in tqdm.tqdm(range(num_frames), desc='VAE Downscale', leave=False):
    #     overlap_seq[i] = pipe._encode(overlap_seq[i])
    mask_seq = [F.interpolate(mask, size=(frame_h, frame_w), mode='nearest') for mask in mask_seq]

    logu.info(f"Mask rate: {[mask.sum() / mask.numel() for mask in mask_seq]}")

    # Add extra noise
    if noise_seq and gamma > 0:
        logu.info(f"Add extra noise with gamma={gamma}")
        latent_timestep = torch.Tensor([timestep])
        overlap_seq = [mask * (gamma * pipe.scheduler.add_noise(ovlp, noise, latent_timestep) + (1 - gamma) * ovlp) + (1 - mask) * ovlp
                    for ovlp, mask, noise in zip(overlap_seq, mask_seq, noise_seq)]

    # Mix with original frame latents (e.g. background, ...)
    # overlap_seq = [torch.where(mask != 0, ovlp, frame) for frame, ovlp, mask in zip(frame_seq, overlap_seq, mask_seq)]

    return overlap_seq


def schedule(
    step: int,
    timestep: int,
    schedule_type: str,
    alpha_start: float,
    alpha_end: float = None,
    start_step: int = 0,
    end_step: int = 1000,
    every_step: int = 1,
    start_timestep: int = 1000,
    end_timestep: int = 0,
    **kwargs,
):
    r"""
    Return parameters (e.g. alpha etc.) for overlap algorithm.
    """
    if step < start_step or step > end_step or step % every_step != 0 or timestep > start_timestep or timestep < end_timestep:
        return 0  # 0 means no overlap
    t = 1 - (timestep / 1000)  # Increase from 0 to 1
    if schedule_type == "constant":
        alpha = alpha_start
    elif schedule_type == "linear":
        alpha = alpha_start + (alpha_end - alpha_start) * t
    elif schedule_type == "cosine":
        alpha = alpha_start + (alpha_end - alpha_start) * (1 - torch.cos(t * torch.pi)) / 2
    elif schedule_type == "exponential":
        alpha = alpha_start * (alpha_end / alpha_start) ** t
    else:
        raise TypeError(f"Alpha schedule type `{schedule_type}` is not supported")

    return alpha