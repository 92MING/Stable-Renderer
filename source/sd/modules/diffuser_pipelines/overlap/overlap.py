import time
import tqdm
import numpy

from typing import List, Literal, Tuple, Protocol
from abc import abstractmethod, ABC
from dataclasses import dataclass

from diffusers import AutoencoderKL
from .overlap_scheduler import Scheduler
from .algorithms import AverageDistance, FrameDistance, PixelDistance, PerpendicularViewNormal

from .utils import overlap_rate
from ...data_classes.correspondenceMap import CorrespondenceMap
from ...data_classes.common import Rectangle

from ... import log_utils as logu
# from ...func_utils import time_func_decorator

import torch
import torch.nn.functional as F
from torch.multiprocessing import Pool, set_start_method


class OverlapAlgorithm(ABC):
    r"""
    Interface for overlap algorithms
    """
    @abstractmethod
    def __call__(
        self,
        frame_seq: List[torch.Tensor],
        corr_map: CorrespondenceMap,
        step: int = None,
        timestep: int = None,
        **kwargs,
    ):
        pass

class OverlapCore(Protocol):
    def overlap(self,
                frame_seq: List[torch.Tensor],
                corr_map: CorrespondenceMap,
                step: int = None,
                timestep: int = None,
                **kwargs):
        ...


class FrameDistance2:
    def overlap(self,
                latent_seq: torch.Tensor,
                t_all: list,
                h_all: list,
                w_all: list) -> torch.Tensor:
        dtype, device = latent_seq.dtype, latent_seq.device
        # compute dense frame distance from all occurance of vertex
        t_all_tensor = torch.tensor(t_all, dtype=dtype).to(device)
        # gives a covariance-like matrix but elements being abs(a_i - a_j)
        distances_matrix = torch.abs(t_all_tensor.unsqueeze(1) - t_all_tensor)
        # every row of weights * latent_seq is latent_seq weighted by 1/distance
        weights = 1 / (distances_matrix + 1)
        weighted_average_latent_seq = torch.matmul(weights, latent_seq.squeeze()) / weights.sum(dim=0).reshape(-1, 1)
        return weighted_average_latent_seq.reshape_as(latent_seq)

class Average:
    def overlap(self,
                latent_seq: torch.Tensor,
                t_all: list,
                h_all: list,
                w_all: list) -> torch.Tensor:
        dtype, device = latent_seq.dtype, latent_seq.device
        # compute dense frame distance from all occurance of vertex
        t_all_tensor = torch.tensor(t_all, dtype=dtype).to(device)
        # gives a covariance-like matrix but elements being abs(a_i - a_j)
        distances_matrix = torch.abs(t_all_tensor.unsqueeze(1) - t_all_tensor)
        # every row of weights * latent_seq is latent_seq weighted by 1/distance
        # weights = torch.ones_like(distances_matrix)
        weights = torch.ones([len(t_all), len(t_all)]).to(device)
        weighted_average_latent_seq = torch.matmul(weights, latent_seq.squeeze()) / weights.sum(dim=0).reshape(-1, 1)
        return weighted_average_latent_seq.reshape_as(latent_seq)

class PixelDistance2:
    def overlap(self,
                latent_seq: torch.Tensor,
                t_all: list,
                h_all: list,
                w_all: list,) -> torch.Tensor:
        dtype, device = latent_seq.dtype, latent_seq.device
        h_all_tensor = torch.tensor(h_all, dtype=dtype).to(device)
        w_all_tensor = torch.tensor(w_all, dtype=dtype).to(device)
        # compute vertex displacement from the previous occurance of vertex
        h_distances = torch.abs(h_all_tensor.unsqueeze(1) - h_all_tensor)
        w_distances = torch.abs(w_all_tensor.unsqueeze(1) - w_all_tensor)
        distances_matrix = h_distances + w_distances
        del h_distances
        del w_distances

        weights = 1 / (distances_matrix + 1)
        weighted_average_latent_seq = torch.matmul(weights, latent_seq.squeeze()) / weights.sum(dim=0).reshape(-1, 1)
        return weighted_average_latent_seq.reshape_as(latent_seq)

class ViewNormalDistance:
    def overlap(self,
                latent_seq: torch.Tensor,
                t_all: list,
                h_all: list,
                w_all: list,
                view_normal_seq: torch.Tensor):
        dtype, device = latent_seq.dtype, latent_seq.device
        # When the view normal is closer to one, aka directly facing the camera, it should be more trustable
        distance_matrix = torch.abs(torch.ones_like(view_normal_seq).unsqueeze(1) - view_normal_seq)
        weights = 1 / (distance_matrix + 1)
        weighted_latent_seq = weights @ latent_seq.squeeze() / weights.sum(dim=0).reshape(-1, 1)
        return weighted_latent_seq.reshape_as(latent_seq)

class Overlap(OverlapAlgorithm):
    r"""
    Parent class for all overlapping algorithms used in multi_frame_stable_diffusion.py
    """

    def __init__(
        self,
        scheduler: 'Scheduler',
        weight_option: Literal['average', 'adjacent', 'optical_flow', 'frame_distance', 'view_normal'] = 'adjacent',
        max_workers: int = 1,
        verbose: bool = True
    ):
        """
        Create a functional object instance of Overlap class
        :param alpha: Ratio of computed overlap values to original latent frame values in function return
            when set to 1, the returned overlap latent sequence is composed of purely computed values
            when set to 0, the returned overlap latent sequence is composed of all original frame values
        :param max_workers: The number of workers for multiprocessing.
        :param torch_dtype: Data type of function return
        :param verbose: Enable runtime messages
        """
        assert max_workers > 0, "Number of max workers is at least 1"
        self._max_workers = max_workers
        self._verbose = verbose
        self._weight_option = weight_option

        # Module for scheduling alpha, no need to be private
        self.scheduler = scheduler

    @property
    def max_workers(self):
        return self._max_workers

    @max_workers.setter
    def max_workers(self, value: int):
        if value < 1:
            raise ValueError("Number of max workers is at least 1")
        self._max_workers = value

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, value: bool):
        self._verbose = value

    @property
    def weight_option(self):
        return self._weight_option
    
    @weight_option.setter
    def weight_option(self, value: str):
        self._weight_option = value


    @staticmethod
    def overlap_rate(ovlp_seq: List[torch.Tensor], threshold: float = 0.0):
        """
        Debug function to calculate the overlap rate of a sequence of frames.
        :param ovlp_seq: A list of overlapped frames. Note that this should haven't been overlapped with the original frames.
        """
        return overlap_rate(ovlp_seq, threshold)
    

    # @time_func_decorator("Overlap cost", ":")
    def __call__(
        self,
        frame_seq: List[torch.Tensor],
        corr_map: CorrespondenceMap,
        step: int = None,
        timestep: int = None,
        **kwargs,
    ):
        if self._weight_option == "average":
            return self.average_overlap(
                frame_seq=frame_seq,
                corr_map=corr_map,
                step=step,
                timestep=timestep,
            )
        elif self._weight_option == 'adjacent':
            return self.adjacent_overlap(
                frame_seq=frame_seq,
                corr_map=corr_map,
                step=step,
                timestep=timestep,
            )
        elif self._weight_option == 'optical_flow':
            return self.optical_flow_overlap(
                frame_seq=frame_seq,
                corr_map=corr_map,
                step=step,
                timestep=timestep,
            )
        elif self._weight_option == 'frame_distance':
            return self.frame_distance_overlap(
                frame_seq=frame_seq,
                corr_map=corr_map,
                step=step,
                timestep=timestep,
            )
        elif self._weight_option == 'view_normal':
            logu.warn("View normal overlap is buggy")
            return self.view_normal_overlap(
                frame_seq=frame_seq,
                corr_map=corr_map,
                step=step,
                timestep=timestep,
                **kwargs,
            )
        else:
            raise NotImplementedError


    def average_overlap(
            self,
            frame_seq: List[torch.Tensor],
            corr_map: CorrespondenceMap,
            step: int = None,
            timestep: int = None,
    ):
        assert frame_seq[0].shape[2:] == (corr_map.height, corr_map.width), f"frame shape {frame_seq[0].shape[2:]} does not match corr_map shape {(corr_map.height, corr_map.width)}"

        alpha = self.scheduler(step, timestep)
        one_minus_alpha = 1 - alpha
        num_frames = len(frame_seq)
        batch_size, channels, frame_h, frame_w = frame_seq[0].shape
        device = frame_seq[0].device
        frame_seq_dtype = frame_seq[0].dtype

        frame_seq_stack = torch.stack(frame_seq, dim=0)  # [T, B, C, H, W] # Usually [16, 1, 4, 512, 512]
        mask_seq = torch.zeros((num_frames, batch_size, channels, frame_h, frame_w), dtype=torch.uint8, device=device)  # [T, 1, H, W]

        logu.info(f"Scheduler: alpha: {alpha} | timestep: {timestep:.2f}")
        
        len_1_vertex_count = 0
        tic = time.time()
        progress_total = len(corr_map)
        progress_slice = progress_total // 100
        pbar = tqdm.tqdm(total=100, desc='Overlap', unit='%', leave=False)
        
        if self.max_workers == 1:
            for v_i, v_info in enumerate(corr_map.Map.values()):
                pbar.update(1) if v_i % progress_slice == 0 else ...

                if len(v_info) == 1:
                    # no value changes when vertex appear once only
                    len_1_vertex_count += 1
                    continue
                position_trace, frame_index_trace = zip(*v_info)
                y_position_trace, x_position_trace = zip(*position_trace)

                latent_seq = frame_seq_stack[frame_index_trace, :, :, y_position_trace, x_position_trace]
                overlapped_seq = AverageDistance().overlap(latent_seq, frame_index_trace, y_position_trace, x_position_trace)
                
                frame_seq_stack[frame_index_trace, :, :, y_position_trace, x_position_trace] = alpha * overlapped_seq + one_minus_alpha * latent_seq
        else:
            raise NotImplementedError("Multiprocessing not implemented")
        toc = time.time()
        logu.debug(f"Overlap cost: {toc - tic:.2f}s in total | {(toc-tic)/num_frames:.2f}s per frame | Vertex appeared once: {len_1_vertex_count * 100 / max(len(corr_map), 1) :.2f}%") if self.verbose else ...
        

        return frame_seq_stack
    

    def frame_distance_overlap(
            self,
            frame_seq: List[torch.Tensor],
            corr_map: CorrespondenceMap,
            step: int = None,
            timestep: int = None,
    ):
        assert frame_seq[0].shape[2:] == (corr_map.height, corr_map.width), f"frame shape {frame_seq[0].shape[2:]} does not match corr_map shape {(corr_map.height, corr_map.width)}"

        alpha = self.scheduler(step, timestep)
        one_minus_alpha = 1 - alpha
        num_frames = len(frame_seq)
        batch_size, channels, frame_h, frame_w = frame_seq[0].shape
        device = frame_seq[0].device
        frame_seq_dtype = frame_seq[0].dtype

        frame_seq_stack = torch.stack(frame_seq, dim=0)  # [T, B, C, H, W] # Usually [16, 1, 4, 512, 512]
        mask_seq = torch.zeros((num_frames, batch_size, channels, frame_h, frame_w), dtype=torch.uint8, device=device)  # [T, 1, H, W]

        logu.info(f"Scheduler: alpha: {alpha} | timestep: {timestep:.2f}")
        
        len_1_vertex_count = 0
        tic = time.time()
        progress_total = len(corr_map)
        progress_slice = progress_total // 100
        pbar = tqdm.tqdm(total=100, desc='Overlap', unit='%', leave=False)
        
        if self.max_workers == 1:
            for v_i, v_info in enumerate(corr_map.Map.values()):
                pbar.update(1) if v_i % progress_slice == 0 else ...

                if len(v_info) == 1:
                    # no value changes when vertex appear once only
                    len_1_vertex_count += 1
                    continue
                pos_all, t_all = zip(*v_info)
                h_all, w_all = zip(*pos_all)

                latent_seq = frame_seq_stack[t_all, :, :, h_all, w_all]
                overlapped_seq = FrameDistance().overlap(latent_seq, t_all, w_all, h_all)
                
                frame_seq_stack[t_all, :, :, h_all, w_all] = alpha * overlapped_seq + one_minus_alpha * latent_seq
        else:
            raise NotImplementedError("Multiprocessing not implemented")
        toc = time.time()
        logu.debug(f"Overlap cost: {toc - tic:.2f}s in total | {(toc-tic)/num_frames:.2f}s per frame | Vertex appeared once: {len_1_vertex_count * 100 / max(len(corr_map), 1) :.2f}%") if self.verbose else ...
        

        return frame_seq_stack

    def optical_flow_overlap(
        self,
        frame_seq: List[torch.Tensor],
        corr_map: CorrespondenceMap,
        step: int = None,
        timestep: int = None,
    ):
        assert frame_seq[0].shape[2:] == (corr_map.height, corr_map.width), f"frame shape {frame_seq[0].shape[2:]} does not match corr_map shape {(corr_map.height, corr_map.width)}"

        alpha = self.scheduler(step, timestep)
        one_minus_alpha = 1 - alpha
        num_frames = len(frame_seq)
        batch_size, channels, frame_h, frame_w = frame_seq[0].shape
        device = frame_seq[0].device
        frame_seq_dtype = frame_seq[0].dtype

        frame_seq_stack = torch.stack(frame_seq, dim=0)  # [T, B, C, H, W]
        mask_seq = torch.zeros((num_frames, batch_size, channels, frame_h, frame_w), dtype=torch.uint8, device=device)  # [T, 1, H, W]

        logu.info(f"Scheduler: alpha: {alpha} | timestep: {timestep:.2f}")

        tic = time.time()
        progress_total = len(corr_map)
        progress_slice = progress_total // 100
        pbar = tqdm.tqdm(total=100, desc='Overlap', unit='%', leave=False)
        len_1_vertex_count = 0

        if self.max_workers == 1:
            for v_i, v_info in enumerate(corr_map.Map.values()):
                pbar.update(1) if v_i % progress_slice == 0 else ...

                if len(v_info) == 1:
                    # no value changes when vertex appear once only
                    len_1_vertex_count += 1
                    continue
                pos_all, t_all = zip(*v_info)
                h_all, w_all = zip(*pos_all)

                latent_seq = frame_seq_stack[t_all, :, :, h_all, w_all] 
                overlapped_seq = PixelDistance().overlap(latent_seq, t_all, w_all, h_all)
                frame_seq_stack[t_all, :, :, h_all, w_all] = alpha * overlapped_seq + one_minus_alpha * latent_seq
        else:
            raise NotImplementedError(f"Multiprocessing not implemented")

        toc = time.time()
        logu.debug(f"Overlap cost: {toc - tic:.2f}s in total | {(toc - tic) / num_frames:.2f}s per frame | Vertex appeared once: {len_1_vertex_count / len(corr_map) * 100:.2f}%") if self.verbose else ...

        return frame_seq_stack
    
    def view_normal_overlap(
        self,
        frame_seq: List[torch.Tensor],
        corr_map: CorrespondenceMap,
        step: int = None,
        timestep: int = None,
        view_normal_map: List[torch.Tensor] = None,
    ):
        assert frame_seq[0].shape[2:] == (corr_map.height, corr_map.width), f"frame shape {frame_seq[0].shape[2:]} does not match corr_map shape {(corr_map.height, corr_map.width)}"
        assert view_normal_map is not None, "normal_map is None"
        assert view_normal_map[0].shape[:2] == (corr_map.height, corr_map.width), f"normal_map shape {view_normal_map[0].shape[:2]} does not match corr_map shape {(corr_map.height, corr_map.width)}"

        alpha = self.scheduler(step, timestep)
        one_minus_alpha = 1 - alpha
        num_frames = len(frame_seq)
        batch_size, channels, frame_h, frame_w = frame_seq[0].shape
        device = frame_seq[0].device
        frame_seq_dtype = frame_seq[0].dtype

        frame_seq_stack = torch.stack(frame_seq, dim=0)  # [T, B, C, H, W]
        mask_seq = torch.zeros((num_frames, batch_size, channels, frame_h, frame_w), dtype=torch.uint8, device=device)  # [T, 1, H, W]

        logu.info(f"Scheduler: alpha: {alpha} | timestep: {timestep:.2f}")

        tic = time.time()
        progress_total = len(corr_map)
        progress_slice = progress_total // 100
        pbar = tqdm.tqdm(total=100, desc='Overlap', unit='%', leave=False)
        len_1_vertex_count = 0

        if self.max_workers == 1:
            for v_i, v_info in enumerate(corr_map.Map.values()):
                pbar.update(1) if v_i % progress_slice == 0 else ...

                if len(v_info) == 1:
                    # no value changes when vertex appear once only
                    len_1_vertex_count += 1
                    continue
                pos_all, t_all = zip(*v_info)
                h_all, w_all = zip(*pos_all)
                # extract vertex view normal from view normal map 
                view_normal_seq = view_normal_map[t_all, h_all, w_all].squeeze().to(device)

                latent_seq = frame_seq_stack[t_all, :, :, h_all, w_all]
                overlapped_seq = PerpendicularViewNormal().overlap(latent_seq, t_all, h_all, w_all, view_normal_seq)
                frame_seq_stack[t_all, :, :, h_all, w_all] = alpha * overlapped_seq + one_minus_alpha * latent_seq
                
        else:
            raise NotImplementedError(f"Multiprocessing not implemented")

        toc = time.time()
        logu.debug(f"Overlap cost: {toc - tic:.2f}s in total | {(toc - tic) / num_frames:.2f}s per frame | Vertex appeared once: {len_1_vertex_count / len(corr_map) * 100:.2f}%") if self.verbose else ...

        return list(frame_seq_stack.unbind(dim=0))

class ResizeOverlap(Overlap):
    def __init__(
        self,
        scheduler: Scheduler,
        weight_option: str = 'adjacent',
        max_workers: int = 1,
        verbose: bool = True,
        interpolate_mode: str = 'nearest'
    ):
        super().__init__(
            scheduler=scheduler,
            weight_option=weight_option,
            max_workers=max_workers,
            verbose=verbose,
        )
        self._interpolate_mode = interpolate_mode

    @property
    def interpolate_mode(self):
        return self._interpolate_mode

    @interpolate_mode.setter
    def interpolate_mode(self, value: str):
        self._interpolate_mode = value

    def __call__(
        self,
        frame_seq: List[torch.Tensor],
        corr_map: CorrespondenceMap,
        step: int = None,
        timestep: int = None,
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
        alpha = self.scheduler(step, timestep)
        if alpha == 0:
            return frame_seq
        num_frames = len(frame_seq)
        screen_w, screen_h = corr_map.size
        frame_h, frame_w = frame_seq[0].shape[-2:]
        align_corners = False if self.interpolate_mode in ['linear', 'bilinear', 'bicubic', 'trilinear'] else None

        ovlp_seq = [F.interpolate(latents, size=(screen_h, screen_w), mode=self.interpolate_mode, align_corners=align_corners) for latents in frame_seq]
        ovlp_seq = super().__call__(
            ovlp_seq,
            corr_map=corr_map,
            step=step,
            timestep=timestep,
            **kwargs
        )
        ovlp_seq = [F.interpolate(latents, size=(frame_h, frame_w), mode=self.interpolate_mode, align_corners=align_corners) for latents in ovlp_seq]

        logu.debug(f"Resize scale factor: {screen_h / frame_h:.2f} | Overlap ratio: {100 * self.overlap_rate(ovlp_seq, threshold=0):.2f}%")

        # Overlap with original
        ovlp_seq = [torch.where(ovlp_seq[i] != 0, ovlp_seq[i], frame_seq[i]) for i in range(num_frames)]
        return ovlp_seq


class VAEOverlap(Overlap):
    def __init__(
        self,
        vae: AutoencoderKL,
        generator: torch.Generator,
        scheduler: Scheduler,
        max_workers: int = 1,
        verbose: bool = True,
    ):
        super().__init__(
            scheduler=scheduler,
            max_workers=max_workers,
            verbose=verbose,
        )
        self._vae = vae
        self._generator = generator

    @property
    def vae(self):
        return self._vae

    @property
    def generator(self):
        return self._generator

    def _encode(self, image):
        latent_dist = self.vae.encode(image).latent_dist
        latents = latent_dist.sample(generator=self.generator)
        latents = self.vae.config.scaling_factor * latents
        return latents

    def _decode(self, latents):
        latents = 1 / self.vae.config.scaling_factor * latents
        image = self.vae.decode(latents).sample
        return image

    # TODO: Optimize this function
    def __call__(
        self,
        frame_seq: List[torch.Tensor],
        corr_map: CorrespondenceMap,
        step: int = None,
        timestep: int = None,
        **kwargs,
    ):
        """
        Do overlapping with VAE decode/encode a latents to pixel space.
        :param latents_seq: A sequence of latents.
        :param corr_map: A correspondence map.
        :param step: The current step.
        :param timestep: The current timestep.
        :return: A sequence of overlapped latents.
        """

        num_frames = len(frame_seq)
        screen_w, screen_h = corr_map.size
        frame_h, frame_w = frame_seq[0].shape[-2:]

        #! IMPORTANT:
        #! (1) After VAE encoding, 0's in the latents will not be 0's anymore
        #! (2) After VAE decoding, the number of channels will change from 4 to 3. Encoding will do reverse.
        #! (3) Do not overlap original in pixel space and then encode to latent space. This will destroy generation.
        # TODO: However, if we can overlap original at latent space directly, then the destruction might be much less.

        pix_seq = [self._decode(latents) for latents in frame_seq]
        ovlp_seq = super().__call__(
            pix_seq,
            corr_map=corr_map,
            step=step,
            timestep=timestep,
            **kwargs,
        )
        ovlp_seq = [torch.where(ovlp_seq[i] != 0, ovlp_seq[i], pix_seq[i]) for i in range(num_frames)]  # Overlap with original
        ovlp_seq = [self._encode(ovlp_img) for ovlp_img in ovlp_seq]

        # ovlp_seq = [torch.where(abs(ovlp_seq[i] - latents_seq[i]) > 0.1, ovlp_seq[i], latents_seq[i]) for i in range(num_frames)]

        return ovlp_seq