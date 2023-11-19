import torch
import time
import tqdm
import numpy
from typing import List, Literal
import torch.nn.functional as F
from abc import abstractmethod, ABC
from diffusers import AutoencoderKL
from .overlap_scheduler import Scheduler
from .utils import overlap_rate
from ...data_classes.correspondenceMap import CorrespondenceMap
from ... import log_utils as logu
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


class Overlap(OverlapAlgorithm):
    r"""
    Parent class for all overlapping algorithms used in multi_frame_stable_diffusion.py
    """

    def __init__(
        self,
        scheduler: Scheduler,
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
        """
        Do overlapping on a sequence of frames according to the corresponding map.
        ONLY OVERLAP WITH SELF BUT NOT THE ORIGINAL FRAME.
        :param frame_seq: A list of frames. Each element is a tensor of shape [B, C, H, W].
        :param corr_map: A correspondence map.
        :param step: The current step.
        :param timestep: The current timestep.
        :param max_workers: The number of workers for multiprocessing.
        :return: A list of overlapped frames.
        """
        assert frame_seq[0].shape[2:] == (corr_map.height, corr_map.width), f"frame shape {frame_seq[0].shape[2:]} does not match corr_map shape {(corr_map.height, corr_map.width)}"

        alpha = self.scheduler(step, timestep)
        one_minus_alpha = 1 - alpha
        num_frames = len(frame_seq)
        batch_size, channels, frame_h, frame_w = frame_seq[0].shape
        device = frame_seq[0].device
        frame_seq_dtype = frame_seq[0].dtype
        ovlp_seq = torch.zeros((num_frames, channels, frame_h, frame_w), dtype=frame_seq_dtype, device=device)  # [C, H, W]
        # ovlp_count = torch.zeros((num_frames, frame_h, frame_w), dtype=torch.float32, device=device)  # [H, W]
        unbatched_frame_seq = [frame[0] for frame in frame_seq]

        logu.info(f"Scheduler: alpha: {alpha} | timestep: {timestep:.2f}")

        # assert ovlp_seq.device == frame_seq[0].device and ovlp_count.device == frame_seq[0].device, "ovlp_seq and ovlp_count should be on the same device as frame_seq"

        tic = time.time()
        if self.max_workers == 1:
            for v_info in corr_map.Map.values():
                # v_info = [(pos, t) for pos, t in v_info if t < num_frames]
                len_info = len(v_info)
                if len_info == 1:
                    pos, t = v_info[0]
                    h, w = pos
                    ovlp_seq[t, :, h, w] = unbatched_frame_seq[t][:, h, w]
                    # ovlp_count[t, h, w] += 1
                else:
                    value = 0
                    for pos, t in v_info:
                        h, w = pos
                        value += unbatched_frame_seq[t][:, h, w]

                    value /= len_info
                    value *= alpha

                    for pos, t in v_info:
                        h, w = pos
                        ovlp_seq[t, :, h, w] = value + one_minus_alpha * unbatched_frame_seq[t][:, h, w]
                        # ovlp_count[t, h, w] += 1

            # Overlap with self (very fast)
            # ovlp_count = ovlp_count.unsqueeze(1).expand(num_frames, channels, -1, -1)
            # ovlp_seq = torch.where(ovlp_count != 0, ovlp_seq / ovlp_count, ovlp_seq)
            ovlp_seq = ovlp_seq.unsqueeze(1)
            ovlp_seq = [ovlp_seq[i] for i in range(num_frames)]
        else:
            raise NotImplementedError(f"Multiprocessing not implemented")

        toc = time.time()
        logu.debug(f"Overlap cost: {toc - tic:.2f}s in total | {(toc-tic)/num_frames:.2f}s per frame") if self.verbose else ...

        return ovlp_seq

    def adjacent_overlap(
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
        ovlp_seq = torch.zeros((num_frames, channels, frame_h, frame_w), dtype=frame_seq_dtype, device=device)  # [C, H, W]
        unbatched_frame_seq = [frame[0] for frame in frame_seq]

        logu.info(f"Scheduler: alpha: {alpha} | timestep: {timestep:.2f}")

        tic = time.time()
        if self.max_workers == 1:
            for v_info in corr_map.Map.values():
                len_info = len(v_info)
                if len_info == 1:
                    pos, t = v_info[0]
                    h, w = pos
                    ovlp_seq[t, :, h, w] = unbatched_frame_seq[t][:, h, w]
                else:
                    for i in range(len_info):
                        value = 0
                        count = 0
                        pos_self, t_self = v_info[i]
                        h_self, w_self = pos_self
                        for j in range(len_info):
                            pos_other, t_other = v_info[j]
                            h_other, w_other = pos_other
                            distance = abs(t_self - t_other)
                            # weight = numpy.exp(-distance)  # Exponential decay
                            weight = 1 / (distance + 1)  # Linear decay
                            # weight = 1 / (distance ** 2 + 1)  # Quadratic decay
                            value += unbatched_frame_seq[t_other][:, h_other, w_other] * weight
                            count += weight
                        ovlp_seq[t_self, :, h_self, w_self] = alpha * (value / count) + one_minus_alpha * unbatched_frame_seq[t_self][:, h_self, w_self]

            ovlp_seq = ovlp_seq.unsqueeze(1) 
            ovlp_seq = list(ovlp_seq.unbind(dim=0))
        else:
            raise NotImplementedError(f"Multiprocessing not implemented")

        toc = time.time()
        logu.debug(f"Overlap cost: {toc - tic:.2f}s in total | {(toc-tic)/num_frames:.2f}s per frame") if self.verbose else ...

        return ovlp_seq

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

        overlap_seq = torch.stack(frame_seq, dim=0)  # [T, B, C, H, W]
        mask_seq = torch.zeros((num_frames, batch_size, channels, frame_h, frame_w), dtype=torch.uint8, device=device)  # [T, 1, H, W]

        logu.info(f"Scheduler: alpha: {alpha} | timestep: {timestep:.2f}")
        
        len_1_vertex_count = 0
        tic = time.time()
        if self.max_workers == 1:
            for v_info in corr_map.Map.values():
                if len(v_info) == 1:
                    # no value changes when vertex appear once only
                    len_1_vertex_count += 1
                    continue
                pos_all, t_all = zip(*v_info)
                h_all, w_all = zip(*pos_all)
                
                # compute dense frame distance from all occurance of vertex
                t_all_tensor = torch.tensor(t_all, dtype=frame_seq_dtype).to(device)
                
                # gives a covariance-like matrix but elements being abs(a_i - a_j)
                distances_matrix = torch.abs(t_all_tensor.unsqueeze(1) - t_all_tensor)

                # every row of weights * latent_seq is latent_seq weighted by 1/distance
                latent_seq = overlap_seq[t_all, :, :, h_all, w_all]
                weights = 1 / (distances_matrix + 1)
                weighted_average_latent_seq = torch.matmul(weights, latent_seq.squeeze()) / weights.sum(dim=0).reshape(-1, 1)
                weighted_average_latent_seq = weighted_average_latent_seq.reshape_as(latent_seq)
                overlap_seq[t_all, :, :, h_all, w_all] = alpha * weighted_average_latent_seq + one_minus_alpha * latent_seq
        else:
            raise NotImplementedError("Multiprocessing not implemented")
        toc = time.time()
        logu.debug(f"Overlap cost: {toc - tic:.2f}s in total | {(toc-tic)/num_frames:.2f}s per frame | Vertex appeared once: {len_1_vertex_count * 100 /len(corr_map):.2f}%") if self.verbose else ...
        

        return overlap_seq

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

        overlap_seq = torch.stack(frame_seq, dim=0)  # [T, B, C, H, W]
        mask_seq = torch.zeros((num_frames, batch_size, channels, frame_h, frame_w), dtype=torch.uint8, device=device)  # [T, 1, H, W]

        logu.info(f"Scheduler: alpha: {alpha} | timestep: {timestep:.2f}")

        tic = time.time()
        if self.max_workers == 1:
            for v_info in corr_map.Map.values():
                if len(v_info) == 1:
                    # no value changes when vertex appear once only
                    continue
                pos_all, t_all = zip(*v_info)
                h_all, w_all = zip(*pos_all)
                h_all_tensor = torch.tensor(h_all, dtype=frame_seq_dtype).to(device)
                w_all_tensor = torch.tensor(w_all, dtype=frame_seq_dtype).to(device)
                # compute vertex displacement from the previous occurance of vertex
                h_distances = torch.abs(h_all_tensor.unsqueeze(1) - h_all_tensor)
                w_distances = torch.abs(w_all_tensor.unsqueeze(1) - w_all_tensor)
                distances_matrix = h_distances + w_distances

                latent_seq = overlap_seq[t_all, :, :, h_all, w_all] 
                weights = 1 / (distances_matrix + 1)
                weighted_average_latent_seq = torch.matmul(weights, latent_seq.squeeze()) / weights.sum(dim=0).reshape(-1, 1)
                weighted_average_latent_seq = weighted_average_latent_seq.reshape_as(latent_seq)
                overlap_seq[t_all, :, :, h_all, w_all] = alpha * weighted_average_latent_seq + one_minus_alpha * latent_seq
        else:
            raise NotImplementedError(f"Multiprocessing not implemented")

        toc = time.time()
        logu.debug(f"Overlap cost: {toc - tic:.2f}s in total | {(toc - tic) / num_frames:.2f}s per frame") if self.verbose else ...

        return overlap_seq
    
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

        overlap_seq = torch.stack(frame_seq, dim=0)  # [T, B, C, H, W]
        mask_seq = torch.zeros((num_frames, batch_size, channels, frame_h, frame_w), dtype=torch.uint8, device=device)  # [T, 1, H, W]

        logu.info(f"Scheduler: alpha: {alpha} | timestep: {timestep:.2f}")

        tic = time.time()
        progress_total = len(corr_map.Map)
        progress_slice = progress_total // 100
        pbar = tqdm.tqdm(total=100, desc='Overlap', unit='%', leave=False)
        if self.max_workers == 1:
            for v_i, v_info in enumerate(corr_map.Map.values()):
                if len(v_info) == 1:
                    # no value changes when vertex appear once only
                    continue
                pos_all, t_all = zip(*v_info)
                h_all, w_all = zip(*pos_all)
                # extract vertex view normal from view normal map 
                view_normal_seq = view_normal_map[t_all, h_all, w_all].to(device)

                latent_seq = overlap_seq[t_all, :, :, h_all, w_all]
                weighted_latent_seq = latent_seq.squeeze() * view_normal_seq / view_normal_seq.sum(dim=0)
                weighted_latent_seq = weighted_latent_seq.reshape_as(latent_seq)
                overlap_seq[t_all, :, :, h_all, w_all] = alpha * weighted_latent_seq + one_minus_alpha * latent_seq
                
                pbar.update(1) if v_i % progress_slice == 0 else ...
        else:
            raise NotImplementedError(f"Multiprocessing not implemented")

        toc = time.time()
        logu.debug(f"Overlap cost: {toc - tic:.2f}s in total | {(toc - tic) / num_frames:.2f}s per frame") if self.verbose else ...

        return list(overlap_seq.unbind(dim=0))

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