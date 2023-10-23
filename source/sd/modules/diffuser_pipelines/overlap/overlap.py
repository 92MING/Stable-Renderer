import torch
import time
from typing import List
import torch.nn.functional as F
from diffusers import AutoencoderKL
from .utils import overlap_rate
from ...data_classes.correspondenceMap import CorrespondenceMap
from ... import log_utils as logu


class Overlap:
    r"""
    Parent class for all overlapping algorithms used in multi_frame_stable_diffusion.py
    """
    def __init__(
        self,
        alpha: float = 0.5,
        max_workers: int = 1,
        verbose: bool = True):
        """
        Create a functional object instance of Overlap class
        :param alpha: Ratio of computed overlap values to original latent frame values in function return
            when set to 1, the returned overlap latent sequence is composed of purely computed values
            when set to 0, the returned overlap latent sequence is composed of all original frame values
        :param max_workers: The number of workers for multiprocessing.
        :param torch_dtype: Data type of function return
        :param verbose: Enable runtime messages
        """
        assert alpha >= 0 and alpha <= 1, "Value of alpha should lie between 0 to 1"
        assert max_workers > 0, "Number of max workers is at least 1"
        self._alpha = alpha
        self._max_workers = max_workers
        self._verbose = verbose
    
    @property
    def alpha(self):
        return self._alpha
    @alpha.setter
    def alpha(self, value: float):
        if value < 0 or value > 1:
            raise ValueError(f"Value of alpha should lie between 0 to 1")
        self._alpha = value
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
        overlap_kwargs: dict = {},
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

        try:
            start_corr = overlap_kwargs.get('start_corr')
            end_corr = overlap_kwargs.get('end_corr')
            if int(timestep) < start_corr or int(timestep) > end_corr:
                logu.info(f"Scheduling activated, current timestep {timestep}") if self.verbose else ...
                return frame_seq
        except Exception as e:
            logu.error(e)

        num_frames = len(frame_seq)
        batch_size, channels, frame_h, frame_w = frame_seq[0].shape
        device = frame_seq[0].device
        frame_seq_dtype = frame_seq[0].dtype
        ovlp_seq = torch.zeros((num_frames, channels, frame_h, frame_w), dtype=frame_seq_dtype, device=device)  # [C, H, W]
        ovlp_count = torch.zeros((num_frames, frame_h, frame_w), dtype=torch.uint8, device=device)  # [H, W]

        # assert ovlp_seq.device == frame_seq[0].device and ovlp_count.device == frame_seq[0].device, "ovlp_seq and ovlp_count should be on the same device as frame_seq"

        tic = time.time()
        if self.max_workers == 1:
            for v_info in corr_map.Map.values():
                v_info = [(pos, t) for pos, t in v_info if t < num_frames]
                if len(v_info) == 0:
                    continue
                elif len(v_info) == 1:
                    pos, t = v_info[0]
                    h, w = pos
                    ovlp_seq[t, :, h, w] += frame_seq[t][0, :, h, w]
                    ovlp_count[t, h, w] += 1
                    continue
                value = 0
                count = 0
                for pos, t in v_info:
                    h, w = pos
                    value += frame_seq[t][0, :, h, w]
                    count += 1

                value /= count

                for pos, t in v_info:
                    h, w = pos
                    ovlp_seq[t, :, h, w] += self.alpha * value + (1 - self.alpha) * frame_seq[t][0, :, h, w]
                    ovlp_count[t, h, w] += 1

            # Overlap with self (very fast)
            ovlp_count = ovlp_count.unsqueeze(1).expand(num_frames, channels, -1, -1)
            ovlp_seq = torch.where(ovlp_count != 0, ovlp_seq / ovlp_count, ovlp_seq)
            ovlp_seq = ovlp_seq.unsqueeze(1)
            ovlp_seq = [ovlp_seq[i] for i in range(num_frames)]
        else:
            raise NotImplementedError(f"Multiprocessing not implemented")

        toc = time.time()
        logu.debug(f"Overlap cost: {toc - tic:.2f}s in total | {(toc-tic)/num_frames:.2f}s per frame") if self.verbose else ...

        return ovlp_seq

class ResizeOverlap(Overlap):
    def __init__(
        self,
        alpha: float = 0.5,
        max_workers: int = 1,
        verbose: bool = True,
        interpolate_mode: str = 'bilinear'
    ):
        super().__init__(alpha, max_workers, verbose)
        self._interpolate_mode = interpolate_mode
    
    @property
    def interpolate_mode(self):
        return self._interpolate_mode
    @interpolate_mode.setter
    def interpolate_mode(self, value: str):
        self._interpolate_mode = value
    
    def __call__(self,
        latents_seq: List[torch.Tensor],
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
        num_frames = len(latents_seq)
        screen_w, screen_h = corr_map.size
        frame_h, frame_w = latents_seq[0].shape[-2:]
        align_corners = False if self.interpolate_mode in ['linear', 'bilinear', 'bicubic', 'trilinear'] else ...

        ovlp_seq = [F.interpolate(latents, size=(screen_h, screen_w), mode=self.interpolate_mode, align_corners=align_corners) for latents in latents_seq]
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
        ovlp_seq = [torch.where(ovlp_seq[i] != 0, ovlp_seq[i], latents_seq[i]) for i in range(num_frames)]
        return ovlp_seq

class VAEOverlap(Overlap):
    def __init__(
        self,
        vae: AutoencoderKL,
        generator: torch.Generator,
        alpha: float = 0.5,
        max_workers: int = 1,
        verbose: bool = True,
    ):
        super().__init__(alpha, max_workers, verbose)
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
        latents_seq: List[torch.Tensor],
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

        num_frames = len(latents_seq)
        screen_w, screen_h = corr_map.size
        frame_h, frame_w = latents_seq[0].shape[-2:]

        #! IMPORTANT:
        #! (1) After VAE encoding, 0's in the latents will not be 0's anymore
        #! (2) After VAE decoding, the number of channels will change from 4 to 3. Encoding will do reverse.
        #! (3) Do not overlap original in pixel space and then encode to latent space. This will destroy generation.
        # TODO: However, if we can overlap original at latent space directly, then the destruction might be much less.

        pix_seq = [self._decode(latents) for latents in latents_seq]
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