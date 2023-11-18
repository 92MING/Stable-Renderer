import torch
from typing import List, Literal
from ...data_classes import CorrespondenceMap
from .overlap import OverlapAlgorithm
from .utils import value_interpolation
from ... import log_utils as logu

__ALL_SCHEDULER_TYPE__ = ["constant", "linear", "cosine", "exponential"]

class Scheduler:
    r"""
    Scheduler module to schedule overlap algorithms.
    Used to control the behaviour of overlap algorithms.

    Schedulers:
        - constant: Constant value
        - linear: Linearly change from start to end
        - cosine: Cosine change from start to end
    """

    def __init__(
        self,
        every_step: int = 1,

        start_step: int = 0,
        end_step: int = 1000,

        start_timestep: int = 0,
        end_timestep: int = 1000,

        alpha_start: float = 0.0,
        alpha_end: float = 1.0,
        alpha_scheduler_type: Literal["constant", "linear", "cosine", "exponential"] = 'constant',
    ):
        self._every_step = every_step

        self._start_step = start_step
        self._end_step = end_step
        self._start_timestep = start_timestep
        self._end_timestep = end_timestep

        self._alpha_start = alpha_start
        self._alpha_end = alpha_end
        self._alpha_schedule_type = alpha_scheduler_type

    def __call__(
        self,
        step: int = None,
        timestep: int = None,
        **kwargs,
    ):
        r"""
        Return parameters (e.g. alpha etc.) for overlap algorithm.
        """
        if step < self._start_step or step > self._end_step or step % self._every_step != 0 or timestep < self._start_timestep or timestep > self._end_timestep:
            return 0  # 0 means no overlap
        t = 1 - (timestep / 1000)  # Increase from 0 to 1
        
        return value_interpolation(
            t, 
            start=self._alpha_start, 
            end=self._alpha_end, 
            interpolate_function=self._alpha_schedule_type)

# To be deleted
class StartEndScheduler(OverlapAlgorithm):
    r"""
    Decorator for Overlap instances to schedule overlap behaviours
    Overlap algorithm will be activated between start_timestep and end_timestep.
    Note that timestep does not equal to inference step, it is the timestep for stable diffusion.
    The timestep for stable diffusion ranges between 0-1000, and it is the same as inference step when num_inference_steps=1000.
    """

    def __init__(self,
                 start_timestep: int,
                 end_timestep: int,
                 overlap: OverlapAlgorithm,):
        assert start_timestep >= 0 and start_timestep <= 1000 and start_timestep < end_timestep
        assert end_timestep >= 0 and end_timestep <= 1000
        # assert overlap is initialized
        self._start_timestep = start_timestep
        self._end_timestep = end_timestep
        self._overlap = overlap

    @property
    def start_timestep(self):
        return self._start_timestep

    @property
    def end_timestep(self):
        return self._end_timestep

    @property
    def overlap(self):
        return self._overlap

    def __call__(
        self,
        frame_seq: List[torch.Tensor],
        corr_map: CorrespondenceMap,
        step: int = None,
        timestep: int = None,
        **kwargs,
    ):
        if int(timestep) < self.start_timestep or int(timestep) > self.end_timestep:
            logu.info(f"Scheduling activated, overlapping at current timestep {timestep} is skipped") if self.overlap.verbose else ...
            return frame_seq
        else:
            return self.overlap(frame_seq=frame_seq, corr_map=corr_map, step=step, timestep=timestep, **kwargs)

# To be deleted
class AlphaScheduler(OverlapAlgorithm):
    r"""
    Decorator for Overlap instances to schedule overlap behaviours
    Alpha value controls the ratio between computed overlap values and original values in a diffusion step.
    The alpha value will be scheduled between alpha_start and alpha_end according to schedule_mode.
    """

    def __init__(
        self,
        alpha_start: float,
        alpha_end: float,
        schedule_mode: Literal["constant", "linear", "cosine", "exponential"],
        overlap: OverlapAlgorithm
    ):
        assert alpha_start >= 0 and alpha_start <= 1
        assert alpha_end >= 0 and alpha_end <= 1
        assert schedule_mode in ["linear"]
        self._alpha_start = alpha_start
        self._alpha_end = alpha_end
        self._schedule_mode = schedule_mode
        self._overlap = overlap

    @property
    def alpha_start(self):
        return self._alpha_start

    @property
    def alpha_end(self):
        return self._alpha_end

    @property
    def schedule_mode(self):
        return self._schedule_mode

    @property
    def overlap(self):
        return self._overlap

    def __call__(
        self,
        frame_seq: List[torch.Tensor],
        corr_map: CorrespondenceMap,
        step: int = None,
        timestep: int = None,
        **kwargs,
    ):
        t = 1 - (timestep / 1000)  # Increase from 0 to 1
        alpha = value_interpolation(t, self._alpha_start, self._alpha_end, self._schedule_mode)
        self.overlap.alpha = alpha
        logu.info(f"Scheduling activated, alpha value is set to {alpha} at current timestep {timestep}") if self.overlap.verbose else ...
        return self.overlap(frame_seq=frame_seq, corr_map=corr_map, step=step, timestep=timestep, **kwargs)

