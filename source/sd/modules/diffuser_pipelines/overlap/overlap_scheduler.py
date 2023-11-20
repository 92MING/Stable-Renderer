import torch
from typing import List, Literal
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
        if self._alpha_schedule_type == "constant":
            alpha = self._alpha_start
        elif self._alpha_schedule_type == "linear":
            alpha = self._alpha_start + (self._alpha_end - self._alpha_start) * t
        elif self._alpha_schedule_type == "cosine":
            alpha = self._alpha_start + (self._alpha_end - self._alpha_start) * (1 - torch.cos(t * torch.pi)) / 2
        elif self._alpha_schedule_type == "exponential":
            alpha = self._alpha_start * (self._alpha_end / self._alpha_start) ** t
        else:
            raise TypeError(f"Alpha schedule type `{self._alpha_schedule_type}` is not supported")

        return alpha
