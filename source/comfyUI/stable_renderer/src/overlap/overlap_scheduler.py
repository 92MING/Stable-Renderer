from typing import Literal
from .utils import value_interpolation


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

        interpolate_begin: float = 0.0,
        interpolate_end: float = 1.0,
        power: float = 1.0,
        interpolate_type: Literal["constant", "linear", "cosine", "exponential", ""] = 'constant',

        no_interpolate_return: float = 0.0,
    ):
        r"""
        Initialize the scheduler.

        Note: 
            step is not equal to timestep.
        Timestep refers to the denoising timestep, which is ranged from 1000 (noisy) to 0 (clear).
        Step refers to the iteration step, which is consecutive and starts from 0.

        Args:
            every_step (int, optional):
                The scheduler will be activated every `every_step` steps. Default to 1.

            start_step (int, optional):
                The scheduler will be activated from `start_step`. Default to 0.

            end_step (int, optional):
                The scheduler will be activated until `end_step`. Default to 1000.

            start_timestep (int, optional):
                The scheduler will be activated from `start_timestep`. Default to 0.

            end_timestep (int, optional):
                The scheduler will be activated until `end_timestep`. Default to 1000.

            interpolate_begin (float, optional):
                The value at `start_timestep`. Default to 0.0.

            interpolate_end (float, optional):
                The value at `end_timestep`. Default to 1.0.

            power (float, optional): 
                The power of interpolation. Default to 1.0.

            interpolate_type (Literal["constant", "linear", "cosine", "exponential", ""], optional):
                The type of interpolation. Default to 'constant'.

            no_interpolate_return (float, optional)
                The value to return when the scheduler is not activated. Default to 0.0.
        """
        self._every_step = every_step

        self._start_step = start_step
        self._end_step = end_step
        self._start_timestep = start_timestep
        self._end_timestep = end_timestep

        self._interpolate_start = interpolate_begin
        self._interpolate_end = interpolate_end
        self._power = power
        self._interpolate_type = interpolate_type

        self._no_interpolate_return = no_interpolate_return

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
            return self._no_interpolate_return  # 0 means no overlap
        t = 1 - (timestep / 1000)  # Increase from 0 to 1
        
        return value_interpolation(
            t, 
            start=self._interpolate_start, 
            end=self._interpolate_end, 
            power=self._power,
            interpolate_function=self._interpolate_type)