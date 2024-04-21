from comfyUI.types import *
from typing import Literal

from stable_renderer.src.overlap.overlap_scheduler import Scheduler


# class StableRendererOverlapScheduler(StableRendererNodeBase):
    
#     Category = "scheduler"

#     def __call__(self,
#                  every_step: INT(1, 1000) = 1,

#                  start_step: INT(1, 1000) = 1,
#                  end_step: INT(1, 1000) = 1000,

#                  start_timestep: INT(0, 1000) = 0,
#                  end_timestep: INT(0, 1000) = 1000,

#                  interpolate_begin: FLOAT(0, 1) = 0.0,
#                  interpolate_end: FLOAT(0, 1) = 1.0,
#                  power: float = 1.0,
#                  interpolate_type: Literal["constant", "linear", "cosine", "exponential", ""] = 'constant',

#                  no_interpolate_return: FLOAT(0, 1) = 0.0,) -> Scheduler:

#         scheduler = Scheduler(
#             every_step=every_step,
#             start_step=start_step,
#             end_step=end_step,
#             start_timestep=start_timestep,
#             end_timestep=end_timestep,
#             interpolate_begin=interpolate_begin,
#             interpolate_end=interpolate_end,
#             power=power,
#             interpolate_type=interpolate_type,
#             no_interpolate_return=no_interpolate_return,
#         )
#         return scheduler

