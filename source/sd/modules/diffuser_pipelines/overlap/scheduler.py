import torch
from typing import List
from ...data_classes import CorrespondenceMap
from .overlap import OverlapAlgorithm
from ... import log_utils as logu


class StartEndScheduler(OverlapAlgorithm):
    r"""
    Decorator for Overlap instances to schedule overlap behaviours
    """
    def __init__(self,
                 start_timestep: int,
                 end_timestep: int,
                 overlap: OverlapAlgorithm,):
        assert start_timestep >= 0 and start_timestep <= 1000
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