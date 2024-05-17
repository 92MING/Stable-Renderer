import torch
import math
import sys
import comfy.utils
from common_utils.global_utils import is_dev_mode, is_verbose_mode
from common_utils.debug_utils import ComfyUILogger


def lcm(a, b):
    if sys.version_info >= (3, 9):
        return math.lcm(a, b)
    return abs(a*b) // math.gcd(a, b)

class CONDRegular:
    def __init__(self, cond: torch.Tensor):
        self.cond = cond    # the first shape is the batch size
        if is_dev_mode() and is_verbose_mode():
            ComfyUILogger.debug(f"COND({self}) created. shape: {cond.shape}")

    def _copy_with(self, cond):
        return self.__class__(cond)

    def process_cond(self, batch_size, device, repeat=True, **kwargs):
        if not repeat:
            return self._copy_with(self.cond.clone().to(device))
        return self._copy_with(comfy.utils.repeat_to_batch_size(self.cond, batch_size).to(device))

    def can_concat(self, other):
        if self.cond.shape != other.cond.shape:
            return False
        return True

    def concat(self, others):
        conds = [self.cond]
        for x in others:
            conds.append(x.cond)
        return torch.cat(conds)


class CONDNoiseShape(CONDRegular):
    def process_cond(self, batch_size, device, area, repeat=True, **kwargs):
        data = self.cond[:,:,area[2]:area[0] + area[2],area[3]:area[1] + area[3]]
        if not repeat:
            return self._copy_with(data.to(device))
        return self._copy_with(comfy.utils.repeat_to_batch_size(data, batch_size).to(device))


class CONDCrossAttn(CONDRegular):
    def can_concat(self, other):
        s1 = self.cond.shape
        s2 = other.cond.shape
        if s1 != s2:
            if s1[0] != s2[0] or s1[2] != s2[2]: #these 2 cases should not happen
                return False

            mult_min = lcm(s1[1], s2[1])
            diff = mult_min // min(s1[1], s2[1])
            if diff > 4: #arbitrary limit on the padding because it's probably going to impact performance negatively if it's too much
                return False
        return True

    def concat(self, others):
        conds = [self.cond]
        crossattn_max_len = self.cond.shape[1]  # for prompt, it's 77
        for x in others:
            c = x.cond
            crossattn_max_len = lcm(crossattn_max_len, c.shape[1])  # if all are prompts, then it's 77
            conds.append(c)

        out = []
        for c in conds:
            if c.shape[1] < crossattn_max_len:
                c = c.repeat(1, crossattn_max_len // c.shape[1], 1) # padding with repeat doesn't change result
            out.append(c)
        
        if is_dev_mode() and is_verbose_mode():
            ComfyUILogger.debug(f"CrossAttn. All shapes = [{[x.shape for x in out]}]")
        
        return torch.cat(out) # e.g. when 1 pos, 2 neg, the shape will be [9, 77, 768] here

class CONDConstant(CONDRegular):
    def __init__(self, cond):
        self.cond = cond

    def process_cond(self, batch_size, device, **kwargs):
        return self._copy_with(self.cond)

    def can_concat(self, other):
        if self.cond != other.cond:
            return False
        return True

    def concat(self, others):
        return self.cond


__all__ = ['CONDRegular', 'CONDNoiseShape', 'CONDCrossAttn', 'CONDConstant']