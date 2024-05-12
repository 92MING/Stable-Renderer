import torch
import torch.nn.functional as F

from comfyUI.types import *
from stable_rendering.src.data_classes import CorrespondenceMap

class CorrMapLatentNoiseInitializer(StableRenderingNode):
    Category = "Latent"

    @torch.no_grad()
    def __call__(self,
                 width: int,
                 height: int,
                 batch_size: int,
                 seed: INT(0, 0xffffffffffffffff),
                 correspondence_map: CorrespondenceMap,) -> LATENT:
        """Initialize both latent image and noise from correspondence map"""
        latent_generator = torch.manual_seed(seed)
        noise_generator = torch.manual_seed(seed + 1)
        corrMap_width, corrMap_height = correspondence_map.size

        latent = torch.randn([1, 4, corrMap_height, corrMap_width], device="cpu", generator=latent_generator)
        latent = latent.repeat(batch_size, 1, 1, 1)

        noise = torch.randn([1, 4, corrMap_height, corrMap_width], device="cpu", generator=noise_generator)
        noise = noise.repeat(batch_size, 1, 1, 1)
        
        for v_info in correspondence_map.Map.values():
            if len(v_info) == 1:
                continue
            position_trace, frame_index_trace = zip(*v_info)
            y_position_trace, x_position_trace = zip(*position_trace)

            latent[frame_index_trace, :, y_position_trace, x_position_trace] = torch.randn(4)
            noise[frame_index_trace, :, y_position_trace, x_position_trace] = torch.randn(4)

        latent = F.interpolate(latent, size=(height // 8, width // 8), mode="nearest")
        noise = F.interpolate(noise, size=(height // 8, width // 8), mode="nearest")

        return ({"samples":latent, "noise": noise}, )


__all__ = ["CorrMapLatentNoiseInitializer"]