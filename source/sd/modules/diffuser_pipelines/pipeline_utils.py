import os
import torch
import time
from sys import platform
from typing import Sequence

from diffusers import ControlNetModel
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_zero import CrossFrameAttnProcessor
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel

from .multi_frame_stable_diffusion import StableDiffusionImg2VideoPipeline
from .. import log_utils as logu
from utils.global_utils import GetEnv

def load_pipe(
    model_path: str = GetEnv('SD_PATH', 'runwayml/stable-diffusion-v1-5'),
    control_net_model_paths: Sequence[str] = (GetEnv('CONTROLNET_DEPTH_MODEL','lllyasviel/sd-controlnet-depth'),
                                              GetEnv('CONTROLNET_NORMAL_MODEL','lllyasviel/sd-controlnet-normal'), ),
    use_safetensors: bool = True,
    scheduler_type: str = "euler-ancestral",
    no_half: bool = False,
    device=GetEnv('DEVICE','cpu'),
    torch_dtype=torch.float16,
    local_files_only: bool = False,
) -> StableDiffusionImg2VideoPipeline:
    """
    Load a Stable Diffusion pipeline with some controlnets.
    """
    logu.info(f"Loading Stable Diffusion pipeline from {model_path} with controlnets {control_net_model_paths}.")
    tic = time.time()

    no_half = no_half or (platform == 'darwin')
    torch_dtype = torch.float32 if no_half else torch.float16

    control_net_model_paths = [str(path) for path in control_net_model_paths]

    if len(control_net_model_paths) == 0:
        controlnet = None
    elif len(control_net_model_paths) == 1:
        controlnet = ControlNetModel.from_pretrained(
            control_net_model_paths[0],
            torch_dtype=torch_dtype
        )
        controlnet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))
    else:
        controlnets = []
        for control_net_model_path in control_net_model_paths:
            controlnet = None
            try:
                controlnet = ControlNetModel.from_pretrained(
                    control_net_model_path,
                    torch_dtype=torch_dtype
                )
            except Exception as e:
                controlnet = ControlNetModel.from_pretrained(
                    control_net_model_path,
                    torch_dtype=torch_dtype,
                    use_safetensors=True
                )
            logu.info(f"Loaded controlnet from {control_net_model_path}.")
            controlnet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))
            controlnets.append(controlnet)
        controlnet = MultiControlNetModel(controlnets)

    if local_files_only or os.path.isfile(model_path):
        if os.path.exists(model_path):
            model_path = os.path.abspath(model_path)
        assert os.path.isfile(model_path), f"Model path {os.path.abspath(model_path)} is not a file."
        pipe = StableDiffusionImg2VideoPipeline.from_single_file(
            model_path,
            local_files_only=True,
            controlnet=controlnet,
            torch_dtype=torch_dtype,
            use_safetensors=use_safetensors,
            scheduler_type=scheduler_type,
        )
    else:
        pipe: StableDiffusionImg2VideoPipeline = StableDiffusionImg2VideoPipeline.from_pretrained(
            model_path,
            controlnet=controlnet,
            torch_dtype=torch_dtype,
            use_safetensors=use_safetensors,
            scheduler_type=scheduler_type,
        )
    pipe.unet.set_attn_processor(CrossFrameAttnProcessor(batch_size=2))
    pipe.safety_checker = None
    pipe.to(device)

    # Enable XFormers
    try:
        pipe.enable_xformers_memory_efficient_attention(attention_op=None)
    except ModuleNotFoundError:
        logu.warn("XFormers not found. You can install XFormers with `pip install xformers` to speed up the generation.")
    # pipe.enable_model_cpu_offload()

    toc = time.time()
    logu.success(f"Pipe loaded in {toc-tic:.2f}s | device: {device} | dtype: {torch_dtype}.")

    return pipe
