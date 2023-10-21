import os
import torch
from sys import platform
from typing import List
from diffusers import ControlNetModel
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from .multi_frame_stable_diffusion import StableDiffusionImg2VideoPipeline
from .. import log_utils as logu


def load_pipe(
    model_path: str = "runwayml/stable-diffusion-v1-5",
    control_net_model_paths: List[str] = ["lllyasviel/sd-controlnet-depth"],
    use_safetensors: bool = True,
    scheduler_type: str = "euler-ancestral",
    no_half: bool = False,
    device='cpu',
    torch_dtype=torch.float16,
    local_files_only: bool = False,
) -> StableDiffusionImg2VideoPipeline:
    """
    Load a Stable Diffusion pipeline with some controlnets.
    """
    logu.info(f"Loading Stable Diffusion pipeline from {model_path} with controlnets {control_net_model_paths}.")
    no_half = no_half or (platform == 'darwin')
    torch_dtype = torch.float32 if no_half else torch.float16

    if isinstance(control_net_model_paths, str):
        control_net_model_paths = [control_net_model_paths]

    control_net_model_paths = [str(path) for path in control_net_model_paths]

    if len(control_net_model_paths) == 0:
        controlnet = None
    elif len(control_net_model_paths) == 1:
        controlnet = ControlNetModel.from_pretrained(
            control_net_model_paths[0],
            torch_dtype=torch_dtype
        )
    else:
        controlnets = []
        for control_net_model_path in control_net_model_paths:
            controlnet = ControlNetModel.from_pretrained(
                control_net_model_path,
                torch_dtype=torch_dtype
            )
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

    pipe.safety_checker = None
    pipe.to(device)

    # Enable XFormers
    try:
        pipe.enable_xformers_memory_efficient_attention(attention_op=None)
    except ModuleNotFoundError:
        logu.warn("XFormers not found. You can install XFormers with `pip install xformers` to speed up the generation.")
    # pipe.enable_model_cpu_offload()

    logu.success(f"Pipe loaded on {device}, dtype: {torch_dtype}.")

    return pipe
