import random
import torch
import time
import os
import cv2
from PIL import Image
from pathlib import Path
from sd.modules import utils, config, log_utils as logu
from sd.modules.diffuser_pipelines.multi_frame_stable_diffusion import StableDiffusionImg2VideoPipeline
from sd.modules.diffuser_pipelines.pipeline_utils import load_pipe
from sd.modules.diffuser_pipelines.overlap import ResizeOverlap, Scheduler

TEST_DIR = config.test_dir / '2023-11-03_0'


def main():
    # 1. Load pipeline
    pipe: StableDiffusionImg2VideoPipeline = load_pipe(
        model_path=config.sd_model_path,  # Stable Diffusion model path
        control_net_model_paths=[
            "lllyasviel/sd-controlnet-depth",  # Depth model
            "lllyasviel/sd-controlnet-normal",  # Normal model
        ],
        use_safetensors=True,
        scheduler_type="euler-ancestral",
        local_files_only=True,
        device=config.device,
        torch_dtype=torch.float16,  # Float16 is much more faster on GPU
    )

    prompt = "a boat on the lake, best quality, beautiful, good lighting"
    neg_prompt = "low quality, bad anatomy"

    # Prepare torch generator
    seed = 42  # Random seed
    generator = torch.Generator(device=config.device).manual_seed(seed)

    # 2. Prepare images
    n = 10  # Number of frames to use
    start = 2  # Start frame
    end = n  # End frame

    frame_dir = TEST_DIR / "color"
    frame_path_list = utils.list_frames(frame_dir)[:n]
    frames = utils.open_images(frame_path_list)

    width, height = frames[0].size

    # 3. Prepare control images.
    depth_dir = TEST_DIR / "depth"
    normal_dir = TEST_DIR / "normal"
    depth_img_paths = utils.list_frames(depth_dir)[:n]
    depth_images = utils.open_images(depth_img_paths)
    normal_img_paths = utils.list_frames(normal_dir)[:n]
    normal_images = utils.open_images(normal_img_paths)

    logu.debug(f"Frame size: {width}x{height}")
    logu.debug(f"Color map: {[Path(path).name for path in frame_path_list]}")
    logu.debug(f"Depth map: {[Path(path).name for path in depth_img_paths]}")
    logu.debug(f"Normal map: {[Path(path).name for path in normal_img_paths]}")

    control_images = [[e[0], e[1]] for e in zip(depth_images, normal_images)]

    # 4. Prepare correspondence map
    corr_map = utils.make_correspondence_map(TEST_DIR / "id", TEST_DIR / "corr_map.pkl", force_recreate=False, num_frames=n)
    corr_map = utils.truncate_corr_map(corr_map, start=start, end=end)

    latents_dir = TEST_DIR / "latents"
    latents_dir.mkdir(parents=True, exist_ok=True)
    utils.clear_dir(latents_dir)

    # Overlapping
    overlap_scheduler = Scheduler(
        interpolate_begin=1,
        interpolate_type='constant',
    )

    overlap_algorithm = ResizeOverlap(
        scheduler=overlap_scheduler,
    )

    num_inference_steps = 16

    # 5. Run inference
    tic = time.time()
    output_frame_list = pipe.__call__(
        prompt=prompt,
        negative_prompt=neg_prompt,
        correspondence_map=corr_map,
        images=frames[start:end+1],  # Skip first two frames because of bug
        # masks=masks,  # Optional: mask images
        control_images=control_images[start:end+1],
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        strength=0.75,
        generator=generator,
        guidance_scale=7,
        controlnet_conditioning_scale=0.9,
        add_predicted_noise=True,
        callback=utils.save_latents,
        callback_kwargs=dict(save_dir=latents_dir),
        overlap_algorithm=overlap_algorithm,
        same_init_noise=True,
    ).images

    # 6. Save outputs
    toc = time.time()
    output_dir = TEST_DIR / "outputs"
    utils.clear_dir(output_dir)
    for i, image in enumerate(output_frame_list):
        image[0].save(output_dir.joinpath(f"output_{i}.png"))
    utils.make_gif([image[0] for image in output_frame_list], output_dir.joinpath("output.gif"), fps=8)

    logu.info(f"Inference cost: {toc - tic:.2f}s | {(toc - tic) / num_inference_steps:.2f}s/it | {(toc - tic) / (n*num_inference_steps):.2f}s/frame_it")
    logu.success(f"Saved output frames to {output_dir}")


if __name__ == "__main__":
    main()
