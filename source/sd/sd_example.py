import random
import torch
import time
import os
import cv2
from PIL import Image
from pathlib import Path
from modules import utils, config, log_utils as logu
from modules.diffuser_pipelines.multi_frame_stable_diffusion import StableDiffusionImg2VideoPipeline
from modules.diffuser_pipelines.pipeline_utils import load_pipe


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

    prompt = "a boat on the lake, in van gogh style"
    neg_prompt = "low quality, bad anatomy"

    # OPTIONAL: load a negative textual inversion model
    # try:
    #     neg_emb_path, neg_emb_token = config.neg_emb_path, Path(config.neg_emb_path).stem
    #     pipe.load_textual_inversion(neg_emb_path, token=neg_emb_token)
    #     neg_prompt = neg_emb_token + ', ' + neg_prompt
    # except Exception as e:
    #     logu.warn(f"Failed to load negative textual inversion model: {e}")

    # Prepare torch generator
    seed = 42  # Random seed
    generator = torch.Generator(device=config.device).manual_seed(seed)

    # 2. Prepare images
    n = 8  # Number of frames to use
    test_dir = config.test_dir / '2023-10-23_3'

    frame_dir = test_dir / "color"
    frame_path_list = utils.list_frames(frame_dir)[:n]
    frames = utils.open_images(frame_path_list)

    width, height = frames[0].size

    # 3. Prepare control images.
    depth_dir = test_dir / "depth"
    normal_dir = test_dir / "normal"
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
    corr_map = utils.make_correspondence_map(test_dir / "id", test_dir / "corr_map.pkl", force_recreate=False)
    # corr_map = utils.scale_corr_map(corr_map, scale_factor=1)

    output_dir = test_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    latents_dir = test_dir / "latents"
    latents_dir.mkdir(parents=True, exist_ok=True)
    utils.clear_dir(latents_dir)

    num_inference_steps = 16

    base_image_dir = test_dir / "base_color"
    base_image_dir.mkdir(parents=True, exist_ok=True)
    base_img_path = base_image_dir.joinpath(f"base_image.png")
    if base_img_path.is_file():
        base_image = Image.open(base_img_path)
    else:
        base_image = pipe.__call__(
            prompt=prompt,
            negative_prompt=neg_prompt,
            num_frames=1,
            images=frames,
            control_images=control_images,
            width=width,
            height=height,
            num_inference_steps=32,
            strength=0.5,
            generator=generator,
            guidance_scale=7,
            controlnet_conditioning_scale=0.5,
            add_predicted_noise=True,
        ).images[0][0]

        logu.debug(f"Base image type: {type(base_image)}")
        base_image.save(base_img_path)

    # frames = utils.make_base_map(base_image, corr_map, n, base_image_dir, return_pil=True)

    # 5. Run inference
    tic = time.time()
    output_frame_list = pipe.__call__(
        prompt=prompt,
        negative_prompt=neg_prompt,
        correspondence_map=corr_map,
        images=frames,
        num_frames=n,
        # masks=masks,  # Optional: mask images
        control_images=control_images,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        strength=0.75,
        generator=generator,
        guidance_scale=7,
        controlnet_conditioning_scale=0.5,
        add_predicted_noise=True,
        callback=utils.save_latents,
        callback_kwargs=dict(save_dir=latents_dir),
        overlap_algorithm="resize_overlap",
    ).images

    # 6. Save outputs
    toc = time.time()
    utils.clear_dir(output_dir)
    for i, image in enumerate(output_frame_list):
        image[0].save(output_dir.joinpath(f"output_{i}.png"))
    utils.make_gif([image[0] for image in output_frame_list], output_dir.joinpath("output.gif"))

    logu.info(f"Inference cost: {toc - tic:.2f}s | {(toc - tic) / num_inference_steps:.2f}s/it | {(toc - tic) / (n*num_inference_steps):.2f}s/frame_it")
    logu.success(f"Saved output frames to {output_dir}")


if __name__ == "__main__":
    main()
