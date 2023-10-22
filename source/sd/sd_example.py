import random
import torch
import time
import os
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

    prompt = "a boat, lake, water, scenery, best quality"
    neg_prompt = "low quality, bad anatomy"

    # OPTIONAL: load a negative textual inversion model
    try:
        neg_emb_path, neg_emb_token = config.neg_emb_path, Path(config.neg_emb_path).stem
        pipe.load_textual_inversion(neg_emb_path, token=neg_emb_token)
        neg_prompt = neg_emb_token + ', ' + neg_prompt
    except Exception as e:
        logu.warn(f"Failed to load negative textual inversion model: {e}")

    width = 1080
    height = 720

    # Prepare torch generator
    seed = random.randint(0, 9999999999)  # Random seed
    generator = torch.Generator(device=config.device).manual_seed(seed)

    # 2. Prepare images
    n = 4  # Number of frames to utilize
    test_dir = config.test_dir / 'boat'

    frame_dir = test_dir / "color"
    frame_path_list = utils.list_frames(frame_dir)[:n]

    frames = [Image.open(img_path).convert('RGB') for img_path in frame_path_list]

    logu.debug(f"Color: {[Path(path).name for path in frame_path_list]}")

    # 4. Prepare control images.
    depth_dir = test_dir / "depth"
    normal_dir = test_dir / "normal"
    depth_img_paths = utils.list_frames(depth_dir)[:n]
    depth_images = [Image.open(img_path).convert('RGB') for img_path in depth_img_paths]
    normal_img_paths = utils.list_frames(normal_dir)[:n]
    normal_images = [Image.open(img_path).convert('RGB') for img_path in normal_img_paths]

    logu.debug(f"Depth: {[Path(path).name for path in depth_img_paths]}")
    logu.debug(f"Normal: {[Path(path).name for path in normal_img_paths]}")

    control_images = [[e[0], e[1]] for e in zip(depth_images, normal_images)]

    # 5. Prepare correspondence map
    corr_map = utils.make_correspondence_map(test_dir / "id", test_dir / "corr_map.pkl", force_recreate=False)
    # utils.save_corr_map_visualization(corr_map, save_dir=test_dir / "corr_map_vis", n=2)

    output_dir = test_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    latents_dir = test_dir / "latents"
    latents_dir.mkdir(parents=True, exist_ok=True)
    [os.remove(filepath) for filepath in latents_dir.glob("*")]
    assert len(os.listdir(latents_dir)) == 0, f"Latents dir {latents_dir} is not empty!"

    tic = time.time()
    output_frame_list = pipe.__call__(
        prompt=prompt,
        negative_prompt=neg_prompt,
        # correspondence_map=corr_map,
        images=frames,
        # masks=masks,  # Optional: mask images
        control_images=control_images,
        width=width,
        height=height,
        num_inference_steps=16,
        strength=0.75,
        generator=generator,
        guidance_scale=7,
        controlnet_conditioning_scale=0.5,
        add_predicted_noise=True,
        callback=utils.save_latents,
        callback_kwargs=dict(save_dir=latents_dir),
        overlap_kwargs=dict(max_workers=4)
    ).images
    toc = time.time()
    logu.info(f"Total inference time: {toc - tic:.2f}s")

    for i, image in enumerate(output_frame_list):
        image[0].save(output_dir.joinpath(f"output_{i}.png"))

    logu.success(f"Saved output frames to {output_dir}")


if __name__ == "__main__":
    main()
