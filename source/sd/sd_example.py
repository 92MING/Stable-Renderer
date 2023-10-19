import random
import torch
from PIL import Image
from pathlib import Path
from sys import platform
from modules import utils, config, log_utils as logu
from modules.diffuser_pipelines.multi_frame_stable_diffusion import StableDiffusionImg2VideoPipeline
from modules.diffuser_pipelines.pipeline_utils import load_pipe


def main():
    # 1. Load pipeline
    pipe: StableDiffusionImg2VideoPipeline = load_pipe(
        model_path=config.sd_model_path,  # Stable Diffusion model path
        # control_net_model_paths=[
        #     "lllyasviel/sd-controlnet-canny",  # Canny model
        #     "lllyasviel/sd-controlnet-depth"  # Depth model
        # ],
        use_safetensors=True,
        scheduler_type="euler-ancestral",
        local_files_only=True
    )
    pipe.to(config.device)
    test_dir = config.test_dir / 'boat'

    prompt = "a boat, lake, water, scenery, best quality"
    neg_prompt = "low quality, bad anatomy"

    # Optional: load a negative textual inversion model
    try:
        neg_emb_path, neg_emb_token = config.neg_emb_path, Path(config.neg_emb_path).stem
        pipe.load_textual_inversion(neg_emb_path, token=neg_emb_token)
        neg_prompt = neg_emb_token + ', ' + neg_prompt
    except Exception as e:
        logu.warn(f"[WARNING] Failed to load negative textual inversion model: {e}")

    width = 1080
    height = 720

    # Prepare torch generator
    seed = random.randint(0, 9999999999)  # Random seed
    generator = torch.Generator(device=config.device).manual_seed(seed)

    # 2. Prepare images
    n = 4  # Number of frames to utilize
    frame_dir = test_dir / "color"
    frame_path_list = utils.list_frames(frame_dir)[:n]
    logu.debug(f"[DEBUG] Frames: {[path.name for path in frame_path_list]}")
    frames = [Image.open(img_path).convert('RGB') for img_path in frame_path_list]

    # 3. Prepare masks
    # masks = [Image.open(test_dir / "masks/mask.png")]*n  # Simply use single mask for all frames for testing

    # 4. Prepare control images.
    # Simply use canny and depth images for test.
    # canny_images = utils.make_canny_images(frames)
    # depth_img_dir = test_dir / "depths/group_7"
    # depth_img_paths = utils.list_frames(depth_img_dir)[:n]
    # depth_images = [Image.open(img_path).convert('RGB') for img_path in depth_img_paths]

    # control_images = [[e[0], e[1]] for e in zip(canny_images, depth_images)]

    # 5. Prepare correspondence map
    corr_map = utils.make_correspondence_map(test_dir / "id", test_dir / "corr_map.pkl")

    output_frame_list = pipe.__call__(
        prompt=prompt,
        negative_prompt=neg_prompt,
        images=frames,
        # masks=masks,  # Optional: mask images
        # control_images=control_images,
        width=width,
        height=height,
        num_inference_steps=32,
        strength=0.4,
        generator=generator,
        guidance_scale=7,
        # controlnet_conditioning_scale=0.5,
        add_predicted_noise=True,
        callback=utils.save_latents,
        correspondence_map=corr_map,
    ).images

    output_dir = test_dir / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, image in enumerate(output_frame_list):
        image[0].save(output_dir.joinpath(f"output_{i}.png"))

    logu.success(f"[SUCCESS] Saved output frames to {output_dir}")


if __name__ == "__main__":
    main()
