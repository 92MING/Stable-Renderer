import random
import torch
from PIL import Image
from pathlib import Path
from diffuser_pipeline.multi_frame_stable_diffusion import (
    StableDiffusionImg2VideoPipeline,
    load_pipe,
)
from modules import utils, config


def main():
    # 1. Load pipeline
    pipe: StableDiffusionImg2VideoPipeline = load_pipe(
        model_path=config.sd_model_path,  # Stable Diffusion model path
        control_net_model_paths=[
            "lllyasviel/sd-controlnet-canny",  # Canny model
            "lllyasviel/sd-controlnet-depth"  # Depth model
        ],
        use_safetensors=True,
        scheduler_type="euler-ancestral"
    )
    pipe.to(config.device)

    prompt = "best quality, masterpiece, 1boy, solo, male focus, highres"
    neg_prompt = "low quality, bad anatomy"

    # Optional: load a negative textual inversion model
    try:
        neg_emb_path, neg_emb_token = config.neg_emb_path, Path(config.neg_emb_path).stem
        pipe.load_textual_inversion(neg_emb_path, token=neg_emb_token)
        neg_prompt = neg_emb_token + ', ' + neg_prompt
    except Exception as e:
        print(f"Failed to load negative textual inversion model: {e}")

    width = 848
    height = 480

    # Prepare torch generator
    seed = random.randint(0, 9999999999)  # Random seed
    generator = torch.Generator(device=config.device).manual_seed(seed)

    # 2. Prepare images
    n = 8  # Number of frames to utilize
    frame_dir = Path("test/groups/group_7")
    frame_path_list = utils.list_frames(frame_dir)[:n]
    frames = [Image.open(img_path).convert('RGB') for img_path in frame_path_list]

    # 3. Prepare masks
    masks = [Image.open("test/masks/mask.png")]*n  # Simply use single mask for all frames for testing

    # 4. Prepare control images.
    # Simply use canny and depth images for test.
    canny_images = utils.make_canny_images(frames)
    depth_img_dir = Path('test/depths/group_7')
    depth_img_paths = utils.list_frames(depth_img_dir)[:n]
    depth_images = [Image.open(img_path).convert('RGB') for img_path in depth_img_paths]

    control_images = [[e[0], e[1]] for e in zip(canny_images, depth_images)]

    output_frame_list = pipe.__call__(
        prompt=prompt,
        negative_prompt=neg_prompt,
        images=frames,
        masks=masks,  # Optional: mask images
        control_images=control_images,
        width=width,
        height=height,
        num_inference_steps=32,
        strength=0.8,
        generator=generator,
        guidance_scale=7,
        controlnet_conditioning_scale=0.5,
        add_predicted_noise=True,
        # callback=utils.view_latents,
    ).images

    output_dir = Path("test/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, image in enumerate(output_frame_list):
        image[0].save(output_dir.joinpath(f"output_frame_{i}.png"))


if __name__ == "__main__":
    main()
