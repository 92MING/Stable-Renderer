import os, sys
source_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(source_dir)

from sd.run_pipe import RunPipeConfig, run_pipe

config = RunPipeConfig(
        frames_dir = None,  # to get the latest frames
        
        controlnet_depth_model='lllyasviel/sd-controlnet-depth',
        controlnet_normal_model='lllyasviel/sd-controlnet-normal',
        controlnet_loose_depth_model= None,
        
        prompt='a luxury golden boat',
        # specific_timesteps=[139, 279, 499],
        num_inference_steps = 10,
        
        overlap_end_value=0.2,
        corrmap_merge_len=8,
        guidance_scale = 1.5,
        img2img_strength=0,
        kernel_radius_end_value=1,
        kernel_radius_start_value=2,
        num_frames=8, 
        
        enable_ancestral_sampling=False,
        disable_ancestral_at_last=True,
        ignore_first_overlap=False,
        
        do_controlnet=False,
        do_overlapping=False, 
        do_img2img=False,

        use_lcm_lora=False,
    )


if __name__ == '__main__':
    run_pipe(config)