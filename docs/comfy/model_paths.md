# Add Extra Model Paths

Put an `extra_model_paths.yaml` under `source/comfyUI`. It will be loaded automatically.


### For a1111 UI

all you have to do is change the base_path to where yours is installed
```yaml
    base_path: path/to/stable-diffusion-webui/

    checkpoints: models/Stable-diffusion
    configs: models/Stable-diffusion
    vae: models/VAE
    loras: |
         models/Lora
         models/LyCORIS
    upscale_models: |
                  models/ESRGAN
                  models/RealESRGAN
                  models/SwinIR
    embeddings: embeddings
    hypernetworks: models/hypernetworks
    controlnet: models/ControlNet
```

### For ComfyUI
your base path should be either an existing comfy install or a central folder where you store all of your models, loras, etc.

```yaml
     base_path: path/to/comfyui/
     checkpoints: models/checkpoints/
     clip: models/clip/
     clip_vision: models/clip_vision/
     configs: models/configs/
     controlnet: models/controlnet/
     embeddings: models/embeddings/
     loras: models/loras/
     upscale_models: models/upscale_models/
     vae: models/vae/
```

### other ui
```yaml
    base_path: path/to/ui
    checkpoints: models/checkpoints
    gligen: models/gligen
    custom_nodes: path/custom_nodes
```