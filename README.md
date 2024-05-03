# Stable Renderer
<p align="center"><img src="docs/imgs/stable-renderer-white.png.png" width="150" height="150"/></p>
<b align="center"><p style="font-size: 20px;">An AI-Rendering Engine</p></b>

## Introduction
...

## TODO

### BUGS Found
- `orbit camera`: flickering when rotation reaches certain angles. Should use quaternion to represent the rotation. 

### Potential BUGS 
- if engine can load multiple workflows at the same time in future, should add codes to prevent the duplicate of node ids.

### Features
- `stable-renderer`
    - baking algorithms/nodes in comfyUI
    - multi camera baking
    - shader for mapping baked result
- `opengl/graphic APIs`
    - suppports other graphic APIs (e.g. Vulkan, DirectX)?
    - general shader obj structure system? (doing, see `shader obj region` in shader.py)
- `engine`
    - ***stream diffusion (important)***
    - light component
    - scene system/ save scene/ load scene 
    - multiple camera's output in the same frame? for multiple camera baking in the initial process? 
- `ui`
    - gameobject hierarchy on the left panel
    - scene view on the center
    - `inspect` panel on the right side?
    - dragging resources to comfyUI directly?
- `node system`
    - shader node
    - 3d object node (3d info --> shader node --> diffuser nodes)
    - specify nodes' GPU device on workflow?


--------------------------------------------
### History
- 3-5-2024:
    - done general mesh loading by using assimp(but still some bugs in some formats)
    - fixed bugs in node system. Increased the stability/performance of the system.
    - available for 2-3 fps in img2img rendering
    - adding new overlapping/corresponding system in stable-rendering
- 28-4-2024:
    - available for passing prompt to comfyUI in engine directly
    - fix bugs in converting texture to tensor through cuda
- 25-4-2024:
    - new types in ComfyUI: hidden/ lazy
    - fixed UI returning value in advanced node system
    - available for simple stable-rendering nodes in ComfyUI
- 15-4-2024:
    - available basic system merging with ComfyUI.
    - added advanced types for ComfyUI: UI/PROMPT/...
    - available for writing back data from cuda to texture
- 8-3-2024:
    - switching to ComfyUI instead of diffusers for sampling
    - implementing advanced node system
    - available for direct transform from texture to cuda
- 6-2-2024:
    - adding more strategies for overlapping
    - fixed bugs found in map's output
    - added sample output maps for quick testing
    - try adding PYQT as an overall UI for this project
- 25-10-2023:
    - fix bugs in normal maps' output
    - available for latent overlapping algorithms
    - adding schedulers for overlapping
    - implemented stable-renderer samplers using diffusers
- 20-10-2023:
    - available for outputting maps during rendering. 
- 19-10-2023:
    - available for simple rendering
    - basic gameObj & component system