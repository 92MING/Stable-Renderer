# Stable Renderer
<p align="center"><img src="docs/imgs/stable-renderer-white.png.png" width="150" height="150"/></p>
<b align="center"><p style="font-size: 20px;">An AI-Rendering Engine</p></b>

## Introduction
...

## TODO

### BUGS Found
- `depth map`: seems only outputting last obj? (when multiple objs are rendered)
- `OBJ load`: error when :
    - loading faces with more than 3 vertices 
    - meets `g` label in .obj

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
    - light component
    - pass prompt to comfyUI during the rendering
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
    - lazy output type