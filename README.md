# Stable Renderer
<img src="docs/imgs/stable-renderer-white.png.png" style="display: block; margin-left: auto; margin-right: auto;" width="120" height="120"/>
<b><p style="text-align: center; font-size: 20px;">An AI-Rendering Engine</p></b>



## TODO

### BUGS
- `depth map`: seems only outputting last obj? (when multiple objs are rendered)
- `OBJ load`: error when :
    - loading faces with more than 3 vertices 
    - meets `g` label in .obj

### Features
- `baking/stable-renderer`
    - baking algorithms/nodes in comfyUI
    - multiple camera's output in the same frame? for multiple camera baking in the initial process? 
    - shader for mapping baked result
- `opengl/graphic APIs`
    - suppports other graphic APIs (e.g. Vulkan, DirectX)?
    - general shader obj structure system? (doing, see `shader obj region` in shader.py)
- `engine`
    - pass prompt to comfyUI during the rendering
    - scene system/ save scene/ load scene 
- `ui`
    - gameobject hierarchy on the left panel
    - scene view on the center
    - `inspect` panel on the right side?
    - dragging resources to comfyUI directly?