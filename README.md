# Stable Renderer
An engine connecting OpenGL & Stable Diffusion

----------------------------------
## Project Structure
```
.
├── resources          # 3d models, textures, shaders, etc.
├── source             # source code
│   ├── runtime        # Engine, GameObject, Component, etc.
│   ├── static         # Shader, Texture, Material, etc.
│   ├── sd             # Stable Diffusion stuff
│   ├── utils          # utility functions
│   ├── scripts        # scripts for different testing purposes
│   └── .env           # environment variables. You should create this yourself
```
----------------------------------
## Engine

#### Shader
 ##### uniform blocks
   ###### Matrices & common data
   ```
     layout (std140) uniform Matrices {
        mat4 model;
        mat4 view;
        mat4 projection;
        mat4 MVP;
        mat4 MVP_IT; // inverse transpose of MVP
        mat4 MV; // model-view matrix
        mat4 MV_IT; // inverse transpose of MV
        vec3 cameraPos;
        vec3 cameraDir;
     };
   ```
   ###### Engine info
   ```
     layout (std140) uniform Engine {
        ivec2 windowSize;
     };
   ```

----------------------------------


