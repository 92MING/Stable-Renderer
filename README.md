# Stable Renderer
An engine connecting OpenGL & Stable Diffusion

----------------------------------
## Project Structure
```
.
├── resources          # 3d models, textures, shaders, etc.
├── source             # source code
│   ├── runtime        # Engine, GameObject, Component, etc.
│   │   ├── managers   # singleton managers, e.g. WindowManager, RuntimeManager, etc.
│   ├── static         # Shader, Texture, Material, etc.
│   ├── sd             # Stable Diffusion stuff
│   ├── utils          # utility functions
│   ├── main.py        # main entry
├── .gitignore
├── README.md
├── license
└── requirements.txt
```

----------------------------------
### TODO & DONE
- [ ] Light rendering / Light components
- [ ] SD manager


- [x] move depth data to color data's alpha channel
----------------------------------


