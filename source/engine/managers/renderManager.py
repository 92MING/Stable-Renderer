import glm
import glfw
import torch
import ctypes
import OpenGL.GL as gl
import numpy as np

from torch import Tensor
from functools import partial
from typing import Union, Optional, Callable, TYPE_CHECKING, TypeAlias, Any
from common_utils.cuda_utils import *
from common_utils.global_utils import is_dev_mode
from common_utils.debug_utils import EngineLogger
from common_utils.stable_render_utils import CorrespondMap, Sprite, EnvPrompt
from common_utils.data_struct.event import AutoSortTask
from .manager import Manager
from .runtimeManager import RuntimeManager
from ..static.shader import Shader
from ..static.texture import Texture
from ..static.enums import *
from ..static.mesh import Mesh

if TYPE_CHECKING:
    from comfyUI.types import EngineData


def _wrapPostProcessTask(render_manager:'RenderManager', shader: "Shader", task: Optional[Callable] = None):
    '''`PostProcess` stage'''
    shader.useProgram()
    gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, 
                              gl.GL_COLOR_ATTACHMENT0, 
                              gl.GL_TEXTURE_2D, 
                              render_manager.CurrentScreenTexture, 
                              0)
    gl.glActiveTexture(gl.GL_TEXTURE0)
    gl.glBindTexture(gl.GL_TEXTURE_2D, render_manager.LastScreenTexture)
    shader.setUniform("screenTexture", 0)
    shader.setUniform("usingSD", int(not render_manager.engine.disableComfyUI))
    task() if task is not None else render_manager._draw_quad()
    render_manager.SwapScreenTexture()

def _wrapDeferRenderTask(render_manager:'RenderManager', shader: Optional["Shader"], task: Optional[Callable] = None):
    '''`DeferRender` stage'''
    shader = shader or render_manager._default_defer_render_shader
    shader.useProgram()
    shader.setUniform("usingSD", int(not render_manager.engine.disableComfyUI))
    render_manager.BindGBufferTexToShader(shader)
    
    task() if task is not None else render_manager._draw_quad()

def _wrapGBufferTask(render_manager:'RenderManager', task: Optional[Callable], shader: "Shader", mesh: Optional["Mesh"]):
    '''`GBuffer` stage (submit attribute for stable diffusion)'''
    shader.useProgram()
    render_manager.BindGBufferTexToShader(shader)
    if mesh is not None:
        if not mesh.cullback:
            gl.glDisable(gl.GL_CULL_FACE)
        else:
            gl.glEnable(gl.GL_CULL_FACE)
    if task is not None:
        task()
    elif mesh is not None:
        mesh.draw()

IdenticalGBufferDrawCallback: TypeAlias = Callable[["Texture", "Texture", "Texture", "Texture", "Texture", "Texture"], Any]
'''
The callback after identical gbuffer drawing is done. 
Parameters:
    1. color texture
    2. id texture
    3. pos texture
    4. normal texture
    5. noise texture
    6. depth texture
''' 

def _wrapIdenticalGBufferTask(render_manager:'RenderManager',
                              task: Optional[Callable],
                              shader: "Shader",
                              mesh: Optional["Mesh"], 
                              callback: Optional[IdenticalGBufferDrawCallback] = None,
                              save_to_temp=False):
    '''Each time, z-buffer will be cleared before drawing new data.'''  
    gl.glClear(gl.GL_DEPTH_BUFFER_BIT | gl.GL_COLOR_BUFFER_BIT) # type: ignore
    shader.useProgram()
    render_manager.BindGBufferTexToShader(shader)
    if mesh is not None:
        if not mesh.cullback:
            gl.glDisable(gl.GL_CULL_FACE)
        else:
            gl.glEnable(gl.GL_CULL_FACE)
    done = False
    if task is not None:
        task()
        done = True
    elif mesh is not None:
        mesh.draw()
        done = True
    if done:
        if callback is not None:
            callback(render_manager.idFBOTex, 
                    render_manager.posFBOTex, 
                    render_manager.normal_and_depth_FBOTex, 
                    render_manager.noiseFBOTex, 
                    render_manager.colorFBOTex, 
                    render_manager.depthFBOTex)
        if save_to_temp:    # save to temp buffer, these temp buffers will be write to the real buffer after all identical gbuffer tasks are done
            current_normal_and_depth = render_manager.normal_and_depth_FBOTex.tensor(update=True, flip=True)
            current_depth = current_normal_and_depth[..., -1].squeeze()
            current_normal = current_normal_and_depth[..., :-1]
            
            closer_pixels = current_depth > render_manager._depth_buffer_temp
            
            render_manager._depth_buffer_temp[closer_pixels] = current_depth[closer_pixels]
            render_manager._normal_buffer_temp[closer_pixels] = current_normal[closer_pixels]
            render_manager._color_buffer_temp[closer_pixels] = render_manager.colorFBOTex.tensor(update=True, flip=True)[closer_pixels]
            render_manager._id_buffer_temp[closer_pixels] = render_manager.idFBOTex.tensor(update=True, flip=True)[closer_pixels]
            render_manager._pos_buffer_temp[closer_pixels] = render_manager.posFBOTex.tensor(update=True, flip=True)[closer_pixels]
            render_manager._noise_buffer_temp[closer_pixels] = render_manager.noiseFBOTex.tensor(update=True, flip=True)[closer_pixels]
            
class RenderManager(Manager):
    '''Manager of all rendering stuffs'''

    FrameRunFuncOrder = RuntimeManager.FrameRunFuncOrder + 1  # always run after runtimeManager

    data_to_be_added_to_engineData: dict = {}
    '''temporary data that will be added to engineData after each frame. It will be cleared after each frame.'''
    _extra_data = {}
    '''extra data that will be passed to prompt executor'''
    _sprite_infos = {}
    '''dict for submitting to engine data on every frame.'''
    identical_gbuffer_tasks = AutoSortTask()
    '''
    Identical GBuffer is for submitting identical data to avoid overlapping.
    Each time, z-buffer will be cleared before drawing new data.
    When registering identical tasks, you can give a callback to receive the rendered data.
    '''
    gbuffer_tasks = AutoSortTask()
    '''
    GBuffer is for submitting the overall blending data, i.e. front data will be covered by back data in the same pixel.
    If you want to submit identical data to avoid overlapping, use DeferRender instead.
    '''
    defer_tasks = AutoSortTask()
    '''
    Defer tasks will be carried out after diffusion process is done.
    In this progress, you can still receive gbuffer's data(id, pos, normal, ...), and the color buffer will now be the result of diffusion. 
    '''
    post_process_tasks = AutoSortTask()
    '''
    Traditional post process tasks will be carried out after defer tasks.
    Only color framebuffer will be passed to you.
    '''
    
    def __init__(self,
                 enableHDR=True,
                 enableGammaCorrection=True,
                 gamma=2.2,
                 exposure=1.0,
                 saturation=1.0,
                 brightness=1.0,
                 contrast=1.0):
        super().__init__()
        self.engine._renderManager = self # special case, because renderManager is created before engine's assignment
        
        self._init_opengl()
        
        self._init_framebuffers()  # framebuffers for post-processing
        self._init_post_process(enableHDR=enableHDR, enableGammaCorrection=enableGammaCorrection, gamma=gamma, exposure=exposure,
                                saturation=saturation, brightness=brightness, contrast=contrast)
        self._init_quad()  # quad for post-processing
        self._init_light_buffers()
        
    def _init_opengl(self): 
        gl.glClearColor(0, 0, 0, 0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_CULL_FACE)
        gl.glDisable(gl.GL_BLEND)   # blending will be carried out in GBuffer shader
        
    def _init_framebuffers(self):
        self._default_gBuffer_shader = Shader.DefaultGBufferShader()
        '''For submit data to gBuffer'''

        self._default_defer_render_shader = Shader.DefaultDeferShader()
        '''For render gBuffer data to screen'''

        winWidth, winHeight = self.engine.WindowManager.WindowSize
        
        self.colorFBOTex = Texture(name='__COLOR_FBO__', 
                                   width=winWidth, 
                                   height=winHeight, 
                                   format=TextureFormat.RGBA,
                                   s_wrap=TextureWrap.CLAMP_TO_EDGE, 
                                   t_wrap=TextureWrap.CLAMP_TO_EDGE, 
                                   min_filter=TextureFilter.NEAREST, 
                                   mag_filter=TextureFilter.NEAREST, 
                                   internal_format=TextureInternalFormat.RGBA16F, # alpha for masking
                                   data_type=TextureDataType.HALF,
                                   share_to_torch=True)
        '''texture for saving every pixels' color in each frame'''
        self.colorFBOTex.load()
        self._color_buffer_temp = torch.zeros((winHeight, winWidth, 4), dtype=torch.float16, device='cuda') # for saving the output from identical gbuffer
        
        self.idFBOTex = Texture(name='__ID_FBO__',
                                width=winWidth,
                                height=winHeight,
                                format=TextureFormat.RGBA_INT,
                                s_wrap=TextureWrap.CLAMP_TO_EDGE,
                                t_wrap=TextureWrap.CLAMP_TO_EDGE,
                                min_filter=TextureFilter.NEAREST,
                                mag_filter=TextureFilter.NEAREST,
                                internal_format=TextureInternalFormat.RGBA_32UI,  # (spriteID, material id, uv_Xcoord, uv_Ycoord)
                                data_type=TextureDataType.SHORT,
                                share_to_torch=True)
        '''texture for saving every pixels' vertex ID in each frame'''
        self.idFBOTex.load()
        self._id_buffer_temp = torch.zeros((winHeight, winWidth, 4), dtype=torch.int16, device='cuda') # for saving the output from identical gbuffer
        
        self.posFBOTex = Texture(name='__POS_FBO__',
                                width=winWidth,
                                height=winHeight,
                                format=TextureFormat.RGB,
                                s_wrap=TextureWrap.CLAMP_TO_EDGE,
                                t_wrap=TextureWrap.CLAMP_TO_EDGE,
                                min_filter=TextureFilter.NEAREST,
                                mag_filter=TextureFilter.NEAREST,
                                internal_format=TextureInternalFormat.RGB32F,    # use 32 here since cudaGraphicsGLRegisterImage doesn't support RGB16F
                                data_type=TextureDataType.FLOAT,
                                share_to_torch=True)
        '''texture for saving every frame's position in each pixel'''
        self.posFBOTex.load()
        self._pos_buffer_temp = torch.zeros((winHeight, winWidth, 3), dtype=torch.float32, device='cuda') # for saving the output from identical gbuffer
        
        self.normal_and_depth_FBOTex = Texture(name='__NORMAL_FBO__',
                                    width=winWidth,
                                    height=winHeight,
                                    format=TextureFormat.RGBA,
                                    s_wrap=TextureWrap.CLAMP_TO_EDGE,
                                    t_wrap=TextureWrap.CLAMP_TO_EDGE,
                                    min_filter=TextureFilter.NEAREST,
                                    mag_filter=TextureFilter.NEAREST,
                                    internal_format=TextureInternalFormat.RGBA16F,    # alpha channel for saving depth
                                    data_type=TextureDataType.HALF,
                                    share_to_torch=True)
        '''texture for saving every pixel's normal in each frame'''
        self.normal_and_depth_FBOTex.load()
        self._normal_buffer_temp = torch.zeros((winHeight, winWidth, 3), dtype=torch.float16, device='cuda') # for saving the output from identical gbuffer
        self._depth_buffer_temp = torch.zeros((winHeight, winWidth), dtype=torch.float32, device='cuda') 
        # depth is not `ones` because the depth value has been 0-1 inverted in shader, closer object has larger depth value
        
        self.noiseFBOTex = Texture(name='__NOISE_FBO__',
                                   width=winWidth,
                                   height=winHeight,
                                   format=TextureFormat.RGBA,
                                   s_wrap=TextureWrap.REPEAT,
                                   t_wrap=TextureWrap.REPEAT,
                                   min_filter=TextureFilter.NEAREST,
                                   mag_filter=TextureFilter.NEAREST,
                                   internal_format=TextureInternalFormat.RGBA16F,    # 4 channel to make the shape same as a random latent
                                   data_type=TextureDataType.HALF,
                                   share_to_torch=True)
        '''Noise texture(which has the same size of model's latent) for further use in AI rendering'''
        self.noiseFBOTex.load()
        self._noise_buffer_temp = torch.zeros((winHeight, winWidth, 4), dtype=torch.float16, device='cuda') # for saving the output from identical gbuffer
        
        self.depthFBOTex = Texture(name='__DEPTH_FBO__',
                                    width=winWidth,
                                    height=winHeight,
                                    format=TextureFormat.DEPTH,
                                    share_to_torch=False)
        '''
        Warn: depth tex is just for binding to opengl's internal depth test, will not convert into tensor.
        For accessing the depth value, please use `normal_and_depth_FBOTex` instead.
        '''
        self.depthFBOTex.load()
        
        self._gBuffer = gl.glGenFramebuffers(1)
        self.BindFrameBuffer(self._gBuffer)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self.colorFBOTex.texID, 0)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT1, gl.GL_TEXTURE_2D, self.idFBOTex.texID, 0)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT2, gl.GL_TEXTURE_2D, self.posFBOTex.texID, 0)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT3, gl.GL_TEXTURE_2D, self.normal_and_depth_FBOTex.texID, 0)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT4, gl.GL_TEXTURE_2D, self.noiseFBOTex.texID, 0)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_TEXTURE_2D, self.depthFBOTex.texID, 0)
        gl.glDrawBuffers(5, [gl.GL_COLOR_ATTACHMENT0, gl.GL_COLOR_ATTACHMENT1, gl.GL_COLOR_ATTACHMENT2, gl.GL_COLOR_ATTACHMENT3, gl.GL_COLOR_ATTACHMENT4])
        
        if (gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE):
            raise Exception("G-Framebuffer is not complete! Some error occurred.")
        
        self.BindFrameBuffer(0)
        self._default_defer_render_task = partial(_wrapDeferRenderTask, self, None, None)
        
    def _init_post_process(self, enableHDR=True, enableGammaCorrection=True, gamma=2.2, exposure=1.0, saturation=1.0, brightness=1.0, contrast=1.0):
        '''
        initialize post process stuff, e.g. shader, framebuffer, etc.
        '''
        self._enableHDR = enableHDR
        self._enableGammaCorrection = enableGammaCorrection
        self._gamma = gamma
        self._exposure = exposure
        self._saturation = saturation
        self._brightness = brightness
        self._contrast = contrast

        self._default_post_process_shader = Shader.DefaultPostProcessShader()

        def final_draw():
            self._default_post_process_shader.setUniform("enableHDR", self._enableHDR)
            self._default_post_process_shader.setUniform("enableGammaCorrection", self._enableGammaCorrection)
            self._default_post_process_shader.setUniform("gamma", self._gamma)
            self._default_post_process_shader.setUniform("exposure", self._exposure)
            self._default_post_process_shader.setUniform("saturation", self._saturation)
            self._default_post_process_shader.setUniform("brightness", self._brightness)
            self._default_post_process_shader.setUniform("contrast", self._contrast)
            self.BindFrameBuffer(0) # output to screen
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT) # type: ignore
            self.DrawScreen()
        
        self._final_draw = partial(_wrapPostProcessTask, self, self._default_post_process_shader, final_draw)
        '''
        _final_draw is actually a post rendering process but it is not in the post process list.
        Default post process(hdr/gamma...) is applied here.
        '''

        self._screenTexture_1 = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._screenTexture_1)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA16F, self.engine.WindowManager.WindowSize[0],
                        self.engine.WindowManager.WindowSize[1], 0, gl.GL_RGBA, gl.GL_FLOAT, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)

        self._screenTexture_2 = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._screenTexture_2)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA16F, self.engine.WindowManager.WindowSize[0],
                        self.engine.WindowManager.WindowSize[1], 0, gl.GL_RGBA, gl.GL_FLOAT, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)

        self._currentScreenTexture = self._screenTexture_1

        self._postProcessFBO = gl.glGenFramebuffers(1)
        self.BindFrameBuffer(self._postProcessFBO)
        gl.glDrawBuffer(gl.GL_COLOR_ATTACHMENT0)
        self.BindFrameBuffer(0)
        
    def _init_light_buffers(self):
        # TODO
        self._lightShadowFBO = gl.glGenFramebuffers(1)
        self.BindFrameBuffer(self._lightShadowFBO)
        gl.glDrawBuffer(gl.GL_NONE)
        gl.glReadBuffer(gl.GL_NONE)
        self.BindFrameBuffer(0)
        
        self._lightShadowMaps = {}
        '''{(size, dimension): texture id}'''

    def _init_quad(self):
        '''
        initialize stuff about drawing the screen quad
        '''
        self._quadVertices = np.array([
            -1.0, 1.0, 0.0, 0.0, 1.0,  # Left Top
            -1.0, -1.0, 0.0, 0.0, 0.0,  # Left Bottom
            1.0, -1.0, 0.0, 1.0, 0.0,  # Right Bottom
            1.0, 1.0, 0.0, 1.0, 1.0  # Right Top
        ], dtype=np.float32)
        self._quad_indices = np.array([
            0, 1, 2,
            0, 2, 3
        ], dtype=np.uint32)

        self._quadVAO = gl.glGenVertexArrays(1)
        self._quadVBO = gl.glGenBuffers(1)
        self._quadEBO = gl.glGenBuffers(1)
        gl.glBindVertexArray(self._quadVAO)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._quadVBO)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, self._quadVertices.nbytes, self._quadVertices, gl.GL_STATIC_DRAW)
        gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, self._quadEBO)
        gl.glBufferData(gl.GL_ELEMENT_ARRAY_BUFFER, self._quad_indices.nbytes, self._quad_indices,
                        gl.GL_STATIC_DRAW)
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(0, 3, gl.GL_FLOAT, gl.GL_FALSE, 5 * 4, None)
        gl.glEnableVertexAttribArray(1)
        gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, 5 * 4, ctypes.c_void_p(3 * 4))
        gl.glBindVertexArray(0)
        
    def _draw_quad(self):
        '''draw the screen quad with the current screen texture'''
        gl.glBindVertexArray(self._quadVAO)
        gl.glDrawElements(gl.GL_TRIANGLES, 6, gl.GL_UNSIGNED_INT, None)
        gl.glBindVertexArray(0)

    def _execute_gbuffer_tasks(self, identical_buffer= False):
        if identical_buffer:
            allTasks: list[AutoSortTask.TaskWrapper] = list(self.identical_gbuffer_tasks.tempEvents + self.identical_gbuffer_tasks.events)
        else:
            allTasks: list[AutoSortTask.TaskWrapper] = list(self.gbuffer_tasks.tempEvents + self.gbuffer_tasks.events)
        allTasks.sort(key=lambda x: x.order)
        
        for task in allTasks:
            func, order = task.func, task.order
            if order < RenderOrder.TRANSPARENT.value:
                gl.glEnable(gl.GL_DEPTH_TEST)
            elif RenderOrder.TRANSPARENT.value <= order < RenderOrder.OVERLAY.value:
                gl.glDisable(gl.GL_DEPTH_TEST)
            else:  # overlay
                gl.glEnable(gl.GL_DEPTH_TEST)
            try:
                func()
            except Exception as e:
                EngineLogger.warn(f"Render Task ({order}, {func}) Error. Msg: {e}. Skipped.")
        
        if identical_buffer:
            self.identical_gbuffer_tasks._tempEvents.clear()
        else:
            self.gbuffer_tasks._tempEvents.clear()
    
    def BindGBufferTexToShader(self, shader: "Shader"):
        '''
        Bind GBuffer textures to the target shader.
        Please ensure that your shader has the following uniforms:
            - currentColor: sampler2D (vec4)
            - currentIDs: usampler2D (ivec4)
            - currentPos: sampler2D (vec3)
            - currentNormalDepth: sampler2D (vec4)
            - currentNoises: sampler2D  (vec4)
        '''
        if shader is None:
            return
        self.colorFBOTex.bind(0, 'currentColor', shader)
        self.idFBOTex.bind(1, 'currentIDs', shader)
        self.posFBOTex.bind(2, 'currentPos', shader)
        self.normal_and_depth_FBOTex.bind(3, 'currentNormalDepth', shader)
        self.noiseFBOTex.bind(4, 'currentNoises', shader)
    # endregion

    # region opengl
    @property
    def TargetDevice(self)->int:
        '''return the target GPU for running the engine'''
        return self.engine.TargetDevice
    
    @property
    def CurrentFrameBuffer(self):
        return gl.glGetIntegerv(gl.GL_FRAMEBUFFER_BINDING)

    def BindFrameBuffer(self, frameBuffer: int):
        if frameBuffer == self.CurrentFrameBuffer:
            return
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, frameBuffer)
    # endregion

    # region post process
    @property
    def EnableGammaCorrection(self):
        return self._enableGammaCorrection
    
    @EnableGammaCorrection.setter
    def EnableGammaCorrection(self, value: bool):
        self._enableGammaCorrection = value
        
    @property
    def EnableHDR(self):
        return self._enableHDR
    
    @EnableHDR.setter
    def EnableHDR(self, value: bool):
        self._enableHDR = value
    
    @property
    def Gamma(self):
        return self._gamma
    
    @Gamma.setter
    def Gamma(self, value: float):
        self._gamma = value
    
    @property
    def Exposure(self):
        return self._exposure
    
    @Exposure.setter
    def Exposure(self, value: float):
        self._exposure = value
    
    @property
    def Saturation(self):
        return self._saturation
    
    @Saturation.setter
    def Saturation(self, value: float):
        self._saturation = value
    
    @property
    def Contrast(self):
        return self._contrast
    
    @Contrast.setter
    def Contrast(self, value: float):
        self._contrast = value
    
    @property
    def Brightness(self):
        return self._brightness
    
    @Brightness.setter
    def Brightness(self, value: float):
        self._brightness = value
    
    @property
    def CurrentScreenTexture(self):
        return self._currentScreenTexture
    
    @property
    def LastScreenTexture(self):
        return self._screenTexture_1 if self._currentScreenTexture == self._screenTexture_2 else self._screenTexture_2
    
    @property
    def NextScreenTexture(self):
        '''equal to LastScreenTexture'''
        return self.LastScreenTexture
    
    def SwapScreenTexture(self):
        self._currentScreenTexture = self._screenTexture_1 if self._currentScreenTexture == self._screenTexture_2 else self._screenTexture_2
    
    def DrawScreen(self):
        self._draw_quad()
    # endregion

    # region diffusion
    @property
    def ExtraData(self):
        '''`Extra data` is the dictionary that will be passed to prompt executor. It will be clear after each execution'''
        return self._extra_data
    
    def SubmitEnvPrompt(self, prompt: EnvPrompt):
        '''set env prompt for empty areas, like background color'''
        if 'env_prompts' not in self.data_to_be_added_to_engineData:
            self.data_to_be_added_to_engineData['env_prompts'] = []
        if 'frame_indices' not in self.data_to_be_added_to_engineData:
            self.data_to_be_added_to_engineData['frame_indices'] = []
        
        if len(self.data_to_be_added_to_engineData['env_prompts']) > \
            len(self.data_to_be_added_to_engineData['frame_indices'])+1:    # only 1 env prompt for each frame is allowed
                self.data_to_be_added_to_engineData['env_prompts'][-1] = prompt
        else:
            self.data_to_be_added_to_engineData['env_prompts'].append(prompt)
    
    def SubmitCorrmap(self, spriteID:int, materialID:int, corrmap: "CorrespondMap"):
        '''add a correspond map to engine data of this frame's diffusion execution. It will be used for baking.'''
        if 'correspond_maps' not in self.data_to_be_added_to_engineData:
            self.data_to_be_added_to_engineData['correspond_maps'] = {}
        self.data_to_be_added_to_engineData['correspond_maps'][(spriteID, materialID)] = corrmap
    
    def SubmitSprite(self, sprite: "Sprite"):
        '''
        Add a sprite info to engine data of this frame's diffusion execution.
        All sprites' info will be passed to comfyUI inside the `engineData` context. It is used for text prompt masking, baking, etc. 
        Data will be cleared after each execution.
        '''
        if 'sprite_infos' not in self.data_to_be_added_to_engineData:
            self.data_to_be_added_to_engineData['sprite_infos'] = {}
        self.data_to_be_added_to_engineData['sprite_infos'][sprite.spriteID] = sprite
    # endregion
    
    # region render
    def AddIdenticalGBufferTask(self,
                                order: Union[int, float, RenderOrder],
                                task: Optional[Callable] = None,
                                shader: Optional[Shader] = None,
                                mesh: Optional[Mesh] = None,
                                callback: Optional[IdenticalGBufferDrawCallback] = None,
                                save_to_temp: bool = False):
        '''
        Add an identical GBuffer task. It can be used to generate individual instance attributes for diffusion, e.g.
        create depth map for a single object without being overlapped by others.
        
        Args:
            - order: order of the task. The smaller the order, the earlier the task will be executed.
            - task: a callable func that takes no parameter. The task should includes drawing commands. If task is None, you could just give mesh so as to use the mesh's draw.
            - shader: if shader is None, the task will use default gbuffer shader. For shader available uniform variables, plz check the default gbuffer shader's source code.
            - mesh: mesh and task can't be None at the same time. If mesh is None, the task should include those drawing commands.
            - callback: callback after the task is done. The callback will receive the rendered data.
            - save_to_temp: If save_to_temp = True, the rendered data will be saved to temp buffer, and will be write to the real buffer after all identical gbuffer tasks are done.
        '''
        order = order.value if isinstance(order, RenderOrder) else order
        if task is None and mesh is None:
            raise ValueError("task and mesh cannot be both None")
        shader = shader or self._default_gBuffer_shader
        task = partial(_wrapIdenticalGBufferTask, self, task, shader, mesh, callback, save_to_temp)
        self.identical_gbuffer_tasks.addTask(task, order)
    
    def AddGBufferTask(self, 
                      order: Union[int, float, RenderOrder], 
                      task: Optional[Callable] = None, 
                      shader: Optional[Shader] = None, 
                      mesh: Optional[Mesh] = None, 
                      forever: bool = False):
        '''
        Add a Gbuffer task to render queue. Note that the task is actually submitting data to GBuffers, not really rendering.
        Gbuffer tasks can be also treat as a normal color rendering task.
        
        Args:
            - order: order of the task. The smaller the order, the earlier the task will be executed.
            - task: a callable func that takes no parameter. The task should includes drawing commands. If task is None, you could just give mesh so as to use the mesh's draw.
            - shader: if shader is None, the task will use default gbuffer shader. For shader available uniform variables, plz check the default gbuffer shader's source code.
            - mesh: mesh and task can't be None at the same time. If mesh is None, the task should include those drawing commands.
            - forever: If forever = True, the task will be called every frame.
        '''
        order = order.value if isinstance(order, RenderOrder) else order
        if task is None and mesh is None:
            raise ValueError("task and mesh cannot be both None")
        shader = shader or self._default_gBuffer_shader
        task = partial(_wrapGBufferTask, self, task, shader, mesh)
    
        if forever:
            self.gbuffer_tasks.addForeverTask(task, order)
        else:
            self.gbuffer_tasks.addTask(task, order)

    def AddDeferRenderTask(self, task: Optional[Callable] = None, shader: Optional[Shader] = None, order: int|float = 0, forever: bool = True):
        '''
        Add a defer render task to which will be called after all normal render tasks.
        
        Args:
            - shader: shader must have a uniform named "screenTexture".
            - task: task should call DrawScreen() to draw the screen texture finally. If task is None, will just call DrawScreen() instead.
            - order: order of the task. The smaller the order, the earlier the task will be executed.
            - forever: If forever = True, the task will be called every frame.
        '''
        shader = shader or self._default_defer_render_shader
        task = partial(_wrapDeferRenderTask, self, shader, task)
        if forever:
            self.defer_tasks.addForeverTask(task, order)
        else:
            self.defer_tasks.addTask(task, order)

    def AddPostProcessTask(self, shader: Shader, task: Optional[Callable] = None, order: int|float = 0, forever: bool = True):
        '''
        Post process shader must have a uniform named "screenTexture".
        This texture return rgba color.
        '''
        wrap = partial(_wrapPostProcessTask, self, shader, task)
        if forever:
            self.post_process_tasks.addForeverTask(wrap, order)
        else:
            self.post_process_tasks.addTask(wrap, order)
    # endregion

    # region lights
    @property
    def LightShadowFBO(self):
        return self._lightShadowFBO
    
    def _createLightShadowMap(self, size, dimension):
        assert dimension in (2, 3)
        if dimension is not None:
            return  # already created
        shadowMapID = gl.glGenTextures(1)
        if dimension == 2:
            gl.glBindTexture(gl.GL_TEXTURE_2D, shadowMapID)
            gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_DEPTH_COMPONENT, size, size, 0,
                            gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT, None)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_BORDER)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_BORDER)
            gl.glTexParameterfv(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_BORDER_COLOR, glm.vec4(1.0))
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        else:
            gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, shadowMapID)
            for i in range(6):
                gl.glTexImage2D(gl.GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, gl.GL_DEPTH_COMPONENT, size, size, 0, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT, None)   # type: ignore
            gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
            gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
            gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
            gl.glTexParameteri(gl.GL_TEXTURE_CUBE_MAP, gl.GL_TEXTURE_WRAP_R, gl.GL_CLAMP_TO_EDGE)
            gl.glBindTexture(gl.GL_TEXTURE_CUBE_MAP, 0)
        return shadowMapID

    def GetLightShadowMap(self, size:int, dimension:int = 2):
        '''
        Return a shadow map texture id. If map with the same size and dimension doesn't exist, create one.
        #TODO: Light system is not yet done.
        '''
        assert dimension in (2, 3) and size > 0
        if (size, dimension) not in self._lightShadowMaps:
            self._lightShadowMaps[(size, dimension)] = self._createLightShadowMap(size, dimension)
        return self._lightShadowMaps[(size, dimension)]

    def DeleteLightShadowMap(self, mapID):
        for key, id in self._lightShadowMaps.items():
            if id == mapID:
                self._lightShadowMaps.pop(key)
                gl.glDeleteTextures(1, [id])
                break
    # endregion

    # region run
    def debug_mode_on_frame_run(self):
        '''
        This method will be ran when `debug_mode`=True, instead of `on_frame_run`.
        ComfyUI & some other complicated stuffs will be disabled in this mode. Only the basic rendering process will be executed.
        This is for quick testing the basic rendering process.
        '''
        self.BindFrameBuffer(0)
        
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT) # type: ignore
        
        self._execute_gbuffer_tasks()
    
    def _save_frame_data(self):
        if 'frame_indices' not in self.data_to_be_added_to_engineData:
            self.data_to_be_added_to_engineData['frame_indices'] = []
        self.data_to_be_added_to_engineData['frame_indices'].append(self.engine.RuntimeManager.FrameCount)
        
        if 'color_maps' not in self.data_to_be_added_to_engineData:
            self.data_to_be_added_to_engineData['color_maps'] = self.colorFBOTex.tensor(update=True, flip=True).clone()
        else:
            self.data_to_be_added_to_engineData['color_maps'] = torch.cat([self.data_to_be_added_to_engineData['color_maps'], 
                                                                            self.colorFBOTex.tensor(update=True, flip=True)], dim=0)
        
        if 'id_maps' not in self.data_to_be_added_to_engineData:
            self.data_to_be_added_to_engineData['id_maps'] = self.idFBOTex.tensor(update=True, flip=True).clone()
        else:
            self.data_to_be_added_to_engineData['id_maps'] = torch.cat([self.data_to_be_added_to_engineData['id_maps'], 
                                                                            self.idFBOTex.tensor(update=True, flip=True)], dim=0)
            
        if 'pos_maps' not in self.data_to_be_added_to_engineData:
            self.data_to_be_added_to_engineData['pos_maps'] = self.posFBOTex.tensor(update=True, flip=True).clone()
        else:
            self.data_to_be_added_to_engineData['pos_maps'] = torch.cat([self.data_to_be_added_to_engineData['pos_maps'], 
                                                                            self.posFBOTex.tensor(update=True, flip=True)], dim=0)    
        
        if 'normal_and_depth_map' not in self.data_to_be_added_to_engineData:
            self.data_to_be_added_to_engineData['normal_and_depth_map'] = self.normal_and_depth_FBOTex.tensor(update=True, flip=True).clone()
        else:
            self.data_to_be_added_to_engineData['normal_and_depth_map'] = torch.cat([self.data_to_be_added_to_engineData['normal_and_depth_map'], 
                                                                                        self.normal_and_depth_FBOTex.tensor(update=True, flip=True)], dim=0)
            
        if 'noise_maps' not in self.data_to_be_added_to_engineData:
            self.data_to_be_added_to_engineData['noise_maps'] = self.noiseFBOTex.tensor(update=True, flip=True).clone()
        else:
            self.data_to_be_added_to_engineData['noise_maps'] = torch.cat([self.data_to_be_added_to_engineData['noise_maps'], 
                                                                            self.noiseFBOTex.tensor(update=True, flip=True)], dim=0)
            
    
    def on_frame_run(self):
        self.BindFrameBuffer(self._gBuffer)
        
        # identical gbuffer tasks
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT) # type: ignore
        self._execute_gbuffer_tasks(identical_buffer=True)
        
        # write back the temp buffer to the real buffer
        self.colorFBOTex.set_data(self._color_buffer_temp.flip(0))
        self.idFBOTex.set_data(self._id_buffer_temp.flip(0))
        self.posFBOTex.set_data(self._pos_buffer_temp.flip(0))
        self.noiseFBOTex.set_data(self._noise_buffer_temp.flip(0))
        depth_data = self._depth_buffer_temp.unsqueeze(-1)
        normal_data = self._normal_buffer_temp
        normal_and_depth_data = torch.cat([normal_data, depth_data], dim=-1).flip(0)
        self.normal_and_depth_FBOTex.set_data(normal_and_depth_data)
        self.depthFBOTex.set_data(1 - depth_data.flip(0))   # depth value is inverted in shader, closer object has larger depth value
        del depth_data, normal_data, normal_and_depth_data
        
        # normal gbuffer tasks
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT) # type: ignore
        self._execute_gbuffer_tasks(identical_buffer=False)  # depth test will be enabled in this function
        
        # output map data for debugging
        if self.engine.DiffusionManager.ShouldOutputFrame:
            colorData = self.colorFBOTex.numpy_data(flipY=True)
            idData = self.idFBOTex.numpy_data(flipY=True)
            posData = self.posFBOTex.numpy_data(flipY=True)
            
            normal_and_depth_data = self.normal_and_depth_FBOTex.numpy_data(flipY=True)
            # depth data is in the alpha channel of `normal_and_depth_data`
            
            normalData = normal_and_depth_data[:, :, :3]
            depthData = normal_and_depth_data[:, :, 3]
            noiseData = self.noiseFBOTex.numpy_data(flipY=True)
            
            diffManager = self.engine.DiffusionManager
            diffManager.OutputMap('color', colorData)
            diffManager.OutputMap('normal', normalData)
            diffManager.OutputNumpyData('id', idData)
            diffManager.OutputNumpyData('pos', posData)
            diffManager.OutputNumpyData('noise', noiseData)
            diffManager.OutputDepthMap(depthData)
            if diffManager.NeedOutputCannyMap:
                diffManager.OutputCannyMap(colorData)
        
        # TODO: combiner process
        
        # refiner process
        if not self.engine.disableComfyUI:
            from comfyUI.types import EngineData
            engineData = None
            
            self._save_frame_data()          
            if self.engine.Mode != EngineMode.BAKE or \
                (self.engine.Mode == EngineMode.BAKE and self.engine.DiffusionManager.ShouldSubmitBake):   
                engineData = EngineData(**self.data_to_be_added_to_engineData)
            
            if engineData is not None:
                context = self.engine.DiffusionManager.SubmitPrompt(engineData, extra_data=self.ExtraData)
                if not context or not context.success:
                    raise ValueError("DiffusionManager.SubmitPrompt() returns None. Skip rendering process.")
                
                inference_result = context.final_output
                new_color_data: Tensor = inference_result.frame_color
                if len(new_color_data.shape) == 4:
                    new_color_data = new_color_data[0]
                new_color_data = new_color_data.flip(0)
                self.colorFBOTex.set_data(new_color_data)
                
                rgba_color_map = self.colorFBOTex.tensor(update=True, flip=True).to(torch.float32)
                rgba_color_map = rgba_color_map.flip(0)
                self.colorFBOTex.set_data(rgba_color_map)
            
                # clear data cache after diffusion submitted & done
                self._extra_data.clear()
                self.data_to_be_added_to_engineData.clear()
            
            
        # defer rendering
        gl.glDisable(gl.GL_DEPTH_TEST)
        self.BindFrameBuffer(self._postProcessFBO)  # output to post process FBO
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self.CurrentScreenTexture, 0)
        if len(self.defer_tasks) > 0:
            self.defer_tasks.execute(ignoreErr=True)  # TODO: apply light strength effect here?
        else:
            self._default_defer_render_task()

        # post process
        gl.glDisable(gl.GL_DEPTH_TEST)  # post process don't need depth test
        self.post_process_tasks.execute(ignoreErr=True)
        self._final_draw()  # default post process shader is used here. Will also bind to FBO 0(screen)
        

    def on_frame_end(self):
        glfw.swap_buffers(self.engine.WindowManager.Window)
    # endregion




__all__ = ['RenderManager']
