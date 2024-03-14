import glm
import glfw
import ctypes
import OpenGL.GL as gl
import numpy as np
import pycuda.driver as cuda_driver

from functools import partial
from typing import Union, Optional, Callable
from utils.cuda_utils import *
from utils.data_struct.event import AutoSortTask
from .manager import Manager
from .runtimeManager import RuntimeManager
from ..static.shader import Shader
from ..static.texture import Texture
from ..static.enums import *
from ..static.mesh import Mesh


def _bindGTexture(shader: Shader, slot: int, textureID: int, name: str):
    gl.glActiveTexture(gl.GL_TEXTURE0 + slot)  # type: ignore
    gl.glBindTexture(gl.GL_TEXTURE_2D, textureID)
    shader.setUniform(name, slot)

def _wrapPostProcessTask(render_manager:'RenderManager', shader, task):
    shader.useProgram()
    gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, 
                              gl.GL_COLOR_ATTACHMENT0, 
                              gl.GL_TEXTURE_2D, 
                              render_manager.CurrentScreenTexture, 
                              0)
    gl.glActiveTexture(gl.GL_TEXTURE0)
    gl.glBindTexture(gl.GL_TEXTURE_2D, render_manager.LastScreenTexture)
    shader.setUniform("screenTexture", 0)
    task() if task is not None else render_manager._draw_quad()
    render_manager.SwapScreenTexture()

def _wrapDeferRenderTask(render_manager:'RenderManager', shader, task):
    shader.useProgram()
    _bindGTexture(shader, 0, render_manager.colorFBOTex.textureID, "gColor")  # type: ignore
    _bindGTexture(shader, 1, render_manager.idFBOTex.textureID, "gID")  # type: ignore
    _bindGTexture(shader, 2, render_manager.posFBOTex.textureID, "gPos")  # type: ignore
    _bindGTexture(shader, 3, render_manager.normal_and_depth_FBOTex.textureID, "gNormal")  # type: ignore
    _bindGTexture(shader, 4, render_manager.noiseFBOTex.textureID, "gNoise")  # type: ignore
    task() if task is not None else render_manager._draw_quad()

def _wrapRenderTask(task, shader, mesh):
    shader.useProgram()
    if mesh is not None:
        shader.setUniform("objID", mesh.meshID)
    task() if task is not None else mesh.draw()

class RenderManager(Manager):
    '''Manager of all rendering stuffs'''

    FrameRunFuncOrder = RuntimeManager.FrameRunFuncOrder + 1  # always run after runtimeManager

    def __init__(self,
                 enableHDR=True,
                 enableGammaCorrection=True,
                 gamma=2.2,
                 exposure=1.0,
                 saturation=1.0,
                 brightness=1.0,
                 contrast=1.0,
                 target_device: Optional[int] = None):
        super().__init__()
        self.engine._renderManager = self # special case, because renderManager is created before engine's assignment
        self._renderTasks = AutoSortTask()
        self._deferRenderTasks = AutoSortTask()
        self._postProcessTasks = AutoSortTask()
        
        if not target_device:
            target_device = get_cuda_device()
        self._target_device = target_device
        self._init_cuda()
        self._init_opengl()
        
        self._init_framebuffers()  # framebuffers for post-processing
        self._init_post_process(enableHDR=enableHDR, enableGammaCorrection=enableGammaCorrection, gamma=gamma, exposure=exposure,
                                saturation=saturation, brightness=brightness, contrast=contrast)
        self._init_quad()  # quad for post-processing
        self._init_light_buffers()

    def release(self):
        self._cuda_context.pop()
    
    # region cuda
    def _init_cuda(self):
        target_device = self.TargetDevice
        cuda_driver.init()
        self._cuda_device = cuda_driver.Device(target_device)
        self._cuda_context = self._cuda_device.make_context()
    # endregion
    
    def _init_opengl(self):
        gl.glClearColor(0, 0, 0, 0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_CULL_FACE)

    def _init_framebuffers(self):
        self._default_gBuffer_shader = Shader.Default_GBuffer_Shader()
        '''For submit data to gBuffer'''

        self._default_defer_render_shader = Shader.Default_Defer_Shader()
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
                                   internalFormat=TextureInternalFormat.RGBA16F, # alpha for masking
                                   data_type=TextureDataType.HALF,
                                   share_to_torch=True)
        '''texture for saving every pixels' color in each frame'''
        self.colorFBOTex.sendToGPU()
        
        self.idFBOTex = Texture(name='__ID_FBO__',
                                width=winWidth,
                                height=winHeight,
                                format=TextureFormat.RGBA_INT,
                                s_wrap=TextureWrap.CLAMP_TO_EDGE,
                                t_wrap=TextureWrap.CLAMP_TO_EDGE,
                                min_filter=TextureFilter.NEAREST,
                                mag_filter=TextureFilter.NEAREST,
                                internalFormat=TextureInternalFormat.RGBA_16I,  # (objID, material id, uv_Xcoord, uv_Ycoord)
                                data_type=TextureDataType.SHORT,
                                share_to_torch=True)
        '''texture for saving every pixels' vertex ID in each frame'''
        self.idFBOTex.sendToGPU()
        
        self.posFBOTex = Texture(name='__POS_FBO__',
                                width=winWidth,
                                height=winHeight,
                                format=TextureFormat.RGB,
                                s_wrap=TextureWrap.CLAMP_TO_EDGE,
                                t_wrap=TextureWrap.CLAMP_TO_EDGE,
                                min_filter=TextureFilter.NEAREST,
                                mag_filter=TextureFilter.NEAREST,
                                internalFormat=TextureInternalFormat.RGB32F,    # use 32 here since cudaGraphicsGLRegisterImage doesn't support RGB16F
                                data_type=TextureDataType.FLOAT,
                                share_to_torch=True)
        '''texture for saving every frame's position in each pixel'''
        self.posFBOTex.sendToGPU()
        
        self.normal_and_depth_FBOTex = Texture(name='__NORMAL_FBO__',
                                    width=winWidth,
                                    height=winHeight,
                                    format=TextureFormat.RGBA,
                                    s_wrap=TextureWrap.CLAMP_TO_EDGE,
                                    t_wrap=TextureWrap.CLAMP_TO_EDGE,
                                    min_filter=TextureFilter.NEAREST,
                                    mag_filter=TextureFilter.NEAREST,
                                    internalFormat=TextureInternalFormat.RGBA16F,    # alpha channel for saving depth
                                    data_type=TextureDataType.HALF,
                                    share_to_torch=True)
        '''texture for saving every pixel's normal in each frame'''
        self.normal_and_depth_FBOTex.sendToGPU()
        
        self.noiseFBOTex = Texture(name='__NOISE_FBO__',
                                   width=winWidth,
                                   height=winHeight,
                                   format=TextureFormat.RGBA,
                                   s_wrap=TextureWrap.REPEAT,
                                   t_wrap=TextureWrap.REPEAT,
                                   min_filter=TextureFilter.NEAREST,
                                   mag_filter=TextureFilter.NEAREST,
                                   internalFormat=TextureInternalFormat.RGBA16F,    # 4 channel to make the shape same as a random latent
                                   data_type=TextureDataType.HALF,
                                   share_to_torch=True)
        '''Noise texture(which has the same size of model's latent) for further use in AI rendering'''
        self.noiseFBOTex.sendToGPU()
        
        self.depthFBOTex = Texture(name='__DEPTH_FBO__',
                                    width=winWidth,
                                    height=winHeight,
                                    format=TextureFormat.DEPTH,
                                    share_to_torch=False)   # depth tex is just for depth test
        self.depthFBOTex.sendToGPU()
        
        self._gBuffer = gl.glGenFramebuffers(1)
        self.BindFrameBuffer(self._gBuffer)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self.colorFBOTex.textureID, 0)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT1, gl.GL_TEXTURE_2D, self.idFBOTex.textureID, 0)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT2, gl.GL_TEXTURE_2D, self.posFBOTex.textureID, 0)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT3, gl.GL_TEXTURE_2D, self.normal_and_depth_FBOTex.textureID, 0)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT4, gl.GL_TEXTURE_2D, self.noiseFBOTex.textureID, 0)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_TEXTURE_2D, self.depthFBOTex.textureID, 0)
        gl.glDrawBuffers(5, [gl.GL_COLOR_ATTACHMENT0, gl.GL_COLOR_ATTACHMENT1, gl.GL_COLOR_ATTACHMENT2, gl.GL_COLOR_ATTACHMENT3, gl.GL_COLOR_ATTACHMENT4])
        
        if (gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE):
            raise Exception("G-Framebuffer is not complete! Some error occurred.")
        
        self.BindFrameBuffer(0)
        self._default_defer_render_task = self._wrapDeferRenderTask()
        
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

        self._default_post_process_shader = Shader.Default_Post_Shader()

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
        self._final_draw = self._wrapPostProcessTask(self._default_post_process_shader, final_draw)
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

    def _wrapRenderTask(self, task: Optional[Callable] = None, shader: Optional[Shader] = None, mesh: Optional[Mesh] = None):
        '''
        This wrapper will help setting the meshID as "objID" to the shader.
        If task is not given, mesh.draw() will be called.
        '''
        if task is None and mesh is None:
            raise ValueError("task and mesh cannot be both None")
        shader = shader or self._default_gBuffer_shader
        return partial(_wrapRenderTask, task, shader, mesh)
    
    def _wrapDeferRenderTask(self, shader=None, task=None):
        '''
        This wrapper help to bind the gBuffer textures to the shader(color, pos, normal, etc.).
        if task is not given, it will draw the screen quad.
        Defer renderer could texture the following textures::
            gColor_and_depth (r,g,b,depth), depth is in [0,1]
            gPos (x,y,z)
            gNormal (x,y,z)
            g_UV_and_ID (u,v, meshID)
        '''
        shader = shader or self._default_defer_render_shader
        return partial(_wrapDeferRenderTask, self, shader, task)
    
    def _wrapPostProcessTask(self, shader: Shader, task: Optional[Callable] = None):
        '''
        This wrapper helps to bind the last screen texture to the shader and draw a quad(if task is not given).
        screen buffer texture will also be swapped after the task is done.
        '''
        return partial(_wrapPostProcessTask, self, shader, task)

    def _execute_render_task(self):
        allTasks = list(self._renderTasks.tempEvents + self._renderTasks.events)
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
                print(f"Render Task ({order}, {func}) Error. Msg: {e}. Skipped.")
        self._renderTasks._tempEvents.clear()

    # endregion

    # region opengl
    @property
    def TargetDevice(self)->int:
        '''return the target GPU for running the engine'''
        return self._target_device
    
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

    # region render
    def AddRenderTask(self, 
                      order: Union[int, RenderOrder], 
                      task: Optional[Callable] = None, 
                      shader: Optional[Shader] = None, 
                      mesh: Optional[Mesh] = None, 
                      forever: bool = False):
        '''
        Add a render task to render queue. Note that the task is actually submitting data to GBuffers, not really rendering.
        :param order: order of the task. The smaller the order, the earlier the task will be executed.
        :param task: a callable func that takes no parameter. The task should includes drawing commands. If task is None, you could just give mesh so as to use the mesh's draw.
        :param shader: if shader is None, the task will use default gbuffer shader. Shader must have a uniform int named "objID".
        :param mesh: mesh and task can't be None at the same time. If mesh is None, the task should include those drawing commands.
        :param forever: If forever = True, the task will be called every frame.
        '''
        order = order.value if isinstance(order, RenderOrder) else order
        if forever:
            self._renderTasks.addForeverTask(self._wrapRenderTask(task, shader, mesh), order)
        else:
            self._renderTasks.addTask(self._wrapRenderTask(task, shader, mesh), order)

    def AddDeferRenderTask(self, task: Optional[Callable] = None, shader: Optional[Shader] = None, order: int = 0, forever: bool = True):
        '''
        Add a defer render task to which will be called after all normal render tasks.
        :param shader: shader must have a uniform named "screenTexture".
        :param task: task should call DrawScreen() to draw the screen texture finally. If task is None, will just call DrawScreen() instead.
        :param order: order of the task. The smaller the order, the earlier the task will be executed.
        :param forever: If forever = True, the task will be called every frame.
        :return:
        '''
        wrap = self._wrapDeferRenderTask(shader, task)
        if forever:
            self._deferRenderTasks.addForeverTask(wrap, order)
        else:
            self._deferRenderTasks.addTask(wrap, order)

    def AddPostProcessTask(self, shader: Shader, task: Optional[Callable] = None, order: int = 0, forever: bool = True):
        '''
        Post process shader must have a uniform named "screenTexture".
        This texture return rgba color.
        '''
        wrap = self._wrapPostProcessTask(shader, task)
        if forever:
            self._postProcessTasks.addForeverTask(wrap, order)
        else:
            self._postProcessTasks.addTask(wrap, order)
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
        '''return a shadow map texture id. If map with the same size and dimension doesn't exist, create one.'''
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
        # direct output when debug mode
        self.BindFrameBuffer(0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT) # type: ignore
        self._execute_render_task()

    def on_frame_run(self):
        # normal render
        
        self.BindFrameBuffer(self._gBuffer)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT) # type: ignore
        self._execute_render_task()  # depth test will be enabled in this function
        
        colorData = self.colorFBOTex.numpy_data(flipY=True)
        #colorData = self.colorFBOTex.update_tensor().cpu().numpy()
        
        # output data to SD
        if self.engine.DiffusionManager.ShouldOutputFrame:    
            idData = self.idFBOTex.numpy_data(flipY=True)
            posData = self.posFBOTex.numpy_data(True)
            
            normal_and_depth_data = self.normal_and_depth_FBOTex.numpy_data(flipY=True)
            # depth data is in the alpha channel of `normal_and_depth_data`
            
            normalData = normal_and_depth_data[:, :, :3]
            depthData = normal_and_depth_data[:, :, 3]
            noiseData = self.noiseFBOTex.numpy_data(flipY=True)
            
            diffuseManager = self.engine.DiffusionManager
            diffuseManager.OutputMap('color', colorData)
            diffuseManager.OutputMap('normal', normalData)
            diffuseManager.OutputNumpyData('id', idData)
            diffuseManager.OutputNumpyData('pos', posData)
            diffuseManager.OutputNumpyData('noise', noiseData)
            diffuseManager.OutputDepthMap(depthData)
        
        # get data back from SD
        # TODO: load the color data back to self._gBuffer_color_and_depth_texture texture, i.e. colorData = ...
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.colorFBOTex.textureID)
        gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, 
                           self.engine.WindowManager.WindowWidth, 
                           self.engine.WindowManager.WindowHeight, 
                           self.colorFBOTex.format.value.gl_format,
                           self.colorFBOTex.data_type.value.gl_data_type,
                           colorData.tobytes())
        # TODO: update color pixel datas, i.e. pixelDict[id] = (oldColor *a + newColor *b), newColor = inverse light intensity of the pixel color
        # TODO: replace corresponding color pixel datas with color data from color dict

        # defer render: normal light effect apply
        gl.glDisable(gl.GL_DEPTH_TEST)
        self.BindFrameBuffer(self._postProcessFBO)  # output to post process FBO
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self.CurrentScreenTexture, 0)
        if len(self._deferRenderTasks) > 0:
            self._deferRenderTasks.execute(ignoreErr=True)  # apply light effect here
        else:
            self._default_defer_render_task()

        # post process
        gl.glDisable(gl.GL_DEPTH_TEST)  # post process don't need depth test
        self._postProcessTasks.execute(ignoreErr=True)
        self._final_draw()  # default post process shader is used here. Will also bind to FBO 0(screen)
        

    def on_frame_end(self):
        glfw.swap_buffers(self.engine.WindowManager.Window)
    # endregion




__all__ = ['RenderManager']
