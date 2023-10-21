import os.path
from .manager import Manager
from .runtimeManager import RuntimeManager
from utils.data_struct.event import AutoSortTask
import OpenGL.GL as gl
import numpy as np
import ctypes
from static.shader import Shader
from static.enums import RenderOrder
from static.mesh import Mesh
import glfw
from typing import Union


class RenderManager(Manager):
    '''Manager of all rendering stuffs'''

    _FrameRunFuncOrder = RuntimeManager._FrameRunFuncOrder + 1  # always run after runtimeManager

    def __init__(self,
                 enableHDR=True,
                 enableGammaCorrection=True,
                 gamma=2.2,
                 exposure=1.0,
                 saturation=1.0,
                 brightness=1.0,
                 contrast=1.0,):
        super().__init__()
        self.engine._renderManager = self
        self._renderTasks = AutoSortTask()
        self._deferRenderTasks = AutoSortTask()
        self._postProcessTasks = AutoSortTask()
        self._init_opengl()  # opengl settings
        self._init_defer_render()  # framebuffers for post-processing
        self._init_post_process(enableHDR=enableHDR, enableGammaCorrection=enableGammaCorrection, gamma=gamma, exposure=exposure,
                                saturation=saturation, brightness=brightness, contrast=contrast)
        self._init_quad()  # quad for post-processing

    # region private
    def _init_opengl(self):
        gl.glClearColor(0, 0, 0, 0)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glEnable(gl.GL_CULL_FACE)

    def _init_defer_render(self):
        self._default_gBuffer_shader = Shader.Default_GBuffer_Shader()
        '''For submit data to gBuffer'''

        self._default_defer_render_shader = Shader.Default_Defer_Shader()
        '''For render gBuffer data to screen'''

        winWidth, winHeight = self.engine.WindowManager.WindowSize

        # color data (rgb)
        self._gBuffer_color_and_depth = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._gBuffer_color_and_depth)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA16F, winWidth, winHeight, 0, gl.GL_RGBA, gl.GL_FLOAT, None)  # color is rgb8f
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)

        # position data (x, y, z)
        self._gBuffer_pos = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._gBuffer_pos)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB16F, winWidth, winHeight, 0, gl.GL_RGB, gl.GL_FLOAT, None)  # position is rgb16f
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)

        # normal data (x, y, z)
        self._gBuffer_normal = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._gBuffer_normal)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB, winWidth, winHeight, 0, gl.GL_RGB, gl.GL_FLOAT, None)  # normal is rgb8f
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)

        # identifier
        self._gBuffer_id = gl.glGenTextures(1)  # ivec3 id = (objID, uv_Xcoord, uv_Ycoord)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._gBuffer_id)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGB16I, winWidth, winHeight, 0, gl.GL_RGB_INTEGER, gl.GL_INT, None)  # id is rgb16i
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)

        # depth data
        self._gBuffer_depth = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._gBuffer_depth)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_DEPTH_COMPONENT, winWidth, winHeight, 0, gl.GL_DEPTH_COMPONENT, gl.GL_FLOAT, None)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_NEAREST)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)

        self._gBuffer = gl.glGenFramebuffers(1)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._gBuffer)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self._gBuffer_color_and_depth, 0)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT1, gl.GL_TEXTURE_2D, self._gBuffer_pos, 0)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT2, gl.GL_TEXTURE_2D, self._gBuffer_normal, 0)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT3, gl.GL_TEXTURE_2D, self._gBuffer_id, 0)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_DEPTH_ATTACHMENT, gl.GL_TEXTURE_2D, self._gBuffer_depth, 0)
        gl.glDrawBuffers(4, [gl.GL_COLOR_ATTACHMENT0, gl.GL_COLOR_ATTACHMENT1, gl.GL_COLOR_ATTACHMENT2, gl.GL_COLOR_ATTACHMENT3])
        if (gl.glCheckFramebufferStatus(gl.GL_FRAMEBUFFER) != gl.GL_FRAMEBUFFER_COMPLETE):
            raise Exception("G-Framebuffer is not complete! Some error occured.")
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        self._default_defer_render_task = self._wrapDeferRenderTask()

    def _init_post_process(self, enableHDR=True, enableGammaCorrection=True, gamma=2.2, exposure=1.0, saturation=1.0, brightness=1.0, contrast=1.0):
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
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)  # output to screen
            gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
            self.DrawScreen()
        self._final_draw = self._wrapPostProcessTask(self._default_post_process_shader, final_draw)
        '''
        _final_draw is actually a post renderring process but it is not in the post process list.
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
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._postProcessFBO)
        gl.glDrawBuffer(gl.GL_COLOR_ATTACHMENT0)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)

    def _init_quad(self):
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

    def _wrapRenderTask(self, task: callable = None, shader: Shader = None, mesh: Mesh = None):
        '''
        This wrapper will help setting the meshID as "objID" to the shader.
        If task is not given, mesh.draw() will be called.
        '''
        if task is None and mesh is None:
            raise ValueError("task and mesh cannot be both None")
        shader = shader or self._default_gBuffer_shader

        def wrap():
            shader.useProgram()
            if mesh is not None:
                shader.setUniform("objID", mesh.meshID)
            task() if task is not None else mesh.draw()
        return wrap

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
        def _bindGtexture(slot, textureID, name):
            gl.glActiveTexture(gl.GL_TEXTURE0 + slot)
            gl.glBindTexture(gl.GL_TEXTURE_2D, textureID)
            shader.setUniform(name, slot)
        shader = shader or self._default_defer_render_shader

        def wrap():
            shader.useProgram()
            _bindGtexture(0, self._gBuffer_color_and_depth, "gColor_and_depth")
            _bindGtexture(1, self._gBuffer_pos, "gPos")
            _bindGtexture(2, self._gBuffer_normal, "gNormal")
            _bindGtexture(3, self._gBuffer_id, "g_UV_and_ID")
            task() if task is not None else self._draw_quad()
        return wrap

    def _wrapPostProcessTask(self, shader: Shader, task: callable = None):
        '''
        This wrapper helps to bind the last screen texture to the shader and draw a quad(if task is not given).
        screen buffer texture will also be swapped after the task is done.
        '''
        def wrap():
            shader.useProgram()
            gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self.CurrentScreenTexture, 0)
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, self.LastScreenTexture)
            shader.setUniform("screenTexture", 0)
            task() if task is not None else self._draw_quad()
            self.SwapScreenTexture()
        return wrap

    def _excute_render_task(self):
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

    def _getTextureImg(self, gTexBufferID, glFormat, glDataType, npDataType, channel_num, flipY=True) -> np.ndarray:
        gl.glBindTexture(gl.GL_TEXTURE_2D, gTexBufferID)
        data = gl.glGetTexImage(gl.GL_TEXTURE_2D, 0, glFormat, glDataType)
        data = np.frombuffer(data, dtype=npDataType)
        data = data.reshape((self.engine.WindowManager.WindowSize[1], self.engine.WindowManager.WindowSize[0], channel_num))
        if flipY:
            data = data[::-1, :, :]  # flip array upside down
        return data
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
    def AddRenderTask(self, order: Union[int, RenderOrder], task: callable = None, shader: Shader = None, mesh: Mesh = None, forever: bool = False):
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

    def AddDeferRenderTask(self, task: callable = None, shader: Shader = None, order: int = 0, forever: bool = True):
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

    def AddPostProcessTask(self, shader: Shader, task: callable = None, order: int = 0, forever: bool = True):
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

    # region run
    def _onFrameRun(self):
        # normal render
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._gBuffer)
        gl.glEnable(gl.GL_DEPTH_TEST)
        gl.glClear(gl.GL_COLOR_BUFFER_BIT | gl.GL_DEPTH_BUFFER_BIT)
        self._excute_render_task()  # depth test will be enabled in this function

        # region SD
        # output data to SD
        colorAndDepthData = self._getTextureImg(self._gBuffer_color_and_depth, gl.GL_RGBA, gl.GL_FLOAT, np.float32, 4, flipY=True)
        colorData = colorAndDepthData[:, :, :3]
        depthData = colorAndDepthData[:, :, 3]
        posData = self._getTextureImg(self._gBuffer_pos, gl.GL_RGB, gl.GL_FLOAT, np.float32, 3, flipY=True)
        normalData = self._getTextureImg(self._gBuffer_normal, gl.GL_RGB, gl.GL_FLOAT, np.float32, 3, flipY=True)
        idData = self._getTextureImg(self._gBuffer_id, gl.GL_RGB_INTEGER, gl.GL_INT, np.int32, 3, flipY=True)

        if self.engine.SDManager.ShouldOutputFrame:
            sdManager = self.engine.SDManager
            sdManager.OutputMap('color', colorData)
            sdManager.OutputMap('pos', posData)
            sdManager.OutputMap('normal', normalData)
            sdManager.OuputIdMap(idData)
            sdManager.OutputDepthMap(depthData)

        # Code run normally until here, pending fixes for idData
        # get data back from SD
        # TODO: load the color data back to self._gBuffer_color_and_depth texture, i.e. colorData = ...
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._gBuffer_color_and_depth)
        gl.glTexSubImage2D(gl.GL_TEXTURE_2D, 0, 0, 0, self.engine.WindowManager.WindowWidth, self.engine.WindowManager.WindowHeight, gl.GL_RGB, gl.GL_FLOAT, colorData.tobytes())
        # TODO: update color pixel datas, i.e. pixelDict[id] = (oldColor *a + newColor *b), newColor = inverse light intensity of the pixel color
        # TODO: replace corresponding color pixel datas with color data from color dict
        # endregion

        # defer render: normal light effect apply
        gl.glDisable(gl.GL_DEPTH_TEST)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._postProcessFBO)  # output to post process FBO
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0, gl.GL_TEXTURE_2D, self.CurrentScreenTexture, 0)
        if len(self._deferRenderTasks) > 0:
            self._deferRenderTasks.execute(ignoreErr=True)  # apply light effect here
        else:
            self._default_defer_render_task()

        # post process
        gl.glDisable(gl.GL_DEPTH_TEST)  # post process don't need depth test
        self._postProcessTasks.execute(ignoreErr=True)
        self._final_draw()  # default post process shader is used here. Will also bind to FBO 0(screen)

    def _onFrameEnd(self):
        glfw.swap_buffers(self.engine.WindowManager.Window)
    # endregion


__all__ = ['RenderManager']
