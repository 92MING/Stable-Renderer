import os
if __name__ == '__main__': # for debugging
    import sys
    proj_path = os.path.join(os.path.dirname(__file__), '..', '..')
    proj_path = os.path.abspath(proj_path)
    sys.path.insert(0, proj_path)
    __package__ = 'engine.static'

import torch
import multiprocessing as mp
from multiprocessing import set_start_method
import json
import zipfile
import taichi as ti
import numpy as np
import OpenGL.error
import OpenGL.GL as gl

from PIL import Image
from pathlib import Path
from attr import attrs, attrib
from torch import Tensor
from torchvision.io import read_image, ImageReadMode
from einops import rearrange
from typing import (
    Literal, TypeAlias, Optional, 
    get_args, TYPE_CHECKING
)
from uuid import uuid4
from functools import partial
from concurrent.futures import ProcessPoolExecutor, as_completed
import threading
from tqdm import tqdm
from collections import defaultdict

from common_utils.path_utils import TEMP_DIR, extract_index
from common_utils.global_utils import is_dev_mode
from common_utils.debug_utils import EngineLogger
from common_utils.decorators import Overload
from common_utils.math_utils import init_taichi
from engine.static.resources_obj import ResourcesObj
from engine.static.enums import TextureFormat

if TYPE_CHECKING:
    from engine.static.shader import Shader


@attrs
class IDMap:
    '''
    IDMap represents the ID information of each frame, which is used to build the correspondence map(a packed version of IDMap).
    Note: there could be cases that the idmap obj contains more than 1 frames' id data, e.g. on baking process.
    '''
    
    tensor: torch.Tensor = attrib()
    '''
    The real data of this id-map.
    You can also use a Texture object here, it will be converted to tensor automatically.
    id struct: (spriteID, materialID, map_index, vertexID), where vertexID = width * y + x
    '''

    frame_indices: list[int] = attrib(default=None)
    '''the frame indices of this map. Note that there could be multiple frames' data in this map, so it is a list of int.'''
    @property
    def frame_count(self)->int:
        '''the count of frames in this map.'''
        return len(self.frame_indices)
    
    def __getitem__(self, index:int)->Tensor:
        '''get the id-map tensor of the index-th frame.'''
        return self.tensor[index]
    
    def __len__(self):
        return self.frame_count

    masks: torch.Tensor = attrib(default=None)
    '''
    Mask for the id-map. shape = (N, height, width), 1 means ignore, 0 means not ignore.
    Note that there could be multiple frames' masks in this map.
    '''
    
    @property
    def height(self)->int:
        '''height of the map.'''
        return self.tensor.shape[-2]
    
    @property
    def width(self)->int:
        '''width of the map.'''
        return self.tensor.shape[-1]
    
    def __attrs_post_init__(self):
        from engine.static import Texture
        
        if self.frame_indices is None and self.tensor is not None:
            if isinstance(self.tensor, Texture):
                frame_count = 1
            elif isinstance(self.tensor, torch.Tensor):
                if len(self.tensor.shape) == 4:
                    frame_count = self.tensor.shape[0]
                elif len(self.tensor.shape) == 3:
                    frame_count = 1
                else:
                    raise ValueError("Invalid shape of real ID tensor.")
            else:
                raise ValueError("Invalid type of real ID tensor.")
            self.frame_indices = list(range(frame_count))
        elif isinstance(self.frame_indices, int):
            self.frame_indices = [self.frame_indices,]
        
        if isinstance(self.tensor, Texture):
            self.tensor = self.tensor.tensor(update=True, flip=True).to(torch.int32)
        if isinstance(self.tensor, torch.Tensor) and len(self.tensor.shape) ==3:
            self.tensor = torch.stack([self.tensor, ] * len(self.frame_indices), dim=0)
        
        if self.masks is None:
            self.masks = torch.zeros(self.frame_count, self.tensor.shape[-2], self.tensor.shape[-1], dtype=torch.float32)
        else:
            if len(self.masks.shape) == 2:  # (H, W)
                # add batch dim(frame count)
                self.masks = torch.stack([self.masks, ] * self.frame_count, dim=0)
    
    def __deepcopy__(self):
        tensor = self.tensor.clone()
        masks = self.masks.clone() if self.masks is not None else None
        m = IDMap(frame_indices=self.frame_indices, tensor=tensor, masks=masks) # type: ignore
        return m

    @classmethod
    def from_directory(cls,
                       directory: str,
                       frame_start: int,
                       num_frames: int) -> 'IDMap':
        '''
        Load the IDMap from a directory.

        Args:
            - directory: the directory that contains the id data.
            - frame_start: the start frame index.
            - num_frames: the number of frames to load.
        
        Returns:
            - IDMap object.
        '''
        assert os.path.exists(directory)
        assert frame_start >= 0 and num_frames > 0

        file_filter = lambda fname: fname.endswith(".npy") \
            and os.path.exists(os.path.join(directory, fname))

        filenames = list(filter(file_filter, os.listdir(directory)))
        reordered_filenames = sorted(
            filenames, key=lambda x: extract_index(x, filenames.index(x)))
        
        EngineLogger.debug(f"Reordered filenames: {reordered_filenames}")

        frame_indices = list(
            map(partial(extract_index, i=-1), reordered_filenames)
        )[frame_start: frame_start+num_frames]
        assert all(i != -1 for i in frame_indices), "Illegal filename(s) found."

        id_tensors = [] 
        for filename in reordered_filenames[frame_start: frame_start+num_frames]:
            data_path = os.path.join(directory, filename)

            id_tensor = torch.from_numpy(np.load(data_path)).squeeze()
            if not len(id_tensor.shape) == 3:
                raise ValueError(f"Invalid shape of id tensor: {id_tensor.shape}.")
            if not (id_tensor.shape[-1] == 4 or id_tensor.shape[1] == 4): # not chw or hwc
                raise ValueError(f"Invalid id tensor shape: {id_tensor.shape}.")

            id_tensors.append(id_tensor)

        if any(t.shape != id_tensors[0].shape for t in id_tensors):
            raise ValueError(f"Tensor data has inconsistent shapes.")

        if len(id_tensors) == 0:
            return None
        t = torch.stack(id_tensors, dim=0)

        return IDMap(frame_indices=frame_indices, tensor=t)


init_taichi()

@ti.kernel
def _taichi_find_dup_id_and_treat(id_map: ti.template(), value_map: ti.template(), mode: int):  # type: ignore
    '''
    id_map: flattened id_map. shape = (height * width, 2), cell = (map_index, vertex_id)
    value_map: shape = (height * width, channel_count), cell = rgba(or rgb)
    
    - replace(mode == 0): 
        remove values that have the same id(since the second appearance)
    - replace_avg (mode == 1): 
        average all values that have the same id
    '''
    if mode == 0:
        for i in range(id_map.shape[0]):
            if id_map[i][0] == -1:
                continue
            for j in range(i+1, id_map.shape[0]):
                if id_map[i][0] == id_map[j][0] and id_map[i][1] == id_map[j][1]:   # same id
                    id_map[j][0] = -1
    else:
        for i in range(id_map.shape[0]):
            if id_map[i][0] == -1:
                continue
            total = value_map[i]
            count = 1
            for j in range(i+1, id_map.shape[0]):
                if id_map[i][0] == id_map[j][0] and id_map[i][1] == id_map[j][1]:   # same id
                    total += value_map[j]
                    count += 1
                    id_map[j][0] = -1
            value_map[i] = total / count

def _find_dup_index_and_treat(id_map: Tensor, color_frame: Tensor, mode: Literal['replace', 'replace_avg']):
    '''
    remove the cells that have the same id(since the second appearance) or average them.
    
    id_map: flattened. shape = (height * width, 3), cell = (i, map_index, vertex_id)
    color_frame: shape = (height * width, channel_count + 1), cell = (i, ...rgba(or rgb))
    '''
    mode_int = 0 if mode == 'replace' else 1
    
    id_map_ti = ti.Vector.field(2, ti.i32, shape=(id_map.shape[0], ))
    id_map_ti.from_torch(id_map[..., 1:])   # remove the i
    
    value_map_ti = ti.Vector.field(color_frame.shape[-1]-1, ti.f32, shape=(color_frame.shape[0], ))
    value_map_ti.from_torch(color_frame[..., 1:])   # remove the i
    
    _taichi_find_dup_id_and_treat(id_map_ti, value_map_ti, mode_int)
    
    # add back the i
    id_map = torch.cat([id_map[..., :1], id_map_ti.to_torch()], dim=-1)  
    color_frame = torch.cat([color_frame[..., :1], value_map_ti.to_torch()], dim=-1)
    
    # remove the cells that have map_index == -1
    id_map = id_map[id_map[..., 1] != -1]
    color_frame = color_frame[id_map[..., 0]]
    
    return id_map, color_frame


UpdateMode:TypeAlias = Literal['replace', 'replace_avg', 'first', 'first_avg']
'''
Mode to update the value in the map.

    - replace: 
        replace the old value with the new value
    - replace_avg: 
        average the old value with the new value, and put back
    - first: 
        only update the value if the old cell in not written
    - first_avg: 
        only update the value if the old cell in not written. 
        But if there are multiple values for the same cell in this update process, average them.
'''
LoadVertexPositionMode:TypeAlias = Literal['UV', 'Mesh']
'''
Mode to load vertex positions to CorrespondMap.

    - UV:
        vertex positions are from UV Map, position should be y = 1 - v, x = u
    - Mesh:
        vertex positions are Mesh vertices, position should be (x, y)
'''

frame_queue: Optional[mp.Queue] = None


@attrs(repr=False, eq=False)
class CorrespondMap(ResourcesObj):
    
    BaseClsName = 'CorrespondMap'
    
    k: int = attrib(default=3, kw_only=True)
    '''cache size. View directions are split into k*k parts.'''
    height: int = attrib(default=512, kw_only=True)
    '''default height of the map. This is only for load/dump, it has no relationship with the size of the screen/window.'''
    width: int = attrib(default=512, kw_only=True)
    '''default width of the map. This is only for load/dump, it has no relationship with the size of the screen/window.'''
    channel_count: int = attrib(default=4, kw_only=True)
    '''rgba, rgb, etc.'''
    
    _values: Tensor = attrib(init=False)
    ''' shape = (k^2(map_index), height*width(vertex id), channel_count(color)), vertexID = self.width * y + x'''
    _writtens: Tensor = attrib(init=False)
    '''
    shape = (k^2(map_index), height*width(vertex id)), vertexID = self.width * y + x.
    This represents whether a blob has values or not (since the real value can also be all zeros)
    '''
    
    texID: Optional[int] = attrib(default=None, init=False)
    '''sampler2DArray's texture id in OpenGL. This is for rendering purpose.'''

    vertex_screen_positions: Optional[torch.Tensor] = attrib(default=None, init=False)
    '''
    for runtime baking. structure:
        Tensor in shape of (N, 7), where the 7 elements are 
        (object_id, material_id, map_index, vertex_id, x, y, frame_index).
    call load_vertex_screen_positions to load the data.
    '''
    
    # ================= COLOR METHODS ================= # 

    def __attrs_post_init__(self):
        self._values = torch.zeros(self.k*self.k, self.height * self.width, self.channel_count, dtype=torch.float16)
        self._writtens = torch.zeros(self.k*self.k, self.height * self.width, dtype=torch.bool)
        super().__attrs_post_init__()
    
    def __repr__(self):
        return f"<CorrespondMap: {self.name or 'untitled'}, k={self.k}, size={self.height}x{self.width}, channel_count={self.channel_count}>"
    
    def __getitem__(self, index):
        '''
        You can directly access the color value of the index-th view direction. (height, width, channel_count).
        This is an alias of `self._values[index]`.
        '''
        return self._values[index]
    
    # region gl texture related
    def clear(self):
        '''clear the data in the map.'''
        super().clear() # just for printing logs
        if self.texID is not None:
            try:
                buffer = np.array([self.texID], dtype=np.uint32)
                gl.glDeleteTextures(1, buffer)
            except OpenGL.error.GLError as glErr:
                if glErr.err == 1282:
                    pass
                else:
                    EngineLogger.warn('Warning: failed to delete texture. Error: {}'.format(glErr))
            except Exception as err:
                EngineLogger.warn('Warning: failed to delete texture. Error: {}'.format(err))
            self.texID = None
        self._values.zero_()
        self._writtens.zero_()
    
    def load(self):
        '''load data to opengl.'''
        if self.loaded:
            return
        super().load()
        
        self.texID = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, self.texID)
 
        if self.channel_count == 4:
            format = gl.GL_RGBA
            internalformat = gl.GL_RGBA16F
        elif self.channel_count == 3:
            format = gl.GL_RGB
            internalformat = gl.GL_RGB16F
        elif self.channel_count == 2:
            format = gl.GL_RG
            internalformat = gl.GL_RG16F
        elif self.channel_count == 1:
            format = gl.GL_RED
            internalformat = gl.GL_R16F
        else:
            raise ValueError("Unsupported channel count: ", self.channel_count)

        data =self._values.cpu().numpy().tobytes()
        
        gl.glTexStorage3D(gl.GL_TEXTURE_2D_ARRAY, 
                          1, 
                          internalformat, 
                          self.width, self.height, self.k*self.k)
        gl.glTexSubImage3D(
            gl.GL_TEXTURE_2D_ARRAY,
            0,
            0, 0, 0,
            self.width, self.height, self.k*self.k,
            format,
            gl.GL_HALF_FLOAT,
            data
        )
        
        gl.glTexParameteri(gl.GL_TEXTURE_2D_ARRAY, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D_ARRAY, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D_ARRAY, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D_ARRAY, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        
        self.engine.CatchOpenGLError()
        gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, 0)
    
    def set_data(self, map_index: int, data: bytes):
        gl.glTexSubImage3D(gl.GL_TEXTURE_2D_ARRAY, 
                            0,   # level
                            0, 0, map_index,     # xoffset, yoffset, zoffset
                            self.width, self.height, 1,  # width, height, depth
                            format,  # format
                            gl.GL_HALF_FLOAT,     # type
                            data)
    
    @property
    def loaded(self):
        return self.texID is not None
    
    @Overload
    def bind(self, slot:int, name:str, shader:'Shader'):    # type: ignore
        '''Use shader, and bind texture to a slot and set shader uniform with your given name.'''
        shader.useProgram()
        self.bind(slot, shader.getUniformID(name))
    
    @Overload
    def bind(self):
        '''Bind the texture to the slot.'''
        if self.texID is None:
            raise Exception('Texture is not yet sent to GPU. Cannot bind')
        gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, self.texID)
    
    @Overload
    def bind(self, slot:int, uniformID):
        if self.texID is None:
            raise Exception('Texture is not yet sent to GPU. Cannot bind')
        gl.glActiveTexture(gl.GL_TEXTURE0 + slot)   # type: ignore
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, self.texID)
        gl.glUniform1i(uniformID, slot)
        
    def unbind(self, slot:int):
        gl.glActiveTexture(gl.GL_TEXTURE0 + slot)   # type: ignore
        gl.glBindTexture(gl.GL_TEXTURE_2D_ARRAY, 0)
    # endregion
    
    def numpy_data(self, map_index: int, width:int|None=None, height:int|None =None, dtype=np.float16)->np.ndarray:
        '''return the numpy data of the map_index-th map.'''
        return self.get_map(map_index, height, width).cpu().numpy().astype(dtype)
    
    def get_map(self, index: int, height:int|None=None, width:int|None=None, ):
        '''
        Return the color map of the index-th view direction. (height, width, channel_count).
        The returning map will reshape to the given width and height if they are not None, otherwise, it will keep the original size.
        '''
        if height is None:
            height = self.height
        if width is None:
            width = self.width
        if height*width < self.height * self.width:   # means only the first part of the map is used
            return self._values[index, :width*height].view(height, width, self.channel_count)
        return self._values[index].view(height, width, self.channel_count)
    
    def get_maps(self, height:int|None=None, width:int|None=None):
        '''return (k*k, height, width, channel_count)'''
        if width is None:
            width = self.width
        if height is None:
            height = self.height
        if width * height < self.width * self.height:
            return self._values[:, :width*height].view(self.k*self.k, height, width, self.channel_count)
        return self._values.view(self.k*self.k, height, width, self.channel_count)
    
    def get_written_flag_map(self, index: int, height:int|None=None, width:int|None=None):
        '''
        `written` map indicates whether a cell has been written a value or not. 
        
        shape = (height, width)
        value = 1 means written, 0 means not written.
        '''
        if height is None:
            height = self.height
        if width is None:
            width = self.width
        if height*width < self.height * self.width:   # means only the first part of the map is used
            return self._writtens[index, :width*height].view(height, width)
        return self._writtens[index].view(height, width)
    
    def update(self, 
               color_frames: list[Tensor]|Tensor, 
               id_maps: list[Tensor|IDMap]|Tensor|IDMap,
               spriteID: int|None = None,
               materialID: int|None = None,
               mode: UpdateMode = 'first_avg',
               masks: list[Tensor]|Tensor|None=None,
               ignore_obj_mat_id: bool=False):
        '''
        Args:
            - color_frames: shape = (height, width, channel_count)
            - id_maps: shape = (height, width, 4), cell = (spriteID, materialID, map_index, vertexID), where vertexID = self.width * y + x
            - spriteID: if not None, only update the cells that have the same spriteID.
            - materialID: if not None, only update the cells that have the same materialID.
            - mode:
                - replace: 
                    replace the old value with the new value
                - replace_avg: 
                    average the old value with the new value, and put back
                - first: 
                    only update the value if the old cell in not written
                - first_avg: 
                    only update the value if the old cell in not written. 
                    But if there are multiple values for the same cell in this update process, average them.
            - masks: shape = (h, w), the region to ignore.
            - ignore_obj_mat_id: if True, still update even if the spriteID and materialID are not matching to this map.
            
        Note:
            the size in color_frames & id_maps is just the window size. It has no relationship with the size of the map.
        '''
        if isinstance(color_frames, Tensor):
            if len(color_frames.shape) == 4: # frames are concatenated as a tensor, split them
                color_frames = torch.split(color_frames, 1, dim=0)
            elif len(color_frames.shape) == 3:
                color_frames = [color_frames, ]
            else:
                raise ValueError("The shape of color_frames is invalid. Got: ", color_frames.shape)
            
        if isinstance(id_maps, Tensor):
            if len(id_maps.shape) == 4: # maps are concatenated as a tensor, split them
                id_maps = torch.split(id_maps, 1, dim=0)    # type: ignore
            elif len(id_maps.shape) == 3:
                id_maps = [id_maps, ]
            else:
                raise ValueError("The shape of id_maps is invalid. Got: ", id_maps.shape)
        elif isinstance(id_maps, IDMap):
            id_maps = [id_maps.tensor, ]
        
        if masks is None:
            masks = [None] * len(color_frames)  # type: ignore
        elif masks is not None:
            if isinstance(masks, Tensor):
                if len(masks.shape) == 4 and masks.shape[-1] == 1:  # not yet squeezed
                    masks = torch.split(masks.squeeze(-1), 1, dim=0)
                elif len(masks.shape) == 3:
                    masks = torch.split(masks, 1, dim=0)
                elif len(masks.shape) == 2:
                    masks = [masks,]
                else:
                    raise ValueError("The shape of masks is invalid. Got: ", masks.shape)
        
        if len(color_frames) != len(id_maps):   # type: ignore
            raise ValueError("The length of color_frames and id_maps should be the same.")
        if len(masks) != len(color_frames): # type: ignore
            raise ValueError("The length of masks should be the same as color_frames.")
        
        for i in range(len(id_maps)):   # type: ignore
            m = id_maps[i]  # type: ignore
            if isinstance(m, IDMap):
                id_maps[i] = m.tensor   # type: ignore
        
        for color_frame, id_map, mask in zip(color_frames, id_maps, masks): # type: ignore
            self._update(color_frame=color_frame, 
                         id_map=id_map, 
                         spriteID=spriteID,
                         materialID=materialID,
                         mode=mode, 
                         mask=mask, 
                         ignore_obj_mat_id=ignore_obj_mat_id)  # type: ignore
    
    def _update(self: "CorrespondMap", 
                color_frame: Tensor, 
                id_map: Tensor, 
                spriteID: int|None = None,
                materialID: int|None=None,
                mode: UpdateMode = 'first_avg',
                mask: Optional[Tensor]=None,
                ignore_obj_mat_id: bool=False):
        
        if self.channel_count < color_frame.shape[-1]:
            color_frame = color_frame[..., :self.channel_count]
        elif self.channel_count == 4 and color_frame.shape[-1] == 3:    # add alpha channel
            color_frame = torch.cat([color_frame, torch.ones_like(color_frame[..., :1])], dim=-1)
        
        # flatten maps
        indices = torch.arange(id_map.shape[0]*id_map.shape[1], device=id_map.device).view(-1, 1)
        id_map = id_map.view(-1, id_map.shape[-1])
        id_map = torch.cat([indices, id_map], dim=-1)
        # shape = (height * width, 5)
        # cell = (i, spriteID, materialID, map_index, vertexID)
        
        color_frame = color_frame.view(-1, color_frame.shape[-1])
        color_frame = torch.cat([indices, color_frame], dim=-1)
        # shape = (height * width, channel_count + 1)
        # cell = (i, r, g, b, a)
        
        if mask is not None:
            mask = mask.view(-1, 1) # shape = (height * width), flatten
            id_map = id_map[mask[..., 0]<1]     # 1 means ignore
            color_frame = color_frame[id_map[..., 0]]
        
        if not ignore_obj_mat_id:
            if spriteID is not None:
                id_map = id_map[id_map[..., 1] == spriteID]
            if materialID is not None:
                id_map = id_map[id_map[..., 2] == materialID]
            color_frame = color_frame[id_map[..., 0]]
        
        # remove the spriteID and materialID
        id_map = torch.cat([id_map[..., 0].unsqueeze(-1), id_map[..., 3:]], dim=-1)   # now cell = (i, map_index, vertexID)
        
        if mode in ['first', 'first_avg']:  # only choose cells that are not written
            # self._writtens's shape = (k^2(map_index), width * height(vertex id), 1(bool))
            written = self._writtens[id_map[..., 1], id_map[..., 2]]
            id_map = id_map[~written]
            color_frame = color_frame[~written]
        
        elif mode in ['replace', 'replace_avg']:
            id_map = id_map.contiguous()
            color_frame = color_frame.contiguous()
            id_map, color_frame = _find_dup_index_and_treat(id_map, color_frame, mode)  # type: ignore

        # update the values
        self._values[id_map[..., 1], id_map[..., 2]] = color_frame[..., 1:].to(self._values.dtype)
        self._writtens[id_map[..., 1], id_map[..., 2]] = True
    
    def dump(self, path: str|Path, name: Optional[str]=None, zip=False):
        name = name or self.name
        real_name = name
        count = 1
        while os.path.exists(os.path.join(path, real_name)):
            real_name = f"{name}_{count}"
            count += 1
        
        if zip:
            random_name = str(uuid4()).replace('-', '')
            real_path = TEMP_DIR / f"{random_name}.zip"
        else:
            real_path = os.path.join(path, real_name)
        os.makedirs(real_path, exist_ok=True)
            
        for i in range(self.k*self.k):
            map_data = self.get_map(i)
            
            img = 255. * map_data.cpu().numpy()
            img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
            img.save(os.path.join(real_path, f"{i}.png"))

            # also output as img, for easy debug & check. 
            # white means written, black means not written
            written = self.get_written_flag_map(i)
            img = 255. * written.cpu().numpy()
            img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8), mode='L')
            img.save(os.path.join(real_path, f"{i}_written.png"))
        
        meta = {    # spriteID & matID will not be saved, since it will be generated when loading
            "k": self.k,
            "height": self.height,
            "width": self.width,
            "channel_count": self.channel_count,
            "name": name
        }
        with open(os.path.join(real_path, 'meta.json'), 'w') as f:
            json.dump(meta, f)
        
        if zip:
            count = 1
            real_name = name
            while os.path.exists(os.path.join(path, f"{real_name}.zip")):
                real_name = f"{name}_{count}"
                count += 1
            saving_path = os.path.join(path, f"{real_name}.zip")
            
            with zipfile.ZipFile(saving_path, 'w') as z:
                for i in range(self.k*self.k):
                    z.write(os.path.join(real_path, f"{i}.png"), f"{i}.png")
                    z.write(os.path.join(real_path, f"{i}_written.png"), f"{i}_written.png")
                    os.remove(os.path.join(real_path, f"{i}.png"))
                    os.remove(os.path.join(real_path, f"{i}_written.png"))
                    
                z.write(os.path.join(real_path, 'meta.json'), 'meta.json')
                os.remove(os.path.join(real_path, 'meta.json'))
                z.close()
            os.rmdir(real_path) # will only remove when the folder is empty
    
    @classmethod
    def Load(cls, path: str|Path, name: Optional[str]=None):
        '''
        Load the correspond map from the disk.
        
        Args:
            - path: the path of the map. It should be a folder or a zip file.
            - name: the name of the map. If given, it will overwrite the name in the meta file.
            - spriteID: force to assign the spriteID to the map.
            - materialID: force to assign the materialID to the map.
        '''
        is_zip = False
        if os.path.isfile(path):
            is_zip = True
            random_name = str(uuid4()).replace('-', '')
            real_path = TEMP_DIR / f"{random_name}"
            with zipfile.ZipFile(path, 'r') as z:
                z.extractall(real_path)
        else:
            real_path = path
        
        with open(os.path.join(real_path, 'meta.json'), 'r') as f:
            meta = json.load(f)
        
        map = cls(name=name or meta['name'], 
                  k=meta['k'], 
                  height=meta['height'], 
                  width=meta['width'], 
                  channel_count=meta['channel_count'])
        
        for i in range(map.k*map.k):
            img = Image.open(os.path.join(real_path, f"{i}.png"))
            img = np.array(img)
            img = torch.tensor(img, dtype=torch.float32) / 255.
            map._values[i] = img.view(-1, map.channel_count)
            
            img = Image.open(os.path.join(real_path, f"{i}_written.png"))
            img = np.array(img)
            img = torch.tensor(img, dtype=torch.float32) / 255.
            img = img.bool()
            map._writtens[i] = img.view(-1)
            
            if is_zip:
                os.remove(os.path.join(real_path, f"{i}.png"))
                os.remove(os.path.join(real_path, f"{i}_written.png"))
        
        if is_zip:
            os.remove(os.path.join(real_path, 'meta.json'))
            os.rmdir(real_path)
        
        return map
    
    def load_vertex_screen_positions(self, id_map: IDMap):
        """
        Load vertex screen positions to CorrespondMap from IDMaps (multiple frames can be included).
        The tensor is in shape of (N, 7), where the 7 elements are 
        (object_id, material_id, map_index, vertex_id, x, y, frame_index).
        """
        id_tensor, frame_indices = id_map.tensor, id_map.frame_indices
        frames, height, width, id_elements = id_tensor.shape

        y_coordinates_tensor = torch.arange(
            width, device=id_tensor.device
        ).view(1, -1).expand(
            height, -1  # (H, W)
        ).view(1, height, width, 1).expand(
            frames, -1, -1, -1  # (B, H, W, 1)
        )
        # assert y_coordinates[1, 52, 12] == 12

        x_coordinates_tensor = torch.arange(
            height, device=id_tensor.device
        ).view(-1, 1).expand(
            -1, width  # (H, W)
        ).view(1, height, width, 1).expand(
            frames, -1, -1, -1  # (B, H, W, 1)
        )
        # assert x_coordinates[1, 52, 12] == 52

        frame_indices_tensor = torch.tensor(
            frame_indices, device=id_tensor.device
        ).view(-1, 1, 1, 1).expand(-1, height, width, 1)
        # (B, H, W, 1), where values in (H, W, 1) equals the frame index

        vertex_screen_info = torch.cat(
            [id_tensor, 
             x_coordinates_tensor,
             y_coordinates_tensor,
             frame_indices_tensor], dim=-1
        )

        # flatten the vertex screen positions
        flat_vertex_screen_info = vertex_screen_info.view(-1, id_elements + 3)

        # Filter out map_index == 2048
        flat_vertex_screen_info = flat_vertex_screen_info[flat_vertex_screen_info[..., 2] != 2048]

        # Filter out object_id == material_id == map_index == vertex_id == 0
        flat_vertex_screen_info = flat_vertex_screen_info[
            (flat_vertex_screen_info[..., 0] != 0) |
            (flat_vertex_screen_info[..., 1] != 0) |
            (flat_vertex_screen_info[..., 2] != 0) |
            (flat_vertex_screen_info[..., 3] != 0)
        ]

        self.vertex_screen_positions = flat_vertex_screen_info

        EngineLogger.info(f"Loaded vertex screen positions from IDMap.")
    
    @property
    def unique_vertex_ids(self) -> torch.Tensor:
        assert self.vertex_screen_positions is not None, "Vertex screen positions are not loaded."
        return self.vertex_screen_positions[..., 3].unique()
    


__all__ = ['IDMap', 'UpdateMode', 'CorrespondMap']



if __name__ == '__main__':  # for debug
    from common_utils.path_utils import TEMP_DIR
    
    def dump_test():
        m = CorrespondMap(name='test', k=3)
        m._values = torch.rand(9, 512*512, 4)
        m.dump(TEMP_DIR)
        
    def load_test():
        m = CorrespondMap.Load(TEMP_DIR / 'test')
        print(m)
        
    def update_test():
        m = CorrespondMap(name='test', k=3)
        fully_red = torch.zeros(512, 512, 4, dtype=torch.float32)
        fully_red[..., 0] = 1.
        fully_red[..., 3] = 1.
        id_map = torch.zeros(512, 512, 4, dtype=torch.int32)
        id_map[..., 2] = 4  # map_index
        id_map[..., 3] = torch.arange(512*512).view(512, 512)   # vertexID
        m.update(fully_red, id_map)
        m.dump(TEMP_DIR, name='update_test')
    
    def load_id_map_test():
        dir_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'resources', 'example-map-outputs', 'miku-sphere', 'id')
        dir_path = os.path.abspath(dir_path)
        id_map = IDMap.from_directory(dir_path,0, 16)
        print(id_map)
    
    def load_vertex_positions_test():
        dir_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', 'resources', 'example-map-outputs', 'miku-sphere', 'id')
        dir_path = os.path.abspath(dir_path)
        print('loading from:', dir_path)
        id_map = IDMap.from_directory(dir_path,0, 4)
        
        m = CorrespondMap(name='test', k=3)
        m.load_vertex_screen_positions(id_map)
        print(len(m.vertex_screen_positions))
        
    # dump_test()
    # load_test()
    # update_test()
    # load_id_map_test()
    load_vertex_positions_test()