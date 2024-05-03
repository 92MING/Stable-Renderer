if __name__ == '__main__': # for debugging
    import sys, os
    proj_path = os.path.join(os.path.dirname(__file__), '..', '..')
    sys.path.insert(0, proj_path)
    __package__ = 'common_utils.stable_render_utils'

import torch
import taichi as ti
from attr import attrs, attrib
from torch import Tensor
from typing import Literal, TypeAlias
from ..global_utils import GetOrAddGlobalValue, SetGlobalValue
from .trace import IDMap


if not GetOrAddGlobalValue('__TAICHI_INITED__', False):
    ti.init(arch=ti.gpu)
    SetGlobalValue('__TAICHI_INITED__', True)

@ti.kernel
def _taichi_find_dup_id_and_treat(id_map: ti.template(), value_map: ti.template(), mode: int):
    '''
    id_map: shape = (height * width, 2), cell = (map_index, vertex_id)
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
    
    id_map: shape = (height * width, 3), cell = (i, map_index, vertex_id)
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


@attrs
class CorrespondenceMap:
    
    k: int = attrib(default=3, kw_only=True)
    '''cache size. View directions are split into k*k parts.'''
    
    objID: int|None = attrib(default=None, kw_only=True)
    '''for matching the cells belonging to this map. If None, it means all cells are matched.'''
    materialID: int|None = attrib(default=None, kw_only=True)
    '''for matching the cells belonging to this map. If None, it means all cells are matched.'''
    
    width: int = attrib(default=512, kw_only=True)
    '''default width of the map. This is only for load/dump, it has no relationship with the size of the screen/window.'''
    height: int = attrib(default=512, kw_only=True)
    '''default height of the map. This is only for load/dump, it has no relationship with the size of the screen/window.'''
    channel_count: int = attrib(default=4, kw_only=True)
    '''rgba, rgb, etc.'''
    
    _values: Tensor = attrib(init=False)
    ''' shape = (k^2(map_index), width * height(vertex id), channel_count(color)), vertexID = self.height * y + x'''
    _writtens: Tensor = attrib(init=False)
    '''
    shape = (k^2(map_index), width * height(vertex id), 1(bool)), vertexID = self.height * y + x.
    This represents whether a blob has values or not (since the real value can also be all zeros)
    '''
    
    def __attrs_post_init__(self):
        self._values = torch.zeros(self.k*self.k, self.width * self.height, self.channel_count, dtype=torch.float32)
        self._writtens = torch.zeros(self.k*self.k, self.width * self.height, dtype=torch.bool)
        
    def __getitem__(self, index):
        return self._values[index]
    
    def get_map(self, index: int, width:int|None=None, height:int|None=None):
        '''return the color map of the index-th view direction. (height, width, channel_count)'''
        if width is None:
            width = self.width
        if height is None:
            height = self.height
        if width * height < self.width * self.height:   # means only the first part of the map is used
            return self._values[index, :width*height].view(height, width, self.channel_count)
        return self._values[index].view(height, width, self.channel_count)
    
    def get_maps(self, width:int|None=None, height:int|None=None):
        '''return (k*k, height, width, channel_count)'''
        if width is None:
            width = self.width
        if height is None:
            height = self.height
        if width * height < self.width * self.height:
            return self._values[:, :width*height].view(self.k*self.k, height, width, self.channel_count)
        return self._values.view(self.k*self.k, height, width, self.channel_count)
    
    @torch.no_grad()
    def update(self, 
               color_frames: list[Tensor]|Tensor, 
               id_maps: list[Tensor|IDMap]|Tensor|IDMap,
               mode: UpdateMode = 'first',
               ignore_obj_mat_id: bool=False):
        '''
        Args:
            - color_frames: shape = (width, height, channel_count)
            - id_maps: shape = (width, height, 4), cell = (objID, materialID, map_index, vertexID), where vertexID = self.height * y + x
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
            - ignore_obj_mat_id: if True, still update even if the objID and materialID are not matching to this map.
            
        Note:
            the size in color_frames & id_maps is just the window size. It has no relationship with the size of the map.
        '''
        if not isinstance(color_frames, list):
            color_frames = [color_frames]
        if not isinstance(id_maps, list):
            id_maps = [id_maps]
        for i in range(len(id_maps)):
            m = id_maps[i]
            if isinstance(m, IDMap):
                id_maps[i] = m.tensor
                
        for color_frame, id_map in zip(color_frames, id_maps):
            self._update(color_frame, id_map, mode, ignore_obj_mat_id)
    
    def _update(self: "CorrespondenceMap", 
                color_frame: Tensor, 
                id_map: Tensor, 
                mode: UpdateMode,
                ignore_obj_mat_id: bool=False):
        if self.channel_count < color_frame.shape[-1]:
            color_frame = color_frame[..., :self.channel_count]
        elif self.channel_count == 4 and color_frame.shape[-1] == 3:    # add alpha channel
            color_frame = torch.cat([color_frame, torch.ones_like(color_frame[..., :1])], dim=-1)
        
        # flatten maps
        indices = torch.arange(id_map.shape[0], device=id_map.device).view(-1, 1)
        id_map = id_map.view(-1, id_map.shape[-1])
        id_map = torch.cat([indices, id_map], dim=-1)  
        # shape = (width * height, 5)
        # cell = (i, objID, materialID, map_index, vertexID)
        
        color_frame = color_frame.view(-1, color_frame.shape[-1])
        color_frame = torch.cat([indices, color_frame], dim=-1)
        # shape = (width * height, channel_count + 1)
        # cell = (i, r, g, b, a)
        
        if not ignore_obj_mat_id:
            if self.objID is not None:
                id_map = id_map[id_map[..., 1] == self.objID]
            if self.materialID is not None:
                id_map = id_map[id_map[..., 2] == self.materialID]
            color_frame = color_frame[id_map[..., 0]]
        
        # remove the objID and materialID
        id_map = torch.cat([id_map[..., 0], id_map[..., 3:]], dim=-1)   # now cell = (i, map_index, vertexID)
        
        if mode in ['first', 'first_avg']:  # only choose cells that are not written
            # self._writtens's shape = (k^2(map_index), width * height(vertex id), 1(bool))
            written = self._writtens[id_map[..., 1], id_map[..., 2]]
            id_map = id_map[~written]
            color_frame = color_frame[~written]
        
        elif mode in ['replace', 'replace_avg']:
            id_map, color_frame = _find_dup_index_and_treat(id_map, color_frame, mode)  # type: ignore
            
        # update the values
        self._values[id_map[..., 1], id_map[..., 2]] = color_frame[..., 1:]
        self._writtens[id_map[..., 1], id_map[..., 2]] = True

    


__all__ = ['CorrespondenceMap']


if __name__ == '__main__':  # for debug
    ...