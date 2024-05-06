if __name__ == '__main__': # for debugging
    import sys, os
    proj_path = os.path.join(os.path.dirname(__file__), '..', '..')
    sys.path.insert(0, proj_path)
    __package__ = 'common_utils.stable_render_utils'
    
import torch
import taichi as ti
from typing import TYPE_CHECKING

from ..global_utils import GetOrAddGlobalValue, SetGlobalValue


if not GetOrAddGlobalValue('__TAICHI_INITED__', False):
    ti.init(arch=ti.gpu)
    SetGlobalValue('__TAICHI_INITED__', True)

@ti.func
def _check_values_equal(values: ti.types.ndarray(),
                        value_len:int,
                        index_a_ivec2: ti.math.ivec2,  # (b, i)
                        index_b_ivec2: ti.math.ivec2,  # (b, i)
                        ):
    equal = True
    for i in range(value_len):
        if values[index_a_ivec2[0], index_a_ivec2[1], i] != values[index_b_ivec2[0], index_b_ivec2[1], i]:
            equal = False
    return equal

@ti.func
def _get_similarity(id_flatten_maps: ti.types.ndarray(),            # b, h*w, 4(objID, materialID, map_index, vertexID), pixel space
                    target_start_ivec2: ti.math.ivec2,          # ivec2(b, i), from where to start(cell space)
                    compare_target_start_ivec2: ti.math.ivec2,
                    cell_contains_pixel: int,   # how many pixels in a cell
                    contributions: ti.types.ndarray(),             # (b, h*w), e.g. 1/64, pixel space
                    )->ti.f32: 
    result = 0.0
    target_start_pixel_space = target_start_ivec2[1] * cell_contains_pixel
    compare_start_pixel_space = compare_target_start_ivec2[1] * cell_contains_pixel
    
    for x in range(target_start_pixel_space, target_start_pixel_space + cell_contains_pixel):
        for i in range(compare_start_pixel_space, compare_start_pixel_space + cell_contains_pixel):
            if _check_values_equal(id_flatten_maps, 
                                   id_flatten_maps.shape[-1], 
                                   ti.math.ivec2([target_start_ivec2[0], x]), 
                                   ti.math.ivec2([compare_target_start_ivec2[0], i])
                                   ):
                result += contributions[compare_target_start_ivec2[0], i] * contributions[target_start_ivec2[0], x]
    
    return result

@ti.func
def _sum_up_values(values: ti.types.ndarray(),  # original values
                   new_values: ti.types.ndarray(),  # for containing the summed up values
                   value_len:int,   # e.g. 320 for post-atten, 1 for latent
                   from_index_ivec2: ti.math.ivec2,           # ivec2 (b, index)
                   to_new_value_index_ivec2: ti.math.ivec2,    # ivec2 (b, index)
                   multiplier,  # float or int
                   ):
    multiplier_f = ti.cast(multiplier, ti.f32)
    for i in range(value_len):
        new_values[to_new_value_index_ivec2[0], to_new_value_index_ivec2[1], i] += \
            values[from_index_ivec2[0], from_index_ivec2[1], i] * multiplier_f

@ti.func
def _mult_values(values: ti.types.ndarray(),  # original values
                 index_ivec2: ti.math.ivec2,   # target cell's pos ((b, i) ,cell space)
                 value_len:int,   # e.g. 320 for post-atten, 1 for latent
                 multiplier,  # float or int
                 ):
    multiplier_f = ti.cast(multiplier, ti.f32)
    for i in range(value_len):
        values[index_ivec2[0], index_ivec2[1], i] *= multiplier_f

@ti.func
def _cell_values_overlap(id_flatten_maps: ti.types.ndarray(),                            # (b, h*w, 4), pixel space
                         values: ti.types.ndarray(),                             # e.g. (b, 4096, 320) for post-atten, (b, 4096, 1) for latent
                         new_values: ti.types.ndarray(),                        # for containing the summed up values, same shape as values
                         index_ivec2: ti.math.ivec2,    # target cell's pos ((b, i) ,cell space)
                         cell_contains_pixel: int,   # how many pixels in a cell
                         contributions: ti.types.ndarray(),                      # b, h*w, 1(e.g. 1/64)
                         ):
    '''overlap 1 cell with other cells. 1 Cell means a group of pixels that are in the same cell space, e.g. 1/4096'''
    
    # copy the target cell's value to new_values
    # final latent value = sigma(similarity * latent_value) / sigma(similarity)
    total_sim = 1.0
    
    _sum_up_values(values, 
                   new_values, 
                   values.shape[-1],
                   index_ivec2, 
                   index_ivec2, 
                   1.0) 
    
    for b, i in ti.ndrange(values.shape[0], values.shape[1]):
        if b==index_ivec2[0] and i == index_ivec2[1]:
            continue    # skip the target cell
        
        sim = _get_similarity(id_flatten_maps, 
                              ti.math.ivec2((index_ivec2[0], index_ivec2[1])),
                              ti.math.ivec2((b, i)),
                              cell_contains_pixel,
                              contributions)
        total_sim += sim
        _sum_up_values(values, 
                       new_values, 
                       values.shape[-1],
                       ti.math.ivec2([b, i]), 
                       index_ivec2, 
                       sim)
    
    _mult_values(new_values, index_ivec2, values.shape[-1], 1.0 / total_sim)

@ti.kernel
def cells_value_overlap(id_flatten_maps: ti.types.ndarray(),    # (b, h*w, 4), pixel space
                        values: ti.types.ndarray(),             # e.g. (b, 4096, 320) for post-atten, (b, 4096, 1) for latent
                        new_values: ti.types.ndarray(),         # placeholder for the new values
                        contributions: ti.types.ndarray(),       # (b, h*w), e.g. 1/64, pixel space
                        ):
    cell_contains_pixel = id_flatten_maps.shape[1] // values.shape[1]
    for b, i in ti.ndrange(values.shape[0], values.shape[1]):
        _cell_values_overlap(id_flatten_maps, 
                             values, 
                             new_values, 
                             ti.math.ivec2([b, i]), 
                             cell_contains_pixel, 
                             contributions)


if __name__ == '__main__':
    id_map = torch.Tensor([[[[1, 2, 3, 4], [1, 2, 3, 4]], [[1, 2, 3, 4], [0,0,0,4]]]])
    id_map_flatten = id_map.view(1, -1, 4).contiguous()
    
    values = torch.Tensor([[[1.0,1.0,1.0,1.0,1.0,1.0,1.0,1.0], [2.0,2.0,2.0,2.0,2.0,2.0,2.0,2.0]]])
    new_values = torch.zeros_like(values)
    contributions = torch.ones(1, 4) / 2
    
    print(new_values)
    cells_value_overlap(id_map_flatten, values, new_values, contributions)
    print(new_values)