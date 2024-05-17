import torch
import collections
import math

from inspect import signature
from torch import Tensor
from abc import ABC, abstractmethod
from deprecated import deprecated
from typing import (List, Dict, Any, Tuple, Optional, Callable, Literal, TYPE_CHECKING, Sequence, Union)
from functools import partial

from common_utils.debug_utils import ComfyUILogger
from common_utils.global_utils import is_engine_looping, is_dev_mode, is_verbose_mode
from .k_diffusion import sampling as k_diffusion_sampling
from .extra_samplers import uni_pc
from comfy import model_management, model_base
from comfy.conds import CONDRegular

if TYPE_CHECKING:
    from comfyUI.types import ConvertedCondition, SamplerCallback
    from comfy.model_sampling import ModelSamplingProtocol
    from comfy.controlnet import ControlBase


cond_obj = collections.namedtuple('cond_obj', ['input_x', 'mult', 'conditioning', 'area', 'control', 'patches', 'condition_type'])

class ConditionObj(cond_obj):
    '''
    Type hinting class for the namedtuple `cond_obj` in comfyUI's original code. 
    This obj basically acts as a container for 1 single condition for a latent.
    '''
    
    input_x: Tensor
    '''the latent'''
    mult: Tensor
    
    conditioning: Dict[str, CONDRegular]
    '''
    Conditions for this latent. Though this obj is for 1 single latent, it can have multiple condition values, e.g. for different layers.
    e.g. {'c_crossattn': <comfy.conds.CONDCrossAttn object...}
    '''
    area: Tuple[int, int, int, int]
    '''
    Area of this condition to be applied on the latent.
    '''
    control: Optional[Any]
    patches: Optional[Dict[str, Any]]
    condition_type: Literal['pos', 'neg']
    
def get_area_and_mult(conds: "ConvertedCondition",
                      x_in: Tensor, 
                      timestep_in: Tensor,
                      condition_type: Literal['pos', 'neg'] = 'pos',
                      repeat_to_batch=True): # though timestep is a tensor, it is has just 1 value
    if len(x_in.shape) == 3:
        x_in = x_in.unsqueeze(0)
    area = (x_in.shape[2], x_in.shape[3], 0, 0)
    strength = 1.0

    if 'timestep_start' in conds:
        timestep_start = conds['timestep_start']
        if timestep_in[0] > timestep_start:
            return None
    if 'timestep_end' in conds:
        timestep_end = conds['timestep_end']
        if timestep_in[0] < timestep_end:
            return None
    if 'area' in conds:
        area = conds['area']
    if 'strength' in conds:
        strength = conds['strength']

    input_x = x_in[:,:,area[2]:area[0] + area[2],area[3]:area[1] + area[3]]
    if 'mask' in conds:
        # Scale the mask to the size of the input
        # The mask should have been resized as we began the sampling process
        mask_strength = 1.0
        if "mask_strength" in conds:
            mask_strength = conds["mask_strength"]
        mask = conds['mask']
        assert(mask.shape[1] == x_in.shape[2])
        assert(mask.shape[2] == x_in.shape[3])
        mask = mask[:,area[2]:area[0] + area[2],area[3]:area[1] + area[3]] * mask_strength
        mask = mask.unsqueeze(1).repeat(input_x.shape[0] // mask.shape[0], input_x.shape[1], 1, 1)
    else:
        mask = torch.ones_like(input_x)
    mult = mask * strength

    if 'mask' not in conds:
        rr = 8
        if area[2] != 0:
            for t in range(rr):
                mult[:,:,t:1+t,:] *= ((1.0/rr) * (t + 1))
        if (area[0] + area[2]) < x_in.shape[2]:
            for t in range(rr):
                mult[:,:,area[0] - 1 - t:area[0] - t,:] *= ((1.0/rr) * (t + 1))
        if area[3] != 0:
            for t in range(rr):
                mult[:,:,:,t:1+t] *= ((1.0/rr) * (t + 1))
        if (area[1] + area[3]) < x_in.shape[3]:
            for t in range(rr):
                mult[:,:,:,area[1] - 1 - t:area[1] - t] *= ((1.0/rr) * (t + 1))

    conditioning = {}
    model_conds = conds["model_conds"]
    for c in model_conds:
        if is_dev_mode() and is_verbose_mode():
            ComfyUILogger.debug(f'(comfy.samplers.get_area_and_mult) processing condition `{c}`...')
        conditioning[c] = model_conds[c].process_cond(batch_size=x_in.shape[0], device=x_in.device, area=area, repeat=repeat_to_batch)
        # condition size will be repeated to batch_size here(if repeat=True), e.g. [1, 77, 768] -> [4, 77, 768] for batch_size=4
    
    control = conds.get('control', None)

    patches = None
    if 'gligen' in conds:
        gligen = conds['gligen']
        patches = {}
        gligen_type = gligen[0]
        gligen_model = gligen[1]
        if gligen_type == "position":
            gligen_patch = gligen_model.model.set_position(input_x.shape, gligen[2], input_x.device)
        else:
            gligen_patch = gligen_model.model.set_empty(input_x.shape, input_x.device)

        patches['middle_patch'] = [gligen_patch]

    return ConditionObj(input_x, mult, conditioning, area, control, patches, condition_type)

def cond_equal_size(c1: dict[str, CONDRegular], c2: dict[str, CONDRegular]):
    if c1 is c2:
        return True
    if c1.keys() != c2.keys():
        return False
    for k in c1:
        if not c1[k].can_concat(c2[k]):
            return False
    return True

def can_concat_cond(c1: ConditionObj, c2: ConditionObj):
    if c1.input_x.shape != c2.input_x.shape:
        return False

    def objects_contactable(obj1, obj2):
        if (obj1 is None) != (obj2 is None):
            return False
        if obj1 is not None:
            if obj1 is not obj2:
                return False
        return True

    if not objects_contactable(c1.control, c2.control):
        return False

    if not objects_contactable(c1.patches, c2.patches):
        return False

    return cond_equal_size(c1.conditioning, c2.conditioning)

def cond_cat(c_list: list[dict[str, "CONDRegular"]]):
    temp: dict[str, list["CONDRegular"]] = {}
    for condition in c_list:
        for condition_name in condition:
            if condition_name not in temp:
                temp[condition_name] = []
            temp[condition_name].append(condition[condition_name])  
            # e.g. if 1 pos prompt, 2 neg prompt, then temp['c_crossattn'] = [pos, neg1, neg2]
            # for batch_size=3, then each prompt's shape=[3, 77, 768]
            
    out = {}
    for condition_name in temp:
        conds = temp[condition_name]
        out[condition_name] = conds[0].concat(conds[1:])
    
    return out

def calc_cond_uncond_batch(model: model_base.BaseModel, 
                           cond: list["ConvertedCondition"], 
                           uncond: list["ConvertedCondition"], 
                           x_in: Tensor, 
                           timestep: Tensor,    # e.g. Tensor([0.7297]), the ratio of the current timestep to the total timesteps
                           model_options: dict,     # e.g. {'transformer_options': ...}
                           **kwargs):
    '''
    This method concat the cond & uncond's inference input as a batch, i.e. (2*4*64*64),
    and return both's result(latent space)
    '''
    
    # here, cond[0]['model_conds']['c_crossattn'].cond.shape is still [1, 77, 768]
    if is_dev_mode() and is_verbose_mode():
        pos_cross_attn_shapes = [c['model_conds']['c_crossattn'].cond.shape for c in cond]
        neg_cross_attn_shapes = [c['model_conds']['c_crossattn'].cond.shape for c in uncond]
        ComfyUILogger.debug(f'(comfy.samplers.calc_cond_uncond_batch) pos_cross_attn_shapes={pos_cross_attn_shapes}')
        ComfyUILogger.debug(f'(comfy.samplers.calc_cond_uncond_batch) neg_cross_attn_shapes={neg_cross_attn_shapes}')
    cond = cond or []
    uncond = uncond or []
    
    out_cond = torch.zeros_like(x_in)
    out_count = torch.ones_like(x_in) * 1e-37

    out_uncond = torch.zeros_like(x_in)
    out_uncond_count = torch.ones_like(x_in) * 1e-37

    COND = 0
    UNCOND = 1
    to_run: list[tuple[ConditionObj, Literal[0, 1]]] = []
    
    for i in cond:
        p = get_area_and_mult(i, x_in, timestep, condition_type='pos')
        # in normal case, `get_area_and_mult` will make the origin condition tensor repeated to batch_size
        if p is None:
            continue
        to_run += [(p, COND)]
        
    for i in uncond:
        p = get_area_and_mult(i, x_in, timestep, condition_type='neg')
        if p is None:
            continue
        to_run += [(p, UNCOND)]
        
    while len(to_run) > 0:
        first_cond = to_run[0]
        first_latent_shape = first_cond[0].input_x.shape
        
        to_batch_indices: list[int] = []   # can contains both pos or neg jobs
        
        for i in range(len(to_run)):
            if can_concat_cond(to_run[i][0], first_cond[0]):
                to_batch_indices += [i]

        to_batch_indices.reverse()
        to_batch = to_batch_indices[:1] # start from last one

        free_memory = model_management.get_free_memory(x_in.device)
        for i in range(1, len(to_batch_indices) + 1):   # from 1 to len(to_batch_indices), test the max possible batch size can be run
            possible_batch_indices = to_batch_indices[:len(to_batch_indices)//i]
            input_shape = [len(possible_batch_indices) * first_latent_shape[0]] + list(first_latent_shape)[1:]    # e.g. [2,] + [4, 64, 64] = [2, 4, 64, 64]
            if model.memory_required(input_shape) < free_memory:    # type: ignore
                to_batch = possible_batch_indices
                break

        all_input_x = []    # all latents
        mult = []
        conditions: list[dict[str, "CONDRegular"]] = []
        cond_or_uncond: list[Literal[0, 1]] = []    # e.g. [COND, UNCOND, COND, COND, UNCOND, ...]
        positive_cond_indices: list[int] = []
        area: list[tuple[int,int,int,int]] = []
        control: Optional["ControlBase"] = None
        patches: Optional[dict[str, Any]] = None    # the data for modifying the model's patches
        
        for i in to_batch:
            cond_obj, condition_type = to_run.pop(i)
            
            if condition_type == COND:
                positive_cond_indices.extend(list(range(len(all_input_x), len(all_input_x) + cond_obj.input_x.shape[0])))
            
            all_input_x.append(cond_obj.input_x)
            mult.append(cond_obj.mult)
            conditions.append(cond_obj.conditioning)
            area.append(cond_obj.area)
            control = cond_obj.control
            patches = cond_obj.patches
            cond_or_uncond.append(condition_type)
        
        batch_chunks = len(cond_or_uncond)	 
        batch_chunk_indices: list[list[int]] = []
        offset = 0
        for inp in all_input_x:
            batch_chunk_indices.append(list(range(offset, offset+inp.shape[0])))
            offset += inp.shape[0]
        
        input_x = torch.cat(all_input_x)
        c = cond_cat(conditions)    # e.g. for 1 pos, 2 neg, batch_size=3, here c['c_crossattn'].shape = [9, 77, 768], 9=3*(1+2)
        timestep_ = torch.cat([timestep] * len(cond_or_uncond))[:len(input_x)] # for case existing individual conditions, timestep should be duplicated, so need to slice it

        if control is not None:
            c['control'] = control.get_control(input_x, timestep_, c, len(cond_or_uncond))
  
        transformer_options = {}
        if 'transformer_options' in model_options:
            transformer_options = model_options['transformer_options'].copy()

        if patches is not None:
            if "patches" in transformer_options:
                cur_patches = transformer_options["patches"].copy()
                for p in patches:
                    if p in cur_patches:
                        cur_patches[p] = cur_patches[p] + patches[p]
                    else:
                        cur_patches[p] = patches[p]
                transformer_options["patches"] = cur_patches
            else:
                transformer_options["patches"] = patches

        transformer_options["cond_or_uncond"] = cond_or_uncond[:]
        transformer_options['positive_cond_indices'] = positive_cond_indices
        transformer_options["sigmas"] = timestep

        c['transformer_options'] = transformer_options
        c.update(kwargs)    # extra args are passed to the model from here.

        if 'model_function_wrapper' in model_options:
            output = model_options['model_function_wrapper'](model.apply_model, {"input": input_x, "timestep": timestep_, "c": c, "cond_or_uncond": cond_or_uncond}).chunk(batch_chunks)
        else:
            output = model.apply_model(input_x, timestep_, **c).chunk(batch_chunks) # basemodel apply_model
        del input_x
        
        for o in range(batch_chunks):
            if cond_or_uncond[o] == COND:
                out_cond[:,:,area[o][2]:area[o][0] + area[o][2],area[o][3]:area[o][1] + area[o][3]] += output[o] * mult[o]
                out_count[:,:,area[o][2]:area[o][0] + area[o][2],area[o][3]:area[o][1] + area[o][3]] += mult[o]
            else:
                out_uncond[:,:,area[o][2]:area[o][0] + area[o][2],area[o][3]:area[o][1] + area[o][3]] += output[o] * mult[o]
                out_uncond_count[:,:,area[o][2]:area[o][0] + area[o][2],area[o][3]:area[o][1] + area[o][3]] += mult[o]
        del mult
        
    out_cond /= out_count
    del out_count
    out_uncond /= out_uncond_count
    del out_uncond_count
    return out_cond, out_uncond

# region REAL SAMPLING ENTRY FUNCTION
def sampling_function(model: model_base.BaseModel,
                      x, 
                      timestep, 
                      uncond, 
                      cond, 
                      cond_scale, 
                      model_options={}, 
                      **kwargs):
    '''
    The real, main sampling function shared by all the samplers, for sampling 1 step.
    Returns denoised latent.
    '''
    if math.isclose(cond_scale, 1.0) and model_options.get("disable_cfg1_optimization", False) == False:
        uncond_ = None
    else:
        uncond_ = uncond
    cond_pred, uncond_pred = calc_cond_uncond_batch(model, 
                                                    cond, 
                                                    uncond_,  # type: ignore
                                                    x, 
                                                    timestep, 
                                                    model_options,
                                                    **kwargs)
    if "sampler_cfg_function" in model_options:
        args = {"cond": x - cond_pred, "uncond": x - uncond_pred, "cond_scale": cond_scale, "timestep": timestep, "input": x, "sigma": timestep,
                "cond_denoised": cond_pred, "uncond_denoised": uncond_pred, "model": model, "model_options": model_options}
        cfg_result = x - model_options["sampler_cfg_function"](args)
    else:
        cfg_result = uncond_pred + (cond_pred - uncond_pred) * cond_scale

    for fn in model_options.get("sampler_post_cfg_function", []):
        args = {"denoised": cfg_result, "cond": cond, "uncond": uncond, "model": model, "uncond_denoised": uncond_pred, "cond_denoised": cond_pred,
                "sigma": timestep, "model_options": model_options, "input": x}
        cfg_result = fn(args)

    return cfg_result

# endregion

# region model wrapper classes
class CFGNoisePredictor(torch.nn.Module):
    '''Real wrapper of `BaseModel`'''
    
    def __init__(self, model: model_base.BaseModel):
        super().__init__()
        self.inner_model = model
        
    def apply_model(self, x, timestep, cond, uncond, cond_scale, model_options={}, seed=None, **kwargs):
        if is_dev_mode() and is_verbose_mode():
            ComfyUILogger.debug(f'CFGNoisePredictor.apply_model: x.shape={x.shape}, timestep={timestep}, seed={seed}')
        out = sampling_function(self.inner_model, 
                                x, 
                                timestep, 
                                uncond, 
                                cond, 
                                cond_scale, 
                                model_options=model_options, 
                                seed=seed,
                                **kwargs)
        return out
    
    def forward(self, *args, **kwargs):
        return self.apply_model(*args, **kwargs)

class KSamplerX0Inpaint(torch.nn.Module):
    '''
    `KSamplerX0Inpaint is the wrapper of `CFGNoisePredictor`, and
    `CFGNoisePredictor` is the wrapper of the model.
    '''
    def __init__(self,
                 model: CFGNoisePredictor,
                 sigmas: torch.Tensor):
        super().__init__()
        self.inner_model = model
        self.sigmas = sigmas
        
    def forward(self, x, sigma, uncond, cond, cond_scale, denoise_mask, model_options={}, seed=None, **kwargs):
        if is_dev_mode() and is_verbose_mode():
            ComfyUILogger.debug(f'KSamplerX0Inpaint: x.shape={x.shape}, sigma.shape={sigma.shape}, cond_scale={cond_scale}, model_options={model_options}, seed={seed}')
        
        if denoise_mask is not None:
            if "denoise_mask_function" in model_options:
                denoise_mask = model_options["denoise_mask_function"](sigma, denoise_mask, extra_options={"model": self.inner_model, "sigmas": self.sigmas})
            latent_mask = 1. - denoise_mask
            x = x * denoise_mask + self.inner_model.inner_model.model_sampling.noise_scaling(sigma.reshape([sigma.shape[0]] + [1] * (len(self.noise.shape) - 1)), self.noise, self.latent_image) * latent_mask
        
        out = self.inner_model(x, sigma, cond=cond, uncond=uncond, cond_scale=cond_scale, model_options=model_options, seed=seed, **kwargs)
        if denoise_mask is not None:
            out = out * denoise_mask + self.latent_image * latent_mask  # type: ignore
        return out
# endregion

def simple_scheduler(model, steps):
    s = model.model_sampling
    sigs = []
    ss = len(s.sigmas) / steps
    for x in range(steps):
        sigs += [float(s.sigmas[-(1 + int(x * ss))])]
    sigs += [0.0]
    return torch.FloatTensor(sigs)

def ddim_scheduler(model, steps):
    s = model.model_sampling
    sigs = []
    ss = max(len(s.sigmas) // steps, 1)
    x = 1
    while x < len(s.sigmas):
        sigs += [float(s.sigmas[x])]
        x += ss
    sigs = sigs[::-1]
    sigs += [0.0]
    return torch.FloatTensor(sigs)

def normal_scheduler(model, steps, sgm=False, floor=False):
    s = model.model_sampling
    start = s.timestep(s.sigma_max)
    end = s.timestep(s.sigma_min)

    if sgm:
        timesteps = torch.linspace(start, end, steps + 1)[:-1]
    else:
        timesteps = torch.linspace(start, end, steps)

    sigs = []
    for x in range(len(timesteps)):
        ts = timesteps[x]
        sigs.append(s.sigma(ts))
    sigs += [0.0]
    return torch.FloatTensor(sigs)

def get_mask_aabb(masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate the axis-aligned bounding boxes (AABB) for a batch of masks.

    Args:
        masks (torch.Tensor): A tensor containing binary masks of shape (B, H, W), where B is the batch size,
                              H is the height, and W is the width.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing two tensors:
            - bounding_boxes: A tensor of shape (B, 4) representing the bounding boxes for each mask in the batch.
                              Each bounding box is represented as (x_min, y_min, x_max, y_max).
            - is_empty: A tensor of shape (B,) indicating whether each mask is empty or not. A mask is considered
                        empty if it has no non-zero elements.

    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device, dtype=torch.int)

    b = masks.shape[0]

    bounding_boxes = torch.zeros((b, 4), device=masks.device, dtype=torch.int)
    is_empty = torch.zeros((b), device=masks.device, dtype=torch.bool)
    for i in range(b):
        mask = masks[i]
        if mask.numel() == 0:
            continue
        if torch.max(mask != 0) == False:
            is_empty[i] = True
            continue
        y, x = torch.where(mask)
        bounding_boxes[i, 0] = torch.min(x)
        bounding_boxes[i, 1] = torch.min(y)
        bounding_boxes[i, 2] = torch.max(x)
        bounding_boxes[i, 3] = torch.max(y)

    return bounding_boxes, is_empty

def resolve_areas_and_cond_masks(
        conditions: list["ConvertedCondition"], h: int, w: int, device: str
    ):
    # Modifies the conditions in-place
    # We need to decide on an area outside the sampling loop in order to properly generate opposite areas of equal sizes.
    # While we're doing this, we can also resolve the mask device and scaling for performance reasons
    for i in range(len(conditions)):
        c = conditions[i]
        if 'area' in c:
            area = c['area']
            if area[0] == "percentage":
                # Refer to ConditioningSetAreaPercentage in nodes.py
                # area = ("percentage", height, width, y, x)
                # where height, width, y, x are all values between 0 and 1
                modified = c.copy()
                area = (max(1, round(area[1] * h)), max(1, round(area[2] * w)), round(area[3] * h), round(area[4] * w))
                modified['area'] = area
                c = modified
                conditions[i] = c

        if 'mask' in c:
            """
            Refer to ConditioningSetMask in nodes.py
            mask = {
                "mask: torch.Tensor,
                "set_area_to_bounds": bool
                "mask_strength": float
            }
            """
            mask = c['mask']
            mask = mask.to(device=device)
            modified = c.copy()
            if len(mask.shape) == 2:
                mask = mask.unsqueeze(0)
            if mask.shape[1] != h or mask.shape[2] != w:
                mask = torch.nn.functional.interpolate(mask.unsqueeze(1), size=(h, w), mode='bilinear', align_corners=False).squeeze(1)

            if modified.get("set_area_to_bounds", False):
                bounds = torch.max(torch.abs(mask), dim=0).values.unsqueeze(0)
                boxes, is_empty = get_mask_aabb(bounds)
                if is_empty[0]:
                    # Use the minimum possible size for efficiency reasons. (Since the mask is all-0, this becomes a noop anyway)
                    # Refer to ConditioningSetArea in nodes.py, width & height has minimum value of 64 // 8 = 8
                    modified['area'] = (8, 8, 0, 0)
                else:
                    x_min, y_min, x_max, y_max = boxes[0]
                    H, W, Y, X = (y_max - y_min + 1, x_max - x_min + 1, y_min, x_min)
                    H = max(8, H)
                    W = max(8, W)
                    area = (int(H), int(W), int(Y), int(X))
                    modified['area'] = area

            modified['mask'] = mask
            conditions[i] = modified

def create_cond_with_same_area_if_none(conds, c):
    if 'area' not in c:
        return

    c_area = c['area']
    smallest = None
    for x in conds:
        if 'area' in x:
            a = x['area']
            if c_area[2] >= a[2] and c_area[3] >= a[3]:
                if a[0] + a[2] >= c_area[0] + c_area[2]:
                    if a[1] + a[3] >= c_area[1] + c_area[3]:
                        if smallest is None:
                            smallest = x
                        elif 'area' not in smallest:
                            smallest = x
                        else:
                            if smallest['area'][0] * smallest['area'][1] > a[0] * a[1]:
                                smallest = x
        else:
            if smallest is None:
                smallest = x
    if smallest is None:
        return
    if 'area' in smallest:
        if smallest['area'] == c_area:
            return

    out = c.copy()
    out['model_conds'] = smallest['model_conds'].copy() #TODO: which fields should be copied?
    conds += [out]

def calculate_start_end_timesteps(
        model: model_base.BaseModel,
        conds: list["ConvertedCondition"]):
    """
    Add 'timestep_start' and 'timestep_end' to each conditioning
    if 'start_percent' or 'end_percent' is present.
    """
    s = model.model_sampling
    for t in range(len(conds)):
        x = conds[t]

        timestep_start = None
        timestep_end = None
        if 'start_percent' in x:
            timestep_start = s.percent_to_sigma(x['start_percent'])
        if 'end_percent' in x:
            timestep_end = s.percent_to_sigma(x['end_percent'])

        if (timestep_start is not None) or (timestep_end is not None):
            n = x.copy()
            if (timestep_start is not None):
                n['timestep_start'] = timestep_start
            if (timestep_end is not None):
                n['timestep_end'] = timestep_end
            conds[t] = n

def pre_run_control(model: model_base.BaseModel, conds: list["ConvertedCondition"]):
    """
    Modifies the conds in-place
    Preruns the controlnets in the conds
    Refer to ControlNet class in controlnet.py 
    """
    s = model.model_sampling
    for t in range(len(conds)):
        x = conds[t]

        timestep_start = None
        timestep_end = None
        percent_to_timestep_function = lambda a: s.percent_to_sigma(a)
        if 'control' in x:
            x['control'].pre_run(model, percent_to_timestep_function)

def apply_empty_x_to_equal_area(conds, uncond, name, uncond_fill_func):
    cond_cnets = []
    cond_other = []
    uncond_cnets = []
    uncond_other = []
    for t in range(len(conds)):
        x = conds[t]
        if 'area' not in x:
            if name in x and x[name] is not None:
                cond_cnets.append(x[name])
            else:
                cond_other.append((x, t))
    for t in range(len(uncond)):
        x = uncond[t]
        if 'area' not in x:
            if name in x and x[name] is not None:
                uncond_cnets.append(x[name])
            else:
                uncond_other.append((x, t))

    if len(uncond_cnets) > 0:
        return

    if len(uncond_other) > 0:
        for x in range(len(cond_cnets)):
            temp = uncond_other[x % len(uncond_other)]
            o = temp[0]
            if name in o and o[name] is not None:
                n = o.copy()
                n[name] = uncond_fill_func(cond_cnets, x)
                uncond += [n]
            else:
                n = o.copy()
                n[name] = uncond_fill_func(cond_cnets, x)
                uncond[temp[1]] = n

def encode_model_conds(model_function: Callable[[Dict[str, Any]], Dict[str, Any]],
                       conds: list["ConvertedCondition"],
                       noise: torch.Tensor,
                       device: str,
                       prompt_type: Literal["positive", "negative"],
                       **kwargs):
    # possible kwargs: `latent_image`, `denoise_mask`, `seed`,...

    for t in range(len(conds)):
        x = conds[t]
        params = x.copy()
        params["device"] = device
        params["noise"] = noise
        params["width"] = params.get("width", noise.shape[3] * 8)
        params["height"] = params.get("height", noise.shape[2] * 8)
        params["prompt_type"] = params.get("prompt_type", prompt_type)
        for k in kwargs:
            if k not in params:
                params[k] = kwargs[k]

        # model function create a new list of model conditions
        # for the extra conditions
        out = model_function(**params)

        x = x.copy()
        model_conds = x['model_conds'].copy()
        for k in out:
            model_conds[k] = out[k]
        x['model_conds'] = model_conds
        conds[t] = x
    
    # e.g. for 2 neg prompt, [cond['model_conds']['c_crossattn'].cond.shape for cond in conds] = [[1, 77, 768], [1, 77, 768]]
    return conds


# region K-sampler method wrappers
KSAMPLER_NAMES = ["euler", "euler_ancestral", "heun", "heunpp2","dpm_2", "dpm_2_ancestral",
                  "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_sde_gpu",
                  "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm",]
'''all possible ksampler types'''

SAMPLER_NAMES = KSAMPLER_NAMES + ["ddim", "uni_pc", "uni_pc_bh2"]
'''all possible sampler types. This is the full list of samplers that can be used in the UI.'''

SCHEDULER_NAMES = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"]
'''all possible schedulers'''

class Sampler(ABC):
    '''Base class of samplers.'''
    
    @abstractmethod
    def sample(self, *args, **kwargs):
        raise NotImplementedError

    def max_denoise(self, model_wrap, sigmas):
        max_sigma = float(model_wrap.inner_model.model_sampling.sigma_max)
        sigma = float(sigmas[0])
        return math.isclose(max_sigma, sigma, rel_tol=1e-05) or sigma > max_sigma

def _get_method_param_count(method: Callable) -> int:
    param_count = len(signature(method).parameters)
    return param_count

# callbacks for sampling
def _call_no_arg_callback(c, data_form_upper):
    '''calling no-arg callback'''
    return c()
def _call_1_arg_callback(callback, total_steps, timesteps, sigmas: torch.Tensor, dict_data_from_upper: dict):
    '''when callbacks with only 1 arg accepted is given, the `SamplingCallbackContext` is passed as the only arg.'''
    from comfyUI.types import SamplingCallbackContext
    if isinstance(sigmas, torch.Tensor):
        sigmas = sigmas.squeeze().tolist()  # type: ignore
    context = SamplingCallbackContext(step_index=dict_data_from_upper["i"], 
                                      denoised=dict_data_from_upper["denoised"], 
                                      noise=dict_data_from_upper["x"], 
                                      total_steps=total_steps,
                                      timesteps=timesteps,
                                      sigmas=sigmas)    # type: ignore
    for key, val in dict_data_from_upper.items():
        if key in dir(context):
            continue
        setattr(context, key, val)
    return callback(context)
def _call_4_arg_callback(callback, total_steps, dict_data_from_upper: dict):
    '''This is the origin callback format for sampling methods: (i, denoised, x, total_steps)'''
    return callback(dict_data_from_upper["i"], dict_data_from_upper["denoised"], dict_data_from_upper["x"], total_steps)

class SAMPLER_METHOD(Sampler):
    '''wrapper for sampling methods'''
    def __init__(self, 
                 sampler_function: Callable,
                 extra_options={}, 
                 inpaint_options={}):
        self.sampler_function = sampler_function
        self.extra_options = extra_options
        self.inpaint_options = inpaint_options

    def sample(self,
               model_wrap: CFGNoisePredictor,
               sigmas: torch.Tensor,
               extra_args: Dict[str, Any],
               noise: torch.Tensor,
               callbacks: Union["SamplerCallback", Sequence["SamplerCallback"], None]=None,
               latent_image: Optional[torch.Tensor] = None,
               denoise_mask: Optional[torch.Tensor] = None,
               disable_pbar: bool = False,
               timesteps: Optional[List[int]] = None,
               **kwargs):
        extra_args["denoise_mask"] = denoise_mask
        model_k = KSamplerX0Inpaint(model_wrap, sigmas)
        model_k.latent_image = latent_image # type: ignore
        
        if self.inpaint_options.get("random", False): #TODO: Should this be the default?
            generator = torch.manual_seed(extra_args.get("seed", 41) + 1)
            model_k.noise = torch.randn(noise.shape, generator=generator, device="cpu").to(noise.dtype).to(noise.device)
        else:
            model_k.noise = noise

        noise = model_wrap.inner_model.model_sampling.noise_scaling(sigmas[0], noise, latent_image, self.max_denoise(model_wrap, sigmas))

        k_callbacks = []
        total_steps = len(sigmas) - 1
        if callbacks:
            if not isinstance(callbacks, Sequence):
                callbacks = [callbacks]
            for callback in callbacks:
                callback_arg_count = _get_method_param_count(callback)

                if callback_arg_count == 0: # no arguments
                    k_callbacks.append(partial(_call_no_arg_callback, callback))
                elif callback_arg_count == 1: # pass context
                    k_callbacks.append(partial(_call_1_arg_callback, callback, total_steps, timesteps, sigmas))
                elif callback_arg_count == 4:   # pass (i, denoised, x, total_steps)
                    k_callbacks.append(partial(_call_4_arg_callback, callback, total_steps))
                else:
                    assert False, f"Invalid callback function signature: {callback.__name__}"

        extra_options = extra_args.copy()
        extra_options.update(self.extra_options)
        extra_options.update(kwargs)
        
        samples = self.sampler_function(model_k, 
                                        noise, 
                                        sigmas, 
                                        extra_args=extra_options, 
                                        callbacks=k_callbacks, 
                                        disable=disable_pbar)
        return samples

@deprecated # `KSAMPLER` is the old name of `SAMPLER_METHOD`, now deprecated.
class KSAMPLER(SAMPLER_METHOD):
    '''
    The origin name of `KSAMPLER_METHOD` is `KSAMPLER` actually, but it makes the code a bit messy (since there are 2 `KSampler` classes in the same file),
    For long term consideration, I decided to change the name to `KSAMPLER_METHOD`(which act as a wrapper of a sampler function) here.
    
    For the sake of compatibility, `KSAMPLER` is still available as an alias of `KSAMPLER_METHOD`.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

def get_sampler_method(sampler_name, extra_options={}, inpaint_options={})->SAMPLER_METHOD:
    if sampler_name == "uni_pc":
        return SAMPLER_METHOD(uni_pc.sample_unipc)
    elif sampler_name == "uni_pc_bh2":
        return SAMPLER_METHOD(uni_pc.sample_unipc_bh2)
    elif sampler_name == "ddim":
        return get_sampler_method("euler", extra_options=extra_options, inpaint_options={"random": True})
    elif sampler_name == "dpm_fast":
        def dpm_fast_function(model, noise, sigmas, extra_args, callbacks, disable, timesteps, **kwargs):
            sigma_min = sigmas[-1]
            if sigma_min == 0:
                sigma_min = sigmas[-2]
            total_steps = len(sigmas) - 1
            return k_diffusion_sampling.sample_dpm_fast(model,
                                                        noise, 
                                                        sigma_min, 
                                                        sigmas[0], 
                                                        total_steps, 
                                                        extra_args=extra_args, 
                                                        callbacks=callbacks, 
                                                        disable=disable,
                                                        **kwargs)
        sampler_function = dpm_fast_function
    elif sampler_name == "dpm_adaptive":
        def dpm_adaptive_function(model, noise, sigmas, extra_args, callbacks, disable, **kwargs):
            sigma_min = sigmas[-1]
            if sigma_min == 0:
                sigma_min = sigmas[-2]
            return k_diffusion_sampling.sample_dpm_adaptive(model, noise, sigma_min, sigmas[0], extra_args=extra_args, callbacks=callbacks, disable=disable, **kwargs)
        sampler_function = dpm_adaptive_function
    else:
        sampler_function = getattr(k_diffusion_sampling, "sample_{}".format(sampler_name))

    return SAMPLER_METHOD(sampler_function, extra_options, inpaint_options)

@deprecated # `ksampler` is the old name of `get_sampler_method`, now deprecated.
def ksampler(sampler_name, extra_options={}, inpaint_options={}):
    '''
    Deprecated cuz the old one makes code very messy. 
    Use `get_ksampler_method` instead.
    '''
    return get_sampler_method(sampler_name, extra_options, inpaint_options)

@deprecated # `sampler_object` is the old name of `get_sampler_method`, now deprecated.
def sampler_object(name: str):
    '''!!DEPRECATED!! Use `get_ksampler_method` instead.'''
    return get_sampler_method(name)
# endregion


def sample(model: model_base.BaseModel,
           noise: torch.Tensor,
           positive: list["ConvertedCondition"],
           negative: list["ConvertedCondition"],
           cfg: float,
           device: str,
           sampler_method: SAMPLER_METHOD, # the real sampler method, e.g. sample_euler, usually under k_diffusion.py
           sigmas: torch.Tensor,
           model_options: Dict = {},
           latent_image: torch.Tensor = None,
           denoise_mask: Optional[torch.Tensor] = None,
           callbacks: Union["SamplerCallback", Sequence["SamplerCallback"], None] = None,
           disable_pbar: Optional[bool] = False,
           seed: Optional[int] = None,
           timesteps: List[int]=None,
           **kwargs):
    '''
    This function is the function after `KSampler.sample` is called.
    Seems it is split for sharing some common operations.
    '''
    
    positive = positive[:]
    negative = negative[:]

    resolve_areas_and_cond_masks(positive, noise.shape[2], noise.shape[3], device)
    resolve_areas_and_cond_masks(negative, noise.shape[2], noise.shape[3], device)

    model_wrap = CFGNoisePredictor(model)

    calculate_start_end_timesteps(model, negative)
    calculate_start_end_timesteps(model, positive)

    if latent_image is not None and torch.count_nonzero(latent_image) > 0: #Don't shift the empty latent image.
        latent_image = model.process_latent_in(latent_image)

    if hasattr(model, 'extra_conds'):
        positive = encode_model_conds(model.extra_conds, positive, noise, device, "positive", latent_image=latent_image, denoise_mask=denoise_mask, seed=seed)
        negative = encode_model_conds(model.extra_conds, negative, noise, device, "negative", latent_image=latent_image, denoise_mask=denoise_mask, seed=seed)

    if is_dev_mode() and is_verbose_mode():
        ComfyUILogger.debug(f"(comfy.samplers.sample) positive count={len(positive)}, negative count={len(negative)}")
    
    #make sure each cond area has an opposite one with the same area
    for c in positive:
        create_cond_with_same_area_if_none(negative, c)
    for c in negative:
        create_cond_with_same_area_if_none(positive, c)

    pre_run_control(model, negative + positive)
    
    apply_empty_x_to_equal_area(list(filter(lambda c: c.get('control_apply_to_uncond', False) == True, positive)), negative, 'control', lambda cond_cnets, x: cond_cnets[x])
    apply_empty_x_to_equal_area(positive, negative, 'gligen', lambda cond_cnets, x: cond_cnets[x])

    extra_args = {"cond":positive, "uncond":negative, "cond_scale": cfg, "model_options": model_options, "seed":seed}

    if is_engine_looping():
        disable_pbar = True
    samples = sampler_method.sample(model_wrap=model_wrap, 
                                    sigmas=sigmas, 
                                    extra_args=extra_args, 
                                    noise=noise, 
                                    callbacks=callbacks, 
                                    latent_image=latent_image, 
                                    denoise_mask=denoise_mask, 
                                    disable_pbar=disable_pbar,
                                    timesteps=timesteps,
                                    **kwargs)
    return model.process_latent_out(samples.to(torch.float32))



def calculate_sigmas_scheduler(model, scheduler_name, steps):
    if scheduler_name == "karras":
        sigmas = k_diffusion_sampling.get_sigmas_karras(n=steps, sigma_min=float(model.model_sampling.sigma_min), sigma_max=float(model.model_sampling.sigma_max))
    elif scheduler_name == "exponential":
        sigmas = k_diffusion_sampling.get_sigmas_exponential(n=steps, sigma_min=float(model.model_sampling.sigma_min), sigma_max=float(model.model_sampling.sigma_max))
    elif scheduler_name == "normal":
        sigmas = normal_scheduler(model, steps)
    elif scheduler_name == "simple":
        sigmas = simple_scheduler(model, steps)
    elif scheduler_name == "ddim_uniform":
        sigmas = ddim_scheduler(model, steps)
    elif scheduler_name == "sgm_uniform":
        sigmas = normal_scheduler(model, steps, sgm=True)
    else:
        ComfyUILogger.error("error invalid scheduler" + scheduler_name)
    return sigmas

class KSampler:
    SCHEDULERS = SCHEDULER_NAMES
    SAMPLERS = SAMPLER_NAMES
    DISCARD_PENULTIMATE_SIGMA_SAMPLERS = set(('dpm_2', 'dpm_2_ancestral', 'uni_pc', 'uni_pc_bh2'))

    def __init__(self,
                 model: model_base.BaseModel,
                 steps: int,
                 device: str,
                 sampler: str = None,
                 scheduler: str = None,
                 denoise: float = None,
                 model_options: dict = {}):
        self.model = model
        self.device = device
        if scheduler not in self.SCHEDULERS:
            scheduler = self.SCHEDULERS[0]
        if sampler not in self.SAMPLERS:
            sampler = self.SAMPLERS[0]
        self.scheduler = scheduler
        self.sampler_name = sampler
        self.set_steps(steps, denoise)
        self.denoise = denoise
        self.model_options = model_options

    def calculate_sigmas(self, steps: int) -> Tuple[torch.Tensor, List[int]]:
        sigmas = None

        discard_penultimate_sigma = False
        if self.sampler_name in self.DISCARD_PENULTIMATE_SIGMA_SAMPLERS:
            steps += 1
            discard_penultimate_sigma = True

        sigmas = calculate_sigmas_scheduler(self.model, self.scheduler, steps)
        s: "ModelSamplingProtocol" = self.model.model_sampling
        timesteps = [s.timestep(sigma) for sigma in sigmas]
        if discard_penultimate_sigma:
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
        return sigmas, timesteps    # type: ignore

    def set_steps(self, steps: int, denoise: float = None) -> None:
        self.steps = steps
        if denoise is None or denoise > 0.9999:
            self.sigmas, self.timesteps = self.calculate_sigmas(steps)
            self.sigmas = self.sigmas.to(self.device)
        else:
            new_steps = int(steps/denoise)
            sigmas, self.timesteps = self.calculate_sigmas(new_steps)
            sigmas = sigmas.to(self.device)
            self.sigmas = sigmas[-(steps + 1):]

    def sample(self,
               noise: torch.Tensor,
               positive: list["ConvertedCondition"],
               negative: list["ConvertedCondition"],
               cfg: float,
               latent_image: Optional[torch.Tensor] = None,
               start_step: Optional[int] = None,
               last_step: Optional[int] = None,
               force_full_denoise: bool = False,
               denoise_mask: Optional[torch.Tensor] = None,
               sigmas: Optional[torch.Tensor] = None,
               callbacks: Union["SamplerCallback", Sequence["SamplerCallback"], None] = None,
               disable_pbar: bool = False,
               seed: Optional[int] = None,
               **kwargs) -> torch.Tensor:
        """
        Samples the model using the provided inputs.

        Args:
            noise (torch.Tensor): The input noise tensor.
            positive (list[ConvertedCondition]): The converted positive condition.
            negative (list[ConvertedCondition]): The converted negative condition.
            cfg (float): Classifier Free Guidance value.
            latent_image (Optional[torch.Tensor], optional): The latent image tensor. Defaults to None.
            start_step (Optional[int], optional): The starting step. Defaults to None.
            last_step (Optional[int], optional): The last step. Defaults to None.
            force_full_denoise (bool, optional): Flag to force full denoising. Defaults to False.
            denoise_mask (Optional[torch.Tensor], optional): The denoise mask tensor. Defaults to None.
            sigmas (Optional[torch.Tensor], optional): The sigma tensor calculated according to the scheduler.
                It is a tensor of shape (steps + 1,) with strictly decreasing values. Defaults to None.
            callbacks (List[Callable]): The callback function. Defaults to [].
            disable_pbar (bool, optional): Flag to disable progress bar. Defaults to False.
            seed (Optional[int], optional): The random seed. Defaults to None.

            **kwargs: This args will finally passing through all layers of models (in case it is not being removed by any steps)
        Returns:
            torch.Tensor: The sampled tensor.
        """
        if sigmas is None:
            sigmas = self.sigmas

        if last_step is not None and last_step < (len(sigmas) - 1):
            sigmas = sigmas[:last_step + 1]
            if force_full_denoise:
                sigmas[-1] = 0

        if start_step is not None:
            if start_step < (len(sigmas) - 1):
                sigmas = sigmas[start_step:]
            else:
                if latent_image is not None:
                    return latent_image
                else:
                    return torch.zeros_like(noise)
        sampler = sampler_object(self.sampler_name)
        
        if len(noise.shape) == 3:   # no batch channel, add it
            noise = noise.unsqueeze(0)
        return sample(self.model, 
                      noise, 
                      positive, 
                      negative, 
                      cfg, 
                      self.device, 
                      sampler, 
                      sigmas, 
                      self.model_options, 
                      latent_image=latent_image, 
                      denoise_mask=denoise_mask, 
                      callbacks=callbacks, 
                      disable_pbar=disable_pbar, 
                      seed=seed,
                      timesteps=self.timesteps,
                      **kwargs)
