import torch
import collections
import math

from inspect import signature
from abc import ABC, abstractmethod
from deprecated import deprecated
from typing import (List, Dict, Any, Tuple, Optional, Callable, Literal, TYPE_CHECKING, Sequence, Union)
from types import MethodType
from functools import partial

from common_utils.debug_utils import ComfyUILogger
from common_utils.global_utils import GetOrCreateGlobalValue
from .k_diffusion import sampling as k_diffusion_sampling
from .extra_samplers import uni_pc
from comfy import model_management, model_base

if TYPE_CHECKING:
    from comfyUI.types import ConvertedConditioning, SamplerCallback


def get_area_and_mult(conds, x_in, timestep_in):
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
        conditioning[c] = model_conds[c].process_cond(batch_size=x_in.shape[0], device=x_in.device, area=area)

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

    cond_obj = collections.namedtuple('cond_obj', ['input_x', 'mult', 'conditioning', 'area', 'control', 'patches'])
    return cond_obj(input_x, mult, conditioning, area, control, patches)

def cond_equal_size(c1, c2):
    if c1 is c2:
        return True
    if c1.keys() != c2.keys():
        return False
    for k in c1:
        if not c1[k].can_concat(c2[k]):
            return False
    return True

def can_concat_cond(c1, c2):
    if c1.input_x.shape != c2.input_x.shape:
        return False

    def objects_concatable(obj1, obj2):
        if (obj1 is None) != (obj2 is None):
            return False
        if obj1 is not None:
            if obj1 is not obj2:
                return False
        return True

    if not objects_concatable(c1.control, c2.control):
        return False

    if not objects_concatable(c1.patches, c2.patches):
        return False

    return cond_equal_size(c1.conditioning, c2.conditioning)

def cond_cat(c_list):
    c_crossattn = []
    c_concat = []
    c_adm = []
    crossattn_max_len = 0

    temp = {}
    for x in c_list:
        for k in x:
            cur = temp.get(k, [])
            cur.append(x[k])
            temp[k] = cur

    out = {}
    for k in temp:
        conds = temp[k]
        out[k] = conds[0].concat(conds[1:])

    return out

def calc_cond_uncond_batch(model, cond, uncond, x_in, timestep, model_options):
    out_cond = torch.zeros_like(x_in)
    out_count = torch.ones_like(x_in) * 1e-37

    out_uncond = torch.zeros_like(x_in)
    out_uncond_count = torch.ones_like(x_in) * 1e-37

    COND = 0
    UNCOND = 1

    to_run = []
    for x in cond:
        p = get_area_and_mult(x, x_in, timestep)
        if p is None:
            continue

        to_run += [(p, COND)]
    if uncond is not None:
        for x in uncond:
            p = get_area_and_mult(x, x_in, timestep)
            if p is None:
                continue

            to_run += [(p, UNCOND)]

    while len(to_run) > 0:
        first = to_run[0]
        first_shape = first[0][0].shape
        to_batch_temp = []
        for x in range(len(to_run)):
            if can_concat_cond(to_run[x][0], first[0]):
                to_batch_temp += [x]

        to_batch_temp.reverse()
        to_batch = to_batch_temp[:1]

        free_memory = model_management.get_free_memory(x_in.device)
        for i in range(1, len(to_batch_temp) + 1):
            batch_amount = to_batch_temp[:len(to_batch_temp)//i]
            input_shape = [len(batch_amount) * first_shape[0]] + list(first_shape)[1:]
            if model.memory_required(input_shape) < free_memory:
                to_batch = batch_amount
                break

        input_x = []
        mult = []
        c = []
        cond_or_uncond = []
        area = []
        control = None
        patches = None
        for x in to_batch:
            o = to_run.pop(x)
            p = o[0]
            input_x.append(p.input_x)
            mult.append(p.mult)
            c.append(p.conditioning)
            area.append(p.area)
            cond_or_uncond.append(o[1])
            control = p.control
            patches = p.patches

        batch_chunks = len(cond_or_uncond)
        input_x = torch.cat(input_x)
        c = cond_cat(c)
        timestep_ = torch.cat([timestep] * batch_chunks)

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
        transformer_options["sigmas"] = timestep

        c['transformer_options'] = transformer_options

        if 'model_function_wrapper' in model_options:
            output = model_options['model_function_wrapper'](model.apply_model, {"input": input_x, "timestep": timestep_, "c": c, "cond_or_uncond": cond_or_uncond}).chunk(batch_chunks)
        else:
            output = model.apply_model(input_x, timestep_, **c).chunk(batch_chunks)
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

def sampling_function(model: model_base.BaseModel,
                      x, 
                      timestep, 
                      uncond, 
                      cond, 
                      cond_scale, 
                      model_options={}, 
                      seed=None):
    '''
    The main sampling function shared by all the samplers.
    Returns denoised image.
    '''
    if math.isclose(cond_scale, 1.0) and model_options.get("disable_cfg1_optimization", False) == False:
        uncond_ = None
    else:
        uncond_ = uncond

    cond_pred, uncond_pred = calc_cond_uncond_batch(model, cond, uncond_, x, timestep, model_options)
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

class CFGNoisePredictor(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.inner_model = model
        
    def apply_model(self, x, timestep, cond, uncond, cond_scale, model_options={}, seed=None):
        out = sampling_function(self.inner_model, 
                                x, 
                                timestep, 
                                uncond, 
                                cond, 
                                cond_scale, 
                                model_options=model_options, 
                                seed=seed)
        return out
    
    def forward(self, *args, **kwargs):
        return self.apply_model(*args, **kwargs)

class KSamplerX0Inpaint(torch.nn.Module):
    def __init__(self,
                 model: CFGNoisePredictor,
                 sigmas: torch.Tensor):
        super().__init__()
        self.inner_model = model
        self.sigmas = sigmas
        
    def forward(self, x, sigma, uncond, cond, cond_scale, denoise_mask, model_options={}, seed=None):
        if denoise_mask is not None:
            if "denoise_mask_function" in model_options:
                denoise_mask = model_options["denoise_mask_function"](sigma, denoise_mask, extra_options={"model": self.inner_model, "sigmas": self.sigmas})
            latent_mask = 1. - denoise_mask
            x = x * denoise_mask + self.inner_model.inner_model.model_sampling.noise_scaling(sigma.reshape([sigma.shape[0]] + [1] * (len(self.noise.shape) - 1)), self.noise, self.latent_image) * latent_mask
        out = self.inner_model(x, sigma, cond=cond, uncond=uncond, cond_scale=cond_scale, model_options=model_options, seed=seed)
        if denoise_mask is not None:
            out = out * denoise_mask + self.latent_image * latent_mask
        return out

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
        conditions: "ConvertedConditioning", h: int, w: int, device: str
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
                bounds = torch.max(torch.abs(mask),dim=0).values.unsqueeze(0)
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
        conds: "ConvertedConditioning"):
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

def pre_run_control(model: model_base.BaseModel, conds: "ConvertedConditioning"):
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
                       conds: "ConvertedConditioning",
                       noise: torch.Tensor,
                       device: str,
                       prompt_type: Literal["positive", "negative"],
                       **kwargs):
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
    
    return conds

class Sampler(ABC):
    '''Base class of samplers.'''
    
    @abstractmethod
    def sample(self):
        raise NotImplementedError

    def max_denoise(self, model_wrap, sigmas):
        max_sigma = float(model_wrap.inner_model.model_sampling.sigma_max)
        sigma = float(sigmas[0])
        return math.isclose(max_sigma, sigma, rel_tol=1e-05) or sigma > max_sigma

KSAMPLER_NAMES = ["euler", "euler_ancestral", "heun", "heunpp2","dpm_2", "dpm_2_ancestral",
                  "lms", "dpm_fast", "dpm_adaptive", "dpmpp_2s_ancestral", "dpmpp_sde", "dpmpp_sde_gpu",
                  "dpmpp_2m", "dpmpp_2m_sde", "dpmpp_2m_sde_gpu", "dpmpp_3m_sde", "dpmpp_3m_sde_gpu", "ddpm", "lcm",]
'''all possible ksampler types'''

SAMPLER_NAMES = KSAMPLER_NAMES + ["ddim", "uni_pc", "uni_pc_bh2"]
'''all possible sampler types. This is the full list of samplers that can be used in the UI.'''

SCHEDULER_NAMES = ["normal", "karras", "exponential", "sgm_uniform", "simple", "ddim_uniform"]
'''all possible schedulers'''

_method_param_conut_cache: Dict[int, int] = GetOrCreateGlobalValue('__COMFY_SAMPLER_METHOD_PARAM_COUNT_CACHE__', dict)
def _get_method_param_count(method: Callable) -> int:
    method_id = id(method)
    if method_id in _method_param_conut_cache:
        return _method_param_conut_cache[method_id]
    else:
        if hasattr(method, '__code__'):
            param_count = method.__code__.co_argcount
        else:
            param_count = len(signature(method).parameters)
        if isinstance(method, MethodType):
            param_count -= 1
        _method_param_conut_cache[method_id] = param_count
        return param_count

def _call_context_callback(callback, total_steps, dict_data_from_upper: dict):
    from comfyUI.types import SamplingCallbackContext
    context = SamplingCallbackContext(dict_data_from_upper.pop("i"), 
                                      dict_data_from_upper.pop("denoised"), 
                                      dict_data_from_upper.pop("x"),
                                      total_steps)
    for key, val in dict_data_from_upper.items():
        setattr(context, key, val)
    return callback(context)

class KSAMPLER_METHOD(Sampler):
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
               disable_pbar: bool = False):
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
                    k_callbacks.append(lambda x: callback())
                elif callback_arg_count == 1: # pass context
                    k_callbacks.append(partial(_call_context_callback, callback, total_steps))
                elif callback_arg_count == 4:   # pass (i, denoised, x, total_steps)
                    k_callbacks.append(lambda x: callback(x["i"], x["denoised"], x["x"], total_steps))
                
        samples = self.sampler_function(model_k, noise, sigmas, extra_args=extra_args, callbacks=k_callbacks, disable=disable_pbar, **self.extra_options)
        return samples

@deprecated # `KSAMPLER` is the old name of `KSAMPLER_METHOD`, now deprecated.
class KSAMPLER(KSAMPLER_METHOD):
    '''
    The origin name of `KSAMPLER_METHOD` is `KSAMPLER` actually, but it makes things very messy.
    For the sake of compatibility, `KSAMPLER` is still available as an alias of `KSAMPLER_METHOD`.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

def get_ksampler_method(sampler_name, extra_options={}, inpaint_options={})->KSAMPLER_METHOD:
    if sampler_name == "uni_pc":
        return KSAMPLER_METHOD(uni_pc.sample_unipc)
    elif sampler_name == "uni_pc_bh2":
        return KSAMPLER_METHOD(uni_pc.sample_unipc_bh2)
    elif sampler_name == "ddim":
        return get_ksampler_method("euler", inpaint_options={"random": True})
    elif sampler_name == "dpm_fast":
        def dpm_fast_function(model, noise, sigmas, extra_args, callbacks, disable):
            sigma_min = sigmas[-1]
            if sigma_min == 0:
                sigma_min = sigmas[-2]
            total_steps = len(sigmas) - 1
            return k_diffusion_sampling.sample_dpm_fast(model, noise, sigma_min, sigmas[0], total_steps, extra_args=extra_args, callbacks=callbacks, disable=disable)
        sampler_function = dpm_fast_function
    elif sampler_name == "dpm_adaptive":
        def dpm_adaptive_function(model, noise, sigmas, extra_args, callbacks, disable):
            sigma_min = sigmas[-1]
            if sigma_min == 0:
                sigma_min = sigmas[-2]
            return k_diffusion_sampling.sample_dpm_adaptive(model, noise, sigma_min, sigmas[0], extra_args=extra_args, callbacks=callbacks, disable=disable)
        sampler_function = dpm_adaptive_function
    else:
        sampler_function = getattr(k_diffusion_sampling, "sample_{}".format(sampler_name))

    return KSAMPLER_METHOD(sampler_function, extra_options, inpaint_options)

@deprecated # `ksampler` is the old name of `get_ksampler_method`, now deprecated.
def ksampler(sampler_name, extra_options={}, inpaint_options={}):
    '''
    Deprecated cuz the old one makes code very messy. 
    Use `get_ksampler_method` instead.
    '''
    return get_ksampler_method(sampler_name, extra_options, inpaint_options)

@deprecated
def sampler_object(name: str):
    '''!!DEPRECATED!! Use `get_ksampler_method` instead.'''
    return get_ksampler_method(name)


def wrap_model(model: model_base.BaseModel) -> CFGNoisePredictor:
    model_denoise = CFGNoisePredictor(model)
    return model_denoise

def sample(model: model_base.BaseModel,
           noise: torch.Tensor,
           positive: "ConvertedConditioning",
           negative: "ConvertedConditioning",
           cfg: float,
           device: str,
           sampler: KSAMPLER_METHOD,
           sigmas: torch.Tensor,
           model_options: Dict = {},
           latent_image: torch.Tensor = None,
           denoise_mask: Optional[torch.Tensor] = None,
           callbacks: Union["SamplerCallback", Sequence["SamplerCallback"], None] = None,
           disable_pbar: Optional[bool] = False,
           seed: Optional[int] = None,
           **kwargs):

    positive = positive[:]
    negative = negative[:]

    resolve_areas_and_cond_masks(positive, noise.shape[2], noise.shape[3], device)
    resolve_areas_and_cond_masks(negative, noise.shape[2], noise.shape[3], device)

    model_wrap = wrap_model(model)

    calculate_start_end_timesteps(model, negative)
    calculate_start_end_timesteps(model, positive)

    if latent_image is not None and torch.count_nonzero(latent_image) > 0: #Don't shift the empty latent image.
        latent_image = model.process_latent_in(latent_image)

    if hasattr(model, 'extra_conds'):
        positive = encode_model_conds(model.extra_conds, positive, noise, device, "positive", latent_image=latent_image, denoise_mask=denoise_mask, seed=seed)
        negative = encode_model_conds(model.extra_conds, negative, noise, device, "negative", latent_image=latent_image, denoise_mask=denoise_mask, seed=seed)

    #make sure each cond area has an opposite one with the same area
    for c in positive:
        create_cond_with_same_area_if_none(negative, c)
    for c in negative:
        create_cond_with_same_area_if_none(positive, c)

    pre_run_control(model, negative + positive)

    apply_empty_x_to_equal_area(list(filter(lambda c: c.get('control_apply_to_uncond', False) == True, positive)), negative, 'control', lambda cond_cnets, x: cond_cnets[x])
    apply_empty_x_to_equal_area(positive, negative, 'gligen', lambda cond_cnets, x: cond_cnets[x])

    extra_args = {"cond":positive, "uncond":negative, "cond_scale": cfg, "model_options": model_options, "seed":seed}

    samples = sampler.sample(model_wrap, sigmas, extra_args, noise, callbacks, latent_image, denoise_mask, disable_pbar)
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

    def calculate_sigmas(self, steps: int) -> torch.Tensor:
        sigmas = None

        discard_penultimate_sigma = False
        if self.sampler_name in self.DISCARD_PENULTIMATE_SIGMA_SAMPLERS:
            steps += 1
            discard_penultimate_sigma = True

        sigmas = calculate_sigmas_scheduler(self.model, self.scheduler, steps)

        if discard_penultimate_sigma:
            sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
        return sigmas

    def set_steps(self, steps: int, denoise: float = None) -> None:
        self.steps = steps
        if denoise is None or denoise > 0.9999:
            self.sigmas = self.calculate_sigmas(steps).to(self.device)
        else:
            new_steps = int(steps/denoise)
            sigmas = self.calculate_sigmas(new_steps).to(self.device)
            self.sigmas = sigmas[-(steps + 1):]

    def sample(self,
               noise: torch.Tensor,
               positive: "ConvertedConditioning",
               negative: "ConvertedConditioning",
               cfg: float,
               latent_image: Optional[torch.Tensor] = None,
               start_step: Optional[int] = None,
               last_step: Optional[int] = None,
               force_full_denoise: bool = False,
               denoise_mask: Optional[torch.Tensor] = None,
               sigmas: Optional[torch.Tensor] = None,
               callbacks: Union["SamplerCallback", Sequence["SamplerCallback"], None] = None,
               disable_pbar: bool = False,
               seed: Optional[int] = None) -> torch.Tensor:
        """
        Samples the model using the provided inputs.

        Args:
            noise (torch.Tensor): The input noise tensor.
            positive (ConvertedConditioning): The converted positive condition.
            negative (ConvertedConditioning): The converted negative condition.
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
                      seed=seed)
