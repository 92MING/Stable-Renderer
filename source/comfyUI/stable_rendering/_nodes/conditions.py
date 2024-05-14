import torch
from typing import TYPE_CHECKING, Optional, Literal
from comfyUI.types import *
from common_utils.global_utils import is_dev_mode, is_verbose_mode
from common_utils.debug_utils import ComfyUILogger

if TYPE_CHECKING:
    from comfyUI.comfy.sd import CLIP
    from common_utils.stable_render_utils import SpriteInfos, Sprite
    from engine.static.corrmap import IDMap

def _set_cond_strength(cond, strength: float=1.0):
    cond[1]['strength'] = strength

def _text_encode(clip: 'CLIP', text: str, weight: float=1.0):
    tokens = clip.tokenize(text)
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    ret_cond = [cond, {"pooled_output": pooled}]
    if weight == 1.0:
        return ret_cond
    else:
        return _set_cond_strength(ret_cond, weight)

def _mask_text_encode(clip: 'CLIP', 
                      text= "", 
                      mask: Optional["MASK"]=None, 
                      inverse_mask: bool = False,
                      strength: float=1.0,
                      mode: Literal['default', 'set_cond_area']='default'):
    tokens = clip.tokenize(text)
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    
    if mask is None:
        return _text_encode(clip, text)
    else:
        set_area_to_bounds = False
        if mode == "set_cond_area":
            set_area_to_bounds = True
        
        if inverse_mask:
            mask = 1 - mask
        if len(mask.shape) < 3:
            mask = mask.unsqueeze(0)
            
        cond_data = {
            'pooled_output': pooled,
            'mask': mask,
            'set_area_to_bounds': set_area_to_bounds,
            'mask_strength': strength,
        }
        return [cond, cond_data]

class MaskedTextEncode(StableRenderingNode):
    
    Category = "conditioning"

    def __call__(self, 
                 clip: 'CLIP', 
                 text: STRING(multiline=True) = "", # type: ignore
                 mask: Optional["MASK"]=None,
                 inverse_mask: bool = False,
                 strength: float=1.0,
                 mode: Literal['default', 'set_cond_area']='default')->"CONDITIONING":
        '''
        Same as comfyUI's `CLIPTextEncode` node, but with an additional mask field in the conditioning.
        This is equivalent to the combination of using `ConditioningSetMask` together with `CLIPTextEncode`.
        
        Args:
            - clip: CLIP model
            - text: Text to encode
            - mask: Mask to apply to the text. If None, no mask is applied.
            - inverse_mask: By default, value=0 in mask means apply(i.e. transparent area). If inverse_mask=True, value=1 in mask means apply.
            - strength: Strength of the mask. Default is 1.0.
            - mode: If 'default', the mask is applied to the entire text. If 'set_cond_area', the real mask is the box area defined by the mask bounds.
        '''
        
        return [_mask_text_encode(clip, text, mask, inverse_mask, strength, mode),] # type: ignore

class SceneTextEncode(StableRenderingNode):
    '''
    Nodes for encoding all objects within the current scene, e.g. sprite, background, ...
    This is for the composition of baked result and environment.
    If you don't want area masking, plz leave `IDMap` as None.
    
    Note: if `IDMap` is given, only 1 frame is allowed as input(since it is not for baking).
    '''
    
    Category = "conditioning"
    
    def _get_sprite_mask(self, sprite: "Sprite", idmap: "IDMap")->Optional["MASK"]:
        sprite_id = sprite.spriteID
        if len(idmap) >1:
            if is_dev_mode():
                ComfyUILogger.warn("The idmap has more than 1 element, which is not expected.")
            idmap_tensor = idmap.tensor[0] if len(idmap.tensor.shape)==4 else idmap.tensor
        else:
            idmap_tensor = idmap.tensor
        mask = torch.zeros(1, *idmap_tensor.shape[-3:-1], dtype=torch.float32) # 0 means exclude area
        mask[idmap_tensor[..., 0] == sprite_id] = 1 # 1 means include area
        return mask
    
    def __call__(self, 
                 clip: 'CLIP',
                 sprite_infos: "SpriteInfos", 
                 env_prompts: Optional[EnvPrompts]=None, 
                 idmap: Optional["IDMap"]=None
                 )->tuple[
                     Named["CONDITIONING", 'pos'],  # type: ignore
                     Named["CONDITIONING", 'neg'],  # type: ignore
                 ]:
        conds = []
        neg_conds = []
        
        for sprite in sprite_infos.values():
            # when idmap is None, do normal prompt encode (means all area is included).
            if idmap is None:
                conds += _text_encode(clip, sprite.prompt, sprite.weight)   # type: ignore
            else:
                mask = self._get_sprite_mask(sprite, idmap)
                if mask is not None:    
                    # when the sprite is not occurred in the scene, mask is None.
                    conds.append(_mask_text_encode(clip, sprite.prompt))
        
        if env_prompts:
            for env_prompt in env_prompts:
                if env_prompt.prompt is not None:
                    conds.append(_text_encode(clip=clip, text=env_prompt.prompt, weight=env_prompt.weight))
                if env_prompt.negative_prompt is not None:  # empty string will also do the encoding, since ComfyUI must have a neg prompt.
                    neg_conds.append(_text_encode(clip=clip, text=env_prompt.negative_prompt, weight=env_prompt.negative_weight))
        
        if is_dev_mode() and is_verbose_mode():
            ComfyUILogger.print('SceneTextEncode finished. conds count={}, neg_conds count={}'.format(len(conds), len(neg_conds)))
        
        return conds, neg_conds



__all__ = ['MaskedTextEncode', 'SceneTextEncode']