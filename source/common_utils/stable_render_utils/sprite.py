from attr import attrib, attrs
from ..global_utils import GetOrAddGlobalValue, SetGlobalValue


def get_new_spriteID()->int:
    '''
    Get a new unique sprite id for passing to shader.
    Please note that id 0 is reserved for the meaning of `no sprite`.
    '''
    cur_id: int = GetOrAddGlobalValue('__SPRITE_ID_COUNTER__', 1)   # type: ignore
    SetGlobalValue('__SPRITE_ID_COUNTER__', cur_id + 1)
    return cur_id

@attrs
class Sprite:
    '''Sprite is a basic unit representing an AI character, prompt, id, ..., is included in the sprite object.'''

    spriteID: int = attrib()
    '''
    ID of sprite.
    Note that id must be unique integer for transferring through textures & tensors. 
    '''
    
    prompt: str = attrib(default="")
    '''The prompt of the sprite.'''
    prompt_weight: float = attrib(default=1.0)
    '''The weight of the prompt.'''


class SpriteInfos(dict[int, Sprite]):
    '''
    Sprites is a dictionary of sprite id to sprite object. This type is for transferring data in node system.
    Sprite is a basic unit representing an AI character, prompt, id, ..., is included in the sprite object.
    '''


    
__all__ = ['get_new_spriteID', 'Sprite', 'SpriteInfos']