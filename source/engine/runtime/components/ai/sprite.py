from typing import TYPE_CHECKING, overload

from common_utils.debug_utils import EngineLogger
from common_utils.stable_render_utils import Sprite, get_new_spriteID
from ...component import Component

if TYPE_CHECKING:
    from ...gameObj import GameObject
    

class SpriteInfo(Component):
    '''The component for containing the sprite info.'''
    
    sprite: Sprite | None
    '''the sprite object.'''

    @overload
    def __init__(self, gameObj: "GameObject", enable=True, *, sprite: Sprite|None=None):...
    @overload
    def __init__(self, gameObj: "GameObject", enable=True, *, spriteID: int|None=None, prompt: str='', auto_spriteID=False):...

    def __init__(self, 
                 gameObj: "GameObject", 
                 enable=True,
                 **kwargs,
                 ):
        super().__init__(gameObj, enable)
        if 'sprite' in kwargs:
            self.sprite = kwargs['sprite']
            if self.sprite is not None and not isinstance(self.sprite, Sprite):
                EngineLogger.warn(f'`sprite` should be a Sprite object, not {type(self.sprite)}.')
                self.sprite = None
                
        elif 'spriteID' in kwargs or 'auto_spriteID' in kwargs:
            spriteID = kwargs.get('spriteID', None)
            if spriteID is None:
                if kwargs.get('auto_spriteID', False):
                    spriteID = get_new_spriteID()
            if spriteID is not None:
                self.sprite = Sprite(spriteID=spriteID, prompt=kwargs.get('prompt', ''))
            
    def update(self):
        if self.sprite is not None:
            self.engine.RenderManager.SubmitSprite(self.sprite)



__all__ = ['SpriteInfo']