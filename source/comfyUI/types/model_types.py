'''Type hints & data structs specific to model related stuffs.'''

from typing import (Any, Optional, Type, Union, Protocol, TYPE_CHECKING, TypeAlias)
from common_utils.type_utils import get_cls_name, valueTypeCheck

if TYPE_CHECKING:
    from comfy.sd1_clip import SDClipModel, SDTokenizer, SD1ClipModel, SD1Tokenizer
    from comfy.sdxl_clip import SDXLTokenizer, SDXLClipModel
    from comfy.clip_model import CLIPVisionModelProjection
    from comfy.gligen import Gligen
    from comfy.model_base import BaseModel, SD21UNCLIP

ClipModelType: TypeAlias = Union["SDClipModel", "SD1ClipModel", "SDXLClipModel"] # some other types like SD2ClipModel are inherited from these types, so no need to include them.
'''Base types for ClipModel. '''
TokenizerType: TypeAlias = Union["SDTokenizer", "SD1Tokenizer", "SDXLTokenizer"] # some other types like SD2Tokenizer are inherited from these types, so no need to include them.
'''Base types for Tokenizer. '''
ModelPatcherTypes: TypeAlias = Union['BaseModel', 'SD21UNCLIP', 'SDClipModel', 'SD1ClipModel', 'SDXLClipModel', 'CLIPVisionModelProjection', 'Gligen']
'''types that acceptable for `ModelPatcher` class's first argument `model`. '''


class ClipTargetProtocol(Protocol):
    '''type hints for `ClipTarget` class. '''
    params: dict
    clip: Type[ClipModelType]
    tokenizer: Type[TokenizerType]


class ModelLike:
    
    def __init__(self,
                 name: Optional[str] = None, 
                 identifier: Optional[Any]=None):
        '''
        Inherit from this class to create a model-like object.
        This type is designed for fixing bugs/ better debugging/ advanced features(which the original class lacks).
        
        When doing inheritance, make sure to call the super().__init__ method in the child class, and put `ModelLike` as the first parent class,
        e.g. class A(ModelLike, B, C): ...
        
        Args:
            - name (Optional[str]): The name of the model.
            - identifier (Optional[Any]): The identifier of the model. Its for comparison purposes when __eq__ is called.
        '''
        self._name = name or f'Unknown {self.__class__.__qualname__}'
        self._identifier = identifier or id(self)
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def identifier(self) -> Any:
        return self._identifier
    
    def __eq__(self, value: Any):
        '''
        If both objects have the same identifier(in case both of them have and not None), they are equal.
        Otherwise, they are not equal.
        '''
        result = super().__eq__(value)
        if not result:
            if hasattr(self, 'identifier') and hasattr(value, 'identifier'):
                if get_cls_name(self) == get_cls_name(value):
                    if self.identifier is not None and value.identifier is not None:
                        result = self.identifier == value.identifier
        return result

def is_model_type(model):
    from comfy.model_patcher import ModelPatcher
    from comfy.model_management import LoadedModel
    from comfy.clip_model import CLIPVisionModelProjection
    from comfy.gligen import Gligen
    from comfy.model_base import BaseModel, SD21UNCLIP
    from comfy.sd1_clip import SDClipModel, SD1ClipModel
    return valueTypeCheck(model, 
                          (ModelPatcher, LoadedModel, CLIPVisionModelProjection, Gligen, 
                                BaseModel, SD21UNCLIP, SDClipModel, SD1ClipModel))    # type: ignore


__all__ = ['ClipModelType', 'TokenizerType', 'ModelPatcherTypes', 'ModelLike', 'ClipTargetProtocol', 'is_model_type']