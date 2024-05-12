from attr import attrib, attrs

@attrs
class EnvPrompt:
    '''Prompt for the environment.'''
    
    prompt: str = attrib(default="")
    '''Prompt for the empty area, i.e. area without any object.'''
    
    negative_prompt: str = attrib(default="")
    '''Prompt for the negative area, i.e. area with negative objects.'''
    
    weight: float = attrib(default=1.0)
    '''Weight of the prompt. [0, 1]'''
    
    negative_weight: float = attrib(default=1.0)
    '''Weight of the negative prompt. [0, 1]'''
    
__all__ = ['EnvPrompt']