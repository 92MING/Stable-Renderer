from comfyUI.types import HIDDEN

class EngineFrameData(HIDDEN):
    '''The prompt for submitting to ComfyUI during engine's runtime.'''
    
    @classmethod
    def GetValue(cls, context):
        return context.engine_frame_data