import logging

EngineEditorLogger = logging.getLogger("Engine.Editor")
'''Logger which is active during editor mode. Will be ignore in release mode.'''

EngineRuntimeLogger = logging.getLogger("Engine.Runtime")
'''Logger is active during runtime. It will not be ignore in any mode.'''

__all__ = ['EngineEditorLogger', 'EngineRuntimeLogger']