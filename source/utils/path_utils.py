import os

RESOURCES_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'resources'))
'''resources directory, for obj, shader, texture, etc.'''
SHADER_DIR = os.path.join(RESOURCES_DIR, 'shaders')
'''Default shader directory'''
DEFAULT_QUAD_VS_SHADER_PATH = os.path.join(SHADER_DIR, 'quad_vs.glsl')
'''Default quad vertex shader path'''
DEFAULT_QUAD_FS_SHADER_PATH = os.path.join(SHADER_DIR, 'quad_fs.glsl')
'''Default quad fragment shader path'''

__all__ = ['RESOURCES_DIR', 'SHADER_DIR', 'DEFAULT_QUAD_VS_SHADER_PATH', 'DEFAULT_QUAD_FS_SHADER_PATH']