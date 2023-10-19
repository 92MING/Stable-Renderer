import os, datetime

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

RESOURCES_DIR = os.path.join(PROJECT_DIR, 'resources')
'''resources directory, for obj, shader, texture, etc.'''
SHADER_DIR = os.path.join(RESOURCES_DIR, 'shaders')
'''Default shader directory'''
DEFAULT_QUAD_VS_SHADER_PATH = os.path.join(SHADER_DIR, 'quad_vs.glsl')
'''Default quad vertex shader path'''
DEFAULT_QUAD_FS_SHADER_PATH = os.path.join(SHADER_DIR, 'quad_fs.glsl')
'''Default quad fragment shader path'''

OUTPUT_DIR = os.path.join(PROJECT_DIR, 'output')
'''output directory, for runtime map, etc.'''
MAP_OUTPUT_DIR = os.path.join(OUTPUT_DIR, 'runtime_map')
'''runtime map output directory, for saving normal map, pos map, id map, etc., during runtime.'''
def get_map_output_dir():
    '''return a dir under MAP_OUTPUT_DIR with current time as its name'''
    cur_time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    cur_map_output_dir = os.path.join(MAP_OUTPUT_DIR, cur_time)
    os.makedirs(cur_map_output_dir, exist_ok=True)
    return cur_map_output_dir

__all__ = ['RESOURCES_DIR', 'SHADER_DIR', 'DEFAULT_QUAD_VS_SHADER_PATH', 'DEFAULT_QUAD_FS_SHADER_PATH', 'OUTPUT_DIR', 'MAP_OUTPUT_DIR', 'get_map_output_dir']