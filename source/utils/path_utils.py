import os, datetime
from utils.decorator.overload import Overload

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
@Overload
def get_map_output_dir():
    '''return a dir under MAP_OUTPUT_DIR with current time as its name'''
    count = 0
    cur_time = datetime.datetime.now().strftime('%Y-%m-%d') + f'_{count}'
    while os.path.exists(os.path.join(MAP_OUTPUT_DIR, cur_time)):
        count += 1
        cur_time = datetime.datetime.now().strftime('%Y-%m-%d') + f'_{count}'
    cur_map_output_dir = os.path.join(MAP_OUTPUT_DIR, cur_time)
    os.makedirs(cur_map_output_dir, exist_ok=True)
    return cur_map_output_dir
@Overload
def get_map_output_dir(day:int, index:int, month:int=None, year:int=None):
    '''
    Return a subdir under MAP_OUTPUT_DIR with yout specified time.
    If month or year is not specified, use current month or year.
    If no such subdir, raise FileNotFoundError.
    '''
    if month is None:
        month = datetime.datetime.now().strftime('%m')
    if year is None:
        year = datetime.datetime.now().strftime('%Y')
    cur_subdir = os.path.join(MAP_OUTPUT_DIR, f'{year}-{month}-{day}_{index}')
    if not os.path.exists(cur_subdir):
        raise FileNotFoundError(f'No such subdir: {cur_subdir}')
    return cur_subdir



__all__ = ['RESOURCES_DIR', 'SHADER_DIR', 'DEFAULT_QUAD_VS_SHADER_PATH', 'DEFAULT_QUAD_FS_SHADER_PATH', 'OUTPUT_DIR', 'MAP_OUTPUT_DIR', 'get_map_output_dir']