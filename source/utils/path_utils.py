import os, datetime

from typing import Optional
from pathlib import Path



PROJECT_DIR = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
'''project directory, the root directory of the project.'''

SOURCE_DIR = PROJECT_DIR / 'source'
'''source directory, for all source code.'''
ENGINE_DIR = SOURCE_DIR / 'engine'
'''engine directory, for all engine code.'''
SHADER_DIR = ENGINE_DIR / 'shaders'
'''shader directory, for all .glsl code.'''
COMFYUI_DIR = SOURCE_DIR / 'comfyui'
'''comfyui directory, for all comfyui code.'''
UI_DIR = SOURCE_DIR / 'ui'
'''ui directory, for ui interface source code.'''

__all__ = ['PROJECT_DIR', 'SOURCE_DIR', 'ENGINE_DIR', 'SHADER_DIR', 'COMFYUI_DIR', 'UI_DIR',]

RESOURCES_DIR = PROJECT_DIR / 'resources'
'''resources directory, for obj, shader, texture, etc.'''
EXAMPLE_3D_MODEL_DIR = RESOURCES_DIR / 'example-3d-models'
'''3d model directory, contains all example 3d model files for quick test.'''
EXAMPLE_MAP_OUTPUT_DIR = RESOURCES_DIR / 'example-map-outputs'
'''example map output directory, contains all example map output files for quick test.'''

__all__.extend(['RESOURCES_DIR', 'EXAMPLE_3D_MODEL_DIR', 'EXAMPLE_MAP_OUTPUT_DIR',])


OUTPUT_DIR = PROJECT_DIR / 'output'
'''output directory, for runtime map, etc.'''
CACHE_DIR = OUTPUT_DIR / '.cache'
'''cache directory, for caching corr map'''
MAP_OUTPUT_DIR = OUTPUT_DIR / 'runtime_map'
'''runtime map output directory, for saving normal map, pos map, id map, etc., during runtime.'''
GIF_OUTPUT_DIR = OUTPUT_DIR / 'gif'
'''gif output directory, for saving gif result by StableDiffusion'''
TEMP_OUTPUT_DIR = OUTPUT_DIR / 'tmp'
'''temp output directory. For any usage.'''


__all__.extend(['OUTPUT_DIR', 'CACHE_DIR', 'MAP_OUTPUT_DIR', 'GIF_OUTPUT_DIR', 'TEMP_OUTPUT_DIR',])


def get_new_map_output_dir(create_if_not_exists:bool=True):
    '''Return a dir under MAP_OUTPUT_DIR with current time as its name. Will create one with a unique index.'''
    count = 0
    cur_time = datetime.datetime.now().strftime('%Y-%m-%d') + f'_{count}'
    while os.path.exists(os.path.join(MAP_OUTPUT_DIR, cur_time)):
        count += 1
        cur_time = datetime.datetime.now().strftime('%Y-%m-%d') + f'_{count}'
    cur_map_output_dir = os.path.join(MAP_OUTPUT_DIR, cur_time)
    if create_if_not_exists:
        os.makedirs(cur_map_output_dir, exist_ok=True)
    return cur_map_output_dir

def get_map_output_dir(day:int, index:int, month:Optional[int]=None, year:Optional[int]=None):
    '''
    Return a subdir under MAP_OUTPUT_DIR with your specified time.
    If month or year is not specified, use current month or year.
    If no such subdir, raise FileNotFoundError.
    '''
    if month is None:
        month = datetime.datetime.now().strftime('%m')  # type: ignore
    if year is None:
        year = datetime.datetime.now().strftime('%Y')   # type: ignore
    cur_subdir = os.path.join(MAP_OUTPUT_DIR, f'{year}-{month}-{day}_{index}')
    if not os.path.exists(cur_subdir):
        raise FileNotFoundError(f'No such subdir: {cur_subdir}')
    return cur_subdir


__all__.extend(['get_new_map_output_dir', 'get_map_output_dir',])