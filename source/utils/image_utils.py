import os
from datetime import datetime
from PIL.Image import Image
from typing import Sequence

from .debug_utils import DefaultLogger
from .path_utils import GIF_OUTPUT_DIR

def save_images_as_gif(images: Sequence[Image],
                       output_fname: str = 'output.gif', 
                       gif_output_dir: str = GIF_OUTPUT_DIR,
                       add_time_to_name: bool = True):
    '''
    Saves a list of images as a gif file
    
    Args:
        images: list of images(PIL.Image.Image)
        output_fname: name of the output gif file
        gif_output_dir: directory to save the gif file
        add_time_to_name: if True, adds the current time to the output_fname
    '''
    if not os.path.exists(gif_output_dir):
        os.makedirs(gif_output_dir)
    if add_time_to_name:
        path = os.path.join(gif_output_dir, datetime.now().strftime(f"%Y-%m-%d_%H-%M_{output_fname}"))
    else:
        path = os.path.join(gif_output_dir, output_fname)
    images[0].save(path, format="GIF", save_all=True, append_images=images[1:], loop=0)
    DefaultLogger.success(f'Saved image sequence at {path}')



