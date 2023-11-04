import re
import cv2
import os
import numpy
import torch
import tqdm
import json
import pickle
import shutil
import torchvision.transforms as transforms
from typing import List, Union
from PIL import Image
from pathlib import Path
from . import config
from . import log_utils as logu
from .data_classes.correspondenceMap import CorrespondenceMap


def list_frames(frame_dir, return_type: type = Path):
    r"""
    List and sort files with format `*_xxx.png` in a directory.
    :param frame_dir: Directory to list files from.
    :param return_type: Type to return. If None, returns the type of the `frame_dir`.
    :return: List of files in the directory.
    """
    pattern = re.compile(r".*_(\d+).png")
    try:
        frame_list = sorted(listfiles(frame_dir, exts='.png', return_type=Path, return_path=True), key=lambda filename: int(pattern.match(str(filename)).group(1)))
    except AttributeError as e:
        raise AttributeError(f"Frame filename format is not correct. Please make sure all filenames follow format `*_xxx.png`, where `xxx` is an integer.") from e
    if return_type == Path:
        pass
    elif return_type == str:
        frame_list = [str(frame) for frame in frame_list]
    elif return_type == Image.Image:
        frame_list = [Image.open(frame).convert('RGB') for frame in frame_list]
    return frame_list


def open_images(img_path_lst):
    return [Image.open(img_path).convert('RGB') for img_path in img_path_lst]


def listfiles(directory, exts=None, return_type: type = None, return_path: bool = False, return_dir: bool = False, recur: bool = False):
    r"""
    List files in a directory.
    :param directory: Directory to list files from.
    :param exts: List of extensions to filter files by. If None, all files are returned.
    :param return_type: Type to return. If None, returns the type of the `directory`. If `return_path` is True, this must be str.
    :param return_path: Whether to return the path of the file or just the name.
    :param return_dir: Whether to return directories instead of files.
    :param recur: Whether to recursively list files in subdirectories.
    :return: List of files in the directory.
    """
    if exts and return_dir:
        raise ValueError("Cannot return both files and directories")
    if not return_path and return_type != str:
        raise ValueError("Cannot return non-str type when returning name")
    return_type = return_type or type(directory) if return_path else str
    directory = Path(directory)
    files = [
        return_type(filepath) if return_path else filepath.name
        for filepath in directory.iterdir()
        if (not return_dir and filepath.is_file() and (exts is None or filepath.suffix in exts))
        or (return_dir and filepath.is_dir())
    ]

    if recur:
        for subdir in listfiles(directory, return_path=True, return_dir=True):
            files.extend(listfiles(subdir, exts=exts, return_type=return_type, return_path=return_path))

    return files


def make_correspondence_map(from_dir, save_to_path, num_frames=None, force_recreate=False):
    """
    Make correspondence map from a directory of images.
    If the correspondence map already exists, then load it.
    Otherwise, create and save it.
    :param from_dir: The directory of images.
    :param save_to_path: The path to save the correspondence map.
    :return: The correspondence map.
    """
    import pickle
    if save_to_path.is_file() and not force_recreate:
        try:
            with open(save_to_path, 'rb') as f:
                corr_map = pickle.load(f)
        except ModuleNotFoundError as e:
            os.remove(save_to_path)
            logu.warn(f"Correspondence map {save_to_path} is corrupted. It will be re-created.")
    else:
        logu.info(f"Creating correspondence map from {from_dir}")
        corr_map = CorrespondenceMap.from_existing_directory_numpy(
            from_dir,
            enable_strict_checking=False,
            # pixel_position_callback=lambda x, y: (x//8, y//8),
            num_frames=num_frames
        )
        with open(save_to_path, 'wb') as f:
            pickle.dump(corr_map, f)
        logu.success(f"Correspondence map created and saved to {save_to_path}")

    # Optionally save as json file
    json_path = save_to_path.with_suffix('.json')
    if not json_path.is_file():
        logu.info(f"Jsonifying correspondence map...")
        json_map = {}
        for key, value in corr_map.Map.items():
            json_map[str(key)] = str(value)

        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(json_map, f, indent=4)

    return corr_map


def save_latents(i, t, latents, save_dir, prefix='latents', postfix: str = '', decoder='vae-approx', vae=None, save_timestep=False, **kwargs):
    r"""
    Callback function to save vae-approx-decoded latents during inference.
    :param i: The current inference step.
    :param t: The current time step.
    :param latents: The current latents. If is list type, then process the first element.
    """
    if decoder.lower() == 'vae':
        if vae is None:
            raise ValueError("VAE must be provided when decoder is 'vae'")

        from .vae import decode
        decode_func = decode
        decode_kwargs = dict(vae=vae, return_pil=True)
    elif decoder.lower() == 'vae-approx':
        from .vae_approx import latents_to_single_pil_approx
        decode_func = latents_to_single_pil_approx
        decode_kwargs = dict()
    else:
        raise ValueError(f"Unknown decoder type: {decoder}. Must be 'vae' or 'vae-approx'")

    # logu.info(f"Saving latents: step {i} | timestep {t:0f} | decoder {decoder}")

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    prefix = f"{prefix}_" if prefix != '' else ''
    postfix = f"_{postfix}" if postfix != '' else ''

    if isinstance(latents, list):
        image = [decode_func(lat, **decode_kwargs) for lat in latents]
        if save_timestep:
            [image.save(save_dir / f'{prefix}s{i:02d}_t{int(t):03d}_f{j:02d}{postfix}.png') for j, image in enumerate(image)]
        else:
            [image.save(save_dir / f'{prefix}s{i:02d}_f{j:02d}{postfix}.png') for j, image in enumerate(image)]
    else:
        image = decode_func(latents, **decode_kwargs)
        image.save(save_dir / f'{prefix}_s{i:02d}.png')


def save_corr_map_visualization(corr_map: CorrespondenceMap, save_dir: Path, n: int = 8, division: int = 1, stem: str = 'corr_map'):
    r"""
    Visualize the correspondence map and save it to the given directory.
    :param corr_map: The correspondence map.
    :param save_dir: The directory to save the visualization.
    :param n: The number of frames to visualize.
    :param stem: The stem of the saved file.
    """
    logu.info(f"Saving correspondence map visualization...")
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    clear_dir(save_dir)
    n = min(n, corr_map.num_frames)
    frame_w = corr_map.width // division
    frame_h = corr_map.height // division
    image_seq = [numpy.zeros((frame_h, frame_w, 3), dtype=numpy.uint8) for _ in range(n)]
    ovlp_count = [numpy.zeros((frame_h, frame_w), dtype=numpy.uint8) for _ in range(n)]

    color_red = numpy.array([255, 0, 0], dtype=numpy.uint8)
    color_green = numpy.array([0, 255, 0], dtype=numpy.uint8)
    color_blue = numpy.array([0, 0, 255], dtype=numpy.uint8)
    color_white = numpy.array([255, 255, 255], dtype=numpy.uint8)
    color_cyan = numpy.array([0, 255, 255], dtype=numpy.uint8)

    trace_id = list(corr_map.Map.keys())[0]
    for v_id, v_info in corr_map.Map.items():
        info_len = len([t for t_pix_pos, t in v_info if t < n])
        for i in range(info_len):
            t_pix_pos, t = v_info[i]
            h, w = t_pix_pos
            if t < n and h >= 0 and h < frame_h and w >= 0 and w < frame_w:
                ovlp_count[t][h, w] += 1

    # Trace vertex
    for i in range(n):
        for h in range(frame_h):
            for w in range(frame_w):
                if ovlp_count[i][h, w] > 0:
                    ovlp_ratio = ovlp_count[i][h, w] / n
                    image_seq[i][h, w, :] = color_red * ovlp_ratio + color_white * (1 - ovlp_ratio)

    for t_pix_pos, t in corr_map.Map[trace_id]:
        h, w = t_pix_pos
        if t < n and h >= 0 and h < frame_h and w >= 0 and w < frame_w:
            image_seq[t][h-3:h+3, w-3:w+3, :] = color_green

    images = [Image.fromarray(image) for image in image_seq]
    [image.save(save_dir / f"{stem}_{i:02d}.png") for i, image in enumerate(images)]
    make_gif(images, save_dir / f'{stem}.gif', fps=10)

    logu.success(f"CM visualization saved to {save_dir}")


def make_gif(images, save_path: Path, fps: int = 10):
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(save_path, save_all=True, append_images=images[1:], duration=1000//fps, loop=0)
    logu.success(f"Gif saved to {save_path}")


def clear_dir(dirpath):
    r"""
    Clear and make the given directory.
    :param dirpath: The directory path.
    """
    dirpath = Path(dirpath)
    if dirpath.is_dir():
        shutil.rmtree(dirpath)
    dirpath.mkdir(parents=True, exist_ok=True)


def scale_corr_map(corr_map: CorrespondenceMap, scale_factor: float = 0.125):
    r"""
    Scale the correspondence map.
    :param corr_map: The correspondence map.
    :param scale_factor: The scale factor.
    :return: The scaled correspondence map.
    """
    scaled_map = dict()
    for v_id, v_info in corr_map.Map.items():
        v_info = [((int(t_pix_pos[0] * scale_factor), int(t_pix_pos[1] * scale_factor)), t) for t_pix_pos, t in v_info]
        scaled_map[v_id] = v_info
    scaled_w = int(corr_map.width * scale_factor)
    scaled_h = int(corr_map.height * scale_factor)
    return CorrespondenceMap(scaled_map, scaled_w, scaled_h, corr_map.num_frames)


def truncate_corr_map(correspondence_map: CorrespondenceMap, start: int = None, end: int = None):
    r"""
    Truncate the correspondence map of frame index between [start, end).
    :param correspondence_map: The correspondence map.
    :param start: The start frame.
    :param end: The end frame.
    :return: The truncated correspondence map.
    """
    start = max(start, 0) if start is not None else 0
    end = min(end, correspondence_map.num_frames) if end is not None else correspondence_map.num_frames

    truncated_map = dict()
    for v_id, v_info in correspondence_map.Map.items():
        v_info = [(t_pix_pos, t-start) for t_pix_pos, t in v_info if t >= start and t < end]
        if len(v_info) > 0:
            truncated_map[v_id] = v_info
    return CorrespondenceMap(truncated_map, correspondence_map.width, correspondence_map.height, end-start)


def make_rev_corr_map(corr_map: CorrespondenceMap, save_to_path: Path, force_recreate: bool = False):
    save_to_path = Path(save_to_path)
    if save_to_path.is_file() and not force_recreate:
        # Load from file
        if save_to_path.suffix == '.json':
            with open(save_to_path, 'r') as f:
                rev_map = json.load(f)
        elif save_to_path.suffix == '.pkl':
            with open(save_to_path, 'rb') as f:
                rev_map = pickle.load(f)
    else:
        save_to_path.parent.mkdir(parents=True, exist_ok=True)
        rev_map = [dict() for _ in range(corr_map.num_frames)]
        for v_id, v_info in tqdm.tqdm(corr_map.Map.items(), desc='Making reverse correspondence map'):
            for t_pix_pos, t in v_info:
                h, w = t_pix_pos
                pos_str = f"{h},{w}"
                rev_map[t][pos_str] = ','.join([str(int(v)) for v in list(v_id)])
        # Save to file
        if save_to_path.suffix == '.json':
            with open(save_to_path, 'w') as f:
                json.dump(rev_map, f, indent=4)
        elif save_to_path.suffix == '.pkl':
            with open(save_to_path, 'wb') as f:
                pickle.dump(rev_map, f)
    return rev_map


def make_base_map(base_image: Image.Image, corr_map: CorrespondenceMap, num_frames=None, save_to_dir=None, return_pil: bool = True):
    width, height = corr_map.size
    num_frames = num_frames or corr_map.num_frames
    if isinstance(base_image, Image.Image):
        base_numpy = numpy.array(base_image)
    base_map = [numpy.zeros_like(base_numpy) for _ in range(num_frames)]
    assert base_numpy.shape[:2] == (height, width), f"base_image shape {base_numpy.shape[2:]} does not match corr_map shape {(height, width)}"

    for v_id, v_info in tqdm.tqdm(corr_map.Map.items(), desc="Creating base map"):
        for t_pix_pos, t in v_info:
            h, w = t_pix_pos
            if t < num_frames and w >= 0 and w < width and h >= 0 and h < height:
                fh, fw = v_info[0][0]
                base_map[t][h, w, :] = base_numpy[fh, fw, :]
            else:
                break

    if save_to_dir:
        save_to_dir = Path(save_to_dir)
        save_to_dir.mkdir(parents=True, exist_ok=True)
        [cv2.imwrite(str(save_to_dir / f"base_map_{i:02d}.png"), cv2.cvtColor(base_map[i], cv2.COLOR_RGB2BGR)) for i in range(num_frames)]
        logu.success(f"Base map saved to {save_to_dir}")

    if return_pil:
        base_map = [Image.fromarray(base_map[i]) for i in range(num_frames)]

    return base_map
