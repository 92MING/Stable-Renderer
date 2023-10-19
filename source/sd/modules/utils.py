import re
import cv2
import os
import numpy
import torch
import torchvision.transforms as transforms
from typing import List
from PIL import Image
from pathlib import Path
from . import config
from . import log_utils as logu


def list_frames(frame_dir):
    pattern = re.compile(r".*_(\d+).png")
    try:
        frame_list = sorted(listfiles(frame_dir, exts='.png', return_type=Path, return_path=True), key=lambda filename: int(pattern.match(str(filename)).group(1)))
    except AttributeError as e:
        raise AttributeError(f"Frame filename format is not correct. Please make sure all filenames follow format `*_xxx.png`, where `xxx` is an integer.") from e
    return frame_list


def listfiles(directory, exts=None, return_type: type = None, return_path: bool = False, return_dir: bool = False, recur: bool = False):
    if exts and return_dir:
        raise ValueError("Cannot return both files and directories")
    if return_type != str and not return_path:
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


def make_canny_images(images: List[Image.Image], threshold1=100, threshold2=200) -> List[Image.Image]:
    """
    Make canny images from a list of PIL images.
    :param images: list of PIL images
    :param threshold1: first threshold for the hysteresis procedure
    :param threshold2: second threshold for the hysteresis procedure
    :return: list of PIL canny images
    """
    canny_images = []
    for image in images:
        image = numpy.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image = cv2.Canny(image, threshold1, threshold2)
        image = Image.fromarray(image)
        canny_images.append(image)
    return canny_images


def make_depth_images(images: List[Image.Image]):
    """
    Make depth images from a list of PIL images.
    :param images: list of PIL images
    :return: list of depth images
    """
    model_type = "DPT_Large"
    model = torch.hub.load("intel-isl/MiDaS", model_type)
    model.to("cuda")
    model.eval()
    depth_images = []
    for img in images:

        transform = transforms.Compose(
            [
                transforms.Resize(384),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        img_tensor = transform(img).unsqueeze(0).to("cuda")  # add a batch dimension

        with torch.no_grad():
            depth = model(img_tensor)

        depth_img = depth[0].squeeze().cpu().numpy()
        depth_images.append(depth_img)
    return depth_images


def make_correspondence_map(from_dir, save_to_path):
    """
    Make correspondence map from a directory of images.
    If the correspondence map already exists, then load it.
    Otherwise, create and save it.
    :param from_dir: The directory of images.
    :param save_to_path: The path to save the correspondence map.
    :return: The correspondence map.
    """
    import pickle
    from .data_classes.correspondenceMap import CorrespondenceMap
    if save_to_path.is_file():
        try:
            with open(save_to_path, 'rb') as f:
                corr_map = pickle.load(f)
            return corr_map
        except ModuleNotFoundError as e:
            os.remove(save_to_path)
            logu.warn(f"[WARNING] Correspondence map {save_to_path} is corrupted. It will be re-created.")

    logu.info(f"[INFO] Creating correspondence map from {from_dir}")
    corr_map = CorrespondenceMap.from_existing_directory_img(
        from_dir,
        enable_strict_checking=False,
        num_frames=10
    )
    with open(save_to_path, 'wb') as f:
        pickle.dump(corr_map, f)
    logu.success(f"[SUCCESS] Correspondence map created and saved to {save_to_path}")
    return corr_map


def save_latents(i, t, latents):
    """
    Callback function to save vae-approx-decoded latents during inference.
    :param i: The current inference step.
    :param t: The current time step.
    :param latents: The current latents. If is list type, then process the first element.
    """
    from .vae_approx import latents_to_single_pil_approx
    logu.info(f"[INFO] Saving latents at inference step {i} and time step {t}")
    save_dir = config.test_dir / 'latents'
    save_dir.mkdir(parents=True, exist_ok=True)
    if isinstance(latents, list):
        image = [latents_to_single_pil_approx(lat) for lat in latents]
        [image.save(save_dir / f'latents_{i:02d}_{j:02d}.png') for j, image in enumerate(image)]
    else:
        image = latents_to_single_pil_approx(latents)
        image.save(save_dir / f'latents_{i:02d}.png')
