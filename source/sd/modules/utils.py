import re
import cv2
import numpy
import torch
import torchvision.transforms as transforms
from typing import List
from PIL import Image
from pathlib import Path


def list_frames(frame_dir):
    pattern = r"frame_(\d+).png"
    try:
        frame_list = sorted(frame_dir.iterdir(), key=lambda x: int(re.match(pattern, x.name).group(1)))
    except AttributeError as e:
        raise AttributeError(f"Frame filename format is not correct. Please make sure all filenames follow format `frame_xxx.png`, where `xxx` is an integer.") from e
    return frame_list


def listfiles(directory, exts, return_type: type = None, return_path: bool = False, return_dir: bool = False, recur: bool = False):
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


def split_by_index(frames, split_index: list):
    """
    Split frames into groups with similar perceptual hashes.
    """
    groups = []
    split_index.insert(0, 0)
    for i in range(1, len(split_index)):
        groups.append(frames[split_index[i-1]: split_index[i]])
        # print(f"Group {i} = [{split_index[i-1]}, {split_index[i]})")

    return groups


def view_latents(i, t, latents):
    """
    Callback function to view latents during inference.
    :param i: The current inference step.
    :param t: The current time step.
    :param latents: The current latents. If is list type, then process the first element.
    """
    from modules.vae_approx import latents_to_single_pil_approx
    image = latents_to_single_pil_approx(latents[0] if isinstance(latents, list) else latents)
    image.show()
