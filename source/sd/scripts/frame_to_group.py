import sys, os
sys.path.append(os.getcwd())

import modules.utils as utils
import re
import os
from PIL import Image
from pathlib import Path

ROOT = Path(os.getcwd())

if __name__ == "__main__":
    image_dir = ROOT / Path("test/frames")
    pattern = r"frame_(\d+).png"
    img_paths = sorted(image_dir.iterdir(), key=lambda x: int(re.match(pattern, x.name).group(1)))
    images = [Image.open(impath) for impath in img_paths]
    output_dir = Path("test/groups")
    output_dir.mkdir(parents=True, exist_ok=True)
    # groups = utils.split_by_phash(images)
    split_index = [2, 17, 24, 50, 61, 103, 132, 186, 193, 219, 236, 250]
    groups = utils.split_by_index(images, split_index)
    count = 0
    for group_id, group in enumerate(groups):
        group_dir = output_dir.joinpath(f"group_{group_id}")
        group_dir.mkdir(parents=True, exist_ok=True)
        for i, image in enumerate(group):
            image.save(group_dir.joinpath(f"frame_{count}.png"))
            count += 1
