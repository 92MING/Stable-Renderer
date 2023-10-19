import re
from pathlib import Path
from .. import config
from ..utils import listfiles


def load_images(from_dir):
    from_dir = Path(from_dir)
    pattern = re.compile(r'\d+')
    res = []
    for impath in listfiles(from_dir, exts='.png', return_path=True):
        imname = impath.name
        match = re.findall(pattern, imname)
        res.append((str(impath.absolute()), [int(idx) for idx in match]))
    res = sorted(res, key=lambda x: (x[1][0], x[1][1]))
    res = [(e[0], f"step {e[1][0]} | frame {e[1][1]}") for e in res]
    return res
