from PIL import Image
from modules.utils import make_gif, list_frames
from modules import config


img_dir = config.test_dir / 'boat' / 'outputs'
img_paths = list_frames(img_dir)
images = [Image.open(img_path).convert('RGB') for img_path in img_paths]

make_gif(images, save_path=config.test_dir / 'boat' / 'outputs' / 'GIF.gif')
