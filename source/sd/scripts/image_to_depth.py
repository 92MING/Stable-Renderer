import sys, os
sys.path.append(os.getcwd())

import torch
import torchvision.transforms as transforms
import cv2
import tqdm
from PIL import Image
from pathlib import Path
import modules.utils as utils

if __name__ == '__main__':
    # 1. 加载预训练模型
    model_type = "DPT_Large"   # 这只是一个选择，也可以选择其他的模型类型
    model = torch.hub.load("intel-isl/MiDaS", model_type)
    device = ("cuda" if torch.cuda.is_available() else 
             "mps" if torch.backends.mps.is_available() else
             "cpu")
    model.to(device)
    model.eval()

    img_dir = Path('test/groups/group_7')
    out_dir = Path('test/depths/group_7')
    out_dir.mkdir(parents=True, exist_ok=True)

    for img_path in tqdm.tqdm(utils.list_frames(img_dir)):
        img = Image.open(img_path).convert("RGB")

        transform = transforms.Compose(
            [
                transforms.Resize(384),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        img_tensor = transform(img).unsqueeze(0).to(device)  # add a batch dimension

        with torch.no_grad():
            depth = model(img_tensor)

        depth_img = depth[0].squeeze().cpu().numpy()
        cv2.imwrite(str(out_dir / Path(img_path).name), depth_img)

        # plt.imshow(depth_img, cmap='inferno')
        # plt.colorbar()
        # plt.show()
