import cv2
import os
from common_utils.debug_utils import DefaultLogger

if __name__ == '__main__':
    video_path = 'test/video/rick_roll.mp4'
    save_dir = 'test/frames'

    os.makedirs(save_dir, exist_ok=True)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # 读取视频的前30帧（如果fps为30，即第一秒的每一帧）
    count = 0
    for t in range(10):
        for i in range(fps):
            ret, frame = cap.read()
            if not ret:
                DefaultLogger.warn("Can't receive frame (stream end?). Exiting ...")
                break

            # 构造图像的路径，并保存图像
            frame_path = os.path.join(save_dir, f'frame_{count}.png')
            count += 1
            cv2.imwrite(frame_path, frame)

    # 释放视频捕获对象
    cap.release()
