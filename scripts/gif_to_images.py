from PIL import Image


def read_gif(file_path, output_dir):
    images = []
    with Image.open(file_path) as gif:
        try:
            while True:
                gif.seek(gif.tell() + 1)
                image = gif.copy()
                if image is None:
                    break
                images.append(image)
                image.save(f"{output_dir}/{len(images)}.png")
        except EOFError:
            pass
    return images

if __name__ == "__main__":
    path = "/mnt/disk2/Stable-Renderer-Previous/Stable-Renderer/output/comfyui/2024-05-21_06-52-38/animate_diff_00001_.gif"
    output_dir = "/mnt/disk2/Stable-Renderer-Previous/Stable-Renderer/output/comfyui/2024-05-21_06-52-38/"
    read_gif(path, output_dir)
