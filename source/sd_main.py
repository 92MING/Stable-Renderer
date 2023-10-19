from sd.modules.data_classes import CorrespondenceMap

if __name__ == '__main__':
    corr_map = CorrespondenceMap.from_existing_directory_img(
        '/Users/cyruss081115/Documents/Study/Uni/research/stable_render/Stable-Renderer/rendered_frames/maps_per_30_frames/id',
        enable_strict_checking=False,
        num_frames=10)

    print(corr_map.Map)