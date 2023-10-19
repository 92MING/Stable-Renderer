from sd.modules.data_classes import CorrespondenceMap, ImageFrames

if __name__ == '__main__':
    corr_map = CorrespondenceMap.from_existing_directory_img(
        'rendered_frames/map_30_512x512/id',
        enable_strict_checking=False,
        num_frames=10)

    print(corr_map.Map)
    images = ImageFrames.from_existing_directory("rendered_frames/maps_per_30_512x512/color", 10)
    print(len(images))