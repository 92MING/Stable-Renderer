from sd.modules.data_classes import CorrespondenceMap

if __name__ == '__main__':
    corr_map = CorrespondenceMap.from_existing_directory_img(
        'rendered_frames/map_30_512x512/id',
        enable_strict_checking=False,
        num_frames=10)

    print(corr_map.Map)