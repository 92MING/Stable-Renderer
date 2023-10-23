from modules.utils import make_correspondence_map, save_corr_map_visualization, scale_corr_map
from modules import config

test_dir = config.test_dir / '2023-10-23_3'
division = 1
corr_map = make_correspondence_map(
    test_dir / "id",
    test_dir / "corr_map.pkl",
    num_frames=16,
    force_recreate=True,
)
corr_map = scale_corr_map(corr_map, scale_factor=1 / division)
save_corr_map_visualization(corr_map, save_dir=test_dir / "corr_map_vis", n=32, division=division)
