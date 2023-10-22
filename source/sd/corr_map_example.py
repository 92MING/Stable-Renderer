from modules.utils import make_correspondence_map, save_corr_map_visualization
from modules import config

test_dir = config.test_dir / 'boat'
corr_map = make_correspondence_map(
    test_dir / "id",
    test_dir / "corr_map.pkl",
    force_recreate=False,
)
save_corr_map_visualization(corr_map, save_dir=test_dir / "corr_map_vis", n=8, division=4)
