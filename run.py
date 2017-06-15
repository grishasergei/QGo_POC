from prepare.prepare import prepare_training_data
from utils.explorer import empty_dir, create_dir


img_dir = './data/raw/images_test/'
markers_dir = './data/raw/markers_test/'

patches_dir = './data/train/img/'
create_dir(patches_dir)
empty_dir(patches_dir)

density_maps_dir = './data/train/target/'
create_dir(density_maps_dir)
empty_dir(density_maps_dir)

prepare_training_data(img_path=img_dir,
                      markers_path=markers_dir,
                      patch_size=(128, 128),
                      patch_overlay=0,
                      #[0.5, 0.75, 1, 1.5],
                      scales=[1],
                      out_path=[patches_dir, density_maps_dir])