from prepare.prepare import prepare_training_data
from utils.explorer import empty_dir, create_dir
import numpy as np
from os.path import join


img_dir = join('data', 'raw', 'images')
markers_dir = join('data', 'raw', 'markers')

patches_dir = join('data', 'train', 'img', 'img')
create_dir(patches_dir)
empty_dir(patches_dir)

density_maps_dir = join('data', 'train', 'target', 'target')
create_dir(density_maps_dir)
empty_dir(density_maps_dir)

prepare_training_data(img_path=img_dir,
                      markers_path=markers_dir,
                      patch_size=(256, 256),
                      patch_overlay=0.5,
                      scales=np.arange(0.5, 1.3, 0.1),
                      out_path=[patches_dir, density_maps_dir])