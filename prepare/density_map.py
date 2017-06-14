import numpy as np
import json
from scipy.ndimage.filters import gaussian_filter


def get_density_map_from_markers(markers_file, img_size):
    """
    Makes a crowd density map from markers
    :param markers_file: string, path to the markers file
    :param img_size: (int, int), size of the image
    :return: ndarray
    """
    density_map = np.zeros(img_size, dtype=float)

    with open(markers_file) as f:
        markers_data = json.load(f)

    for marker in markers_data['annotations']:
        if marker['type'] != 'dot_marker':
            continue
        density_map[marker['y'], marker['x']] = 1

    density_map = gaussian_filter(density_map, sigma=3)

    return density_map
