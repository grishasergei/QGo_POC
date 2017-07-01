import numpy as np
import json
from scipy.ndimage.filters import gaussian_filter
import keras.backend as K


class DensityMap(object):

    def __init__(self, markers, shape):
        """

        :param markers: list
            [
                {
                    "y": int,
                    "x": int
                }, ...
            ]
        :param shape: (int, int)
        """
        self._markers = markers
        self._shape = shape

    def as_array(self, scale=1):
        """
        Returns density map as a numpy array
        :param scale: float
        :return: ndarray
        """
        shape = (int(self._shape[0] * scale),
                 int(self._shape[1] * scale))

        arr = np.full(shape, 1e-7, dtype=K.floatx())

        for marker in self._markers:
            y = int(marker['y'] * scale)
            x = int(marker['x'] * scale)
            arr[y, x] = 1

        return gaussian_filter(arr, sigma=10)


def get_density_map_from_markers(markers_file, shape):
    """
    Makes a crowd density map from markers
    :param markers_file: string, path to the markers file
    :param shape: (int, int), size of the image
    :return: DensityMap
    """
    with open(markers_file) as f:
        markers_data = json.load(f)

    dot_markers = [m for m in markers_data['annotations'] if m['type'] == 'dot_marker']

    density_map = DensityMap(dot_markers, shape)

    return density_map
