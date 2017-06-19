from scipy.misc import imresize


def scale_image_generator(image, scales):
    """
    Produces a generator of the same image at different scales
    :param image: ndarray
    :param scales: list of floats
    :return: generator
    """
    for scale in scales:
        if scale == 1.0:
            yield image
        else:
            yield imresize(image, scale, interp='bilinear')


def scale_density_map_generator(density_map, scales):
    """
    Produces a generator of the density map at different scales
    :param density_map: DensityMap
    :param scales: list of floats
    :return: generator
    """
    for scale in scales:
        yield density_map.as_array(scale)