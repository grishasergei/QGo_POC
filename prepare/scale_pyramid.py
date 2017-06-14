from scipy.misc import imresize


def scale_pyramid_generator(image, scales):
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
