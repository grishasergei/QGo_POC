import numpy as np


def patches_generator(image, patch_size, overlap, cval):
    """
    Patches generator
    :param image: ndarray
    :param patch_size: (int, int)
    :param overlap: float between 0.0 and 0.95
    :param cval: int, value for padding
    :return: ndarray generator
    """
    if (overlap < 0) or (overlap > 0.95):
        raise Exception('Patch overlap must be between 0 and 0.95. Actual is {}'.format(overlap))

    w_step = int(float(patch_size[0]) * (1.0 - overlap))
    h_step = int(float(patch_size[1]) * (1.0 - overlap))

    image = pad_for_patching(image, patch_size, cval)

    for x in range(0, image.shape[0] - int(patch_size[0] * overlap), w_step):
        for y in range(0, image.shape[1] - int(patch_size[1] * overlap), h_step):
            yield image[x:x+patch_size[0],
                        y:y+patch_size[1]]


def pad_for_patching(image, patch_size, cval):
    """
    Pad an image in order to produce patches of equal size
    :param image: ndarray
    :param patch_size: (int, int)
    :param cval: int
    :return: ndarray, padded image
    """
    h_pad = 0
    if image.shape[0] % patch_size[0] != 0:
        h_pad = patch_size[0] - image.shape[0] % patch_size[0]

    w_pad = 0
    if image.shape[1] % patch_size[1] != 0:
        w_pad = patch_size[1] - image.shape[1] % patch_size[1]

    if w_pad == 0 and h_pad == 0:
        return image

    w_pad_l = int(w_pad / 2)
    w_pad_r = int(w_pad - w_pad_l)

    h_pad_t = int(h_pad / 2)
    h_pad_b = int(h_pad - h_pad_t)

    pad_width = [(h_pad_t, h_pad_b), (w_pad_l, w_pad_r)]

    if len(image.shape) == 3:
        pad_width.append((0, 0))

    return np.pad(image, pad_width, mode='constant', constant_values=cval)