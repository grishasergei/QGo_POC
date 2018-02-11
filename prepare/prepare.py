from os import listdir
from os.path import isfile, join, splitext, basename
from skimage.io import imread, imsave
from .density_map import get_density_map_from_markers
from .scale_pyramid import scale_image_generator, scale_density_map_generator
from .patch import patches_generator
import numpy as np
from prepare.labelme_mask import get_mask_from_labelme_xml

# for python 2 & 3 compatibility
try:
    from itertools import izip
except ImportError:
    izip = zip


def prepare_training_data(img_path, markers_path, patch_size, patch_overlay, scales, out_path):
    """
    Prepares data for training
    :param img_path: string, path to the folder with images
    :param markers_path: string, path to the folder wih corresponding marker files
    :param patch_size: tuple (int, int), size of patches
    :param patch_overlay: float between 0.0 and 0.95, percentage of overlay between patches
    :param scales: array of floats, scale images before extracting patches
    :param out_path: (string, string), folders where patches (0) and corresponding density maps (1) are saved
    :return: nothing, extracted patches and corresponding density maps are saved in out_path
    """
    for image_name, image_orig, density_map_orig in img_density_generator(img_path, markers_path):
        for scale_index, (image, density_map) in enumerate(izip(scale_image_generator(image_orig, scales),
                                                                scale_density_map_generator(density_map_orig, scales))):
            for patch_index, (img_patch, density_map_patch) in enumerate(izip(patches_generator(image, patch_size, patch_overlay, 1),
                                                                              patches_generator(density_map, patch_size, patch_overlay, 1e-7))):
                n = 1
                num_people = density_map_patch.sum()
                if num_people > 20:
                    n += 1
                if num_people > 30:
                    n += 1
                if num_people > 40:
                    n += 2
                for i in range(n):
                    img_patch_name = '{}_patch_{}_{}_{}.jpg'.format(image_name, scale_index, patch_index, i)
                    density_map_patch_name = '{}_density_map_patch_{}_{}_{}'.format(image_name, scale_index, patch_index, i)
                    imsave(join(out_path[0], img_patch_name), img_patch)
                    np.save(join(out_path[1], density_map_patch_name), density_map_patch)


def img_density_generator(img_path, markers_path):
    """

    :param img_path: string, path to the folder with images
    :param markers_path: string, path to the folder wih corresponding marker files
    :return: image and mask pair
    """
    img_files = [f for f in listdir(img_path) if is_file(join(img_path, f), '.jpg')]
    marker_files = [f for f in listdir(markers_path) if is_file(join(markers_path, f), '.txt')]

    if len(img_files) != len(marker_files):
        raise Exception('Number of images is not equal to the number of marker files')

    img_files.sort()
    marker_files.sort()

    for img_file, marker_file in zip(img_files, marker_files):
        if splitext(img_file)[0] != (splitext(marker_file)[0]).split('_')[0]:
            raise Exception('Image {} and Markers {} do not match'.format(img_file, marker_file))

    for img_file, marker_file in zip(img_files, marker_files):
        image = read_image(join(img_path, img_file))
        density_map = get_density_map_from_markers(join(markers_path, marker_file),
                                                   image.shape[0:-1])
        yield splitext(img_file)[0], image, density_map


def img_mask_generator(img_path, mask_path):
    img_files = [f for f in listdir(img_path) if is_file(join(img_path, f), '.jpg')]
    mask_files = [f for f in listdir(mask_path) if is_file(join(mask_path, f), '.xml')]

    if len(img_files) != len(mask_files):
        raise Exception('Number of images is not equal to the number of mask files')

    img_files.sort()
    mask_files.sort()

    for img_file, marker_file in zip(img_files, mask_files):
        if splitext(img_file)[0] != (splitext(marker_file)[0]).split('_')[0]:
            raise Exception('Image {} and Mask {} do not match'.format(img_file, marker_file))

    for img_file, mask_file in zip(img_files, mask_files):
        image = read_image(join(img_path, img_file))
        mask = get_mask_from_labelme_xml(join(mask_path, mask_file))
        yield splitext(img_file)[0], image, mask


def read_image(img_file):
    """
    Reads image from file
    :param img_file: string, path to the image file
    :return: image as a numpy array
    """
    return imread(img_file)


def is_file(path, extension):
    """
    Returns true if path is a file with specified extension, false otherwise
    :param path: string, path to the file
    :param extension: string, desired extension
    :return: boolean
    """
    if not isfile(path):
        return False

    return splitext(basename(path))[1] == extension
