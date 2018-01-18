from prepare.scale_pyramid import scale_image_generator
from prepare.patch import patches
from utils.explorer import create_dir
import numpy as np
from os.path import join
import argparse
from skimage.io import imread, imsave
from os import listdir
from os.path import isfile, splitext, basename


def make_patches(images_path, patch_size, overlap, scales, output_path):
    files = [f for f in listdir(images_path) if isfile(join(images_path, f)) and splitext(basename(f))[1] == '.jpg']

    for f in files:
        image = imread(join(images_path, f))
        for scale_index, scaled_image in enumerate(scale_image_generator(image, scales)):
            for patch_index, patch in enumerate(patches(scaled_image, patch_size, overlap, 0)):
                patch_name = '{}_patch_{}_{}.jpg'.format(f, scale_index, patch_index)
                imsave(join(output_path, patch_name), patch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('images_path', help='path to the folder with images to be processed')
    parser.add_argument('patch_size', nargs=2, type=int, help='size of the patch')
    parser.add_argument('patch_overlap', type=float, help='patch overlap in percentage between 0 and 0.9')
    parser.add_argument('output_path', help='output path')

    parser.add_argument('-smin', '--scale_min', type=float, default=1, help='start scale of the scale pyramid')
    parser.add_argument('-smax', '--scale_max', type=float, default=1, help='end of the scale pyramid range')
    parser.add_argument('-snum', '--scale_num', type=float, default=1, help='number of scales')

    args = parser.parse_args()

    create_dir(args.output_path)
    make_patches(args.images_path,
                 args.patch_size,
                 args.patch_overlap,
                 np.linspace(args.scale_min, args.scale_max, num=args.scale_num),
                 args.output_path)