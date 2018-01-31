from prepare.scale_pyramid import scale_image_generator
from prepare.patch import patches
from prepare.prepare import img_density_generator, scale_density_map_generator, img_mask_generator
from utils.explorer import create_dir
import numpy as np
from os.path import join
import argparse
from skimage.io import imread, imsave
from os import listdir
from os.path import isfile, splitext, basename


# for python 2 & 3 compatibility
try:
    from itertools import izip
except ImportError:
    izip = zip


def make_patches(image_files, patch_size, overlap, scales, output_path):
    for f in image_files:
        image = imread(f)
        for scale_index, scaled_image in enumerate(scale_image_generator(image, scales)):
            for patch_index, patch in enumerate(patches(scaled_image, patch_size, overlap, 0)):
                patch_name = '{}_patch_{}_{}.jpg'.format(splitext(basename(f))[0], scale_index, patch_index)
                imsave(join(output_path, patch_name), patch)


def make_patches_with_counts(images_path, markers_path, patch_size, patch_overlap, scales, output_path, save_patches_as_images):
    patches_list = []
    patch_counts = []
    for image_name, image_orig, density_map_orig in img_density_generator(images_path, markers_path):
        for scale_index, (image, density_map) in enumerate(izip(scale_image_generator(image_orig, scales),
                                                                scale_density_map_generator(density_map_orig, scales))):
            for patch_index, (img_patch, density_map_patch) in enumerate(izip(patches(image, patch_size, patch_overlap, 0),
                                                                              patches(density_map, patch_size, patch_overlap, 0))):
                num_people = density_map_patch.sum()
                patch_counts.append(num_people)
                patches_list.append(img_patch)

                if save_patches_as_images:
                    patch_name = '{}_patch_{}_{}.jpg'.format(image_name, scale_index, patch_index)
                    imsave(join(output_path, patch_name), img_patch)

    patches_list = np.stack(patches_list, axis=0)
    np.save(join(output_path, 'patches'), patches_list, allow_pickle=True)

    patch_counts = np.asarray(patch_counts)
    np.save(join(output_path, 'patch_counts'), patch_counts, allow_pickle=True)


def make_patches_with_masks(images_path, mask_path, patch_size, patch_overlap, scales, output_path, save_patches_as_images):
    patches_list = []
    masks_list = []
    for image_name, image_orig, mask_orig in img_mask_generator(images_path, mask_path):
        for scale_index, (image, mask) in enumerate(izip(scale_image_generator(image_orig, scales),
                                                         scale_image_generator(mask_orig, scales))):
            for patch_index, (img_patch, mask_patch) in enumerate(
                    izip(patches(image, patch_size, patch_overlap, 0),
                         patches(mask, patch_size, patch_overlap, 0))):
                patches_list.append(img_patch)
                masks_list.append(mask_patch)
                if save_patches_as_images:
                    patch_name = '{}_patch_{}_{}.jpg'.format(image_name, scale_index, patch_index)
                    imsave(join(output_path, patch_name), img_patch)
                    mask_name = '{}_mask_{}_{}.jpg'.format(image_name, scale_index, patch_index)
                    imsave(join(output_path, mask_name), mask_patch)

    patches_list = np.stack(patches_list, axis=0)
    np.save(join(output_path, 'patches'), patches_list, allow_pickle=True)

    masks_list = np.stack(masks_list, axis=0)
    np.save(join(output_path, 'masks'), masks_list, allow_pickle=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('mode', action='store', choices=['patches', 'counts', 'masks'], help='What to do: only patches, patches with counts or patches with masks')
    parser.add_argument('images_path', help='path to the folder with images to be processed')
    parser.add_argument('patch_size', nargs=2, type=int, help='size of the patch')
    parser.add_argument('patch_overlap', type=float, help='patch overlap in percentage between 0 and 0.9')
    parser.add_argument('output_path', help='output path')

    parser.add_argument('-smin', '--scale_min', type=float, default=1, help='start scale of the scale pyramid')
    parser.add_argument('-smax', '--scale_max', type=float, default=1, help='end of the scale pyramid range')
    parser.add_argument('-snum', '--scale_num', type=float, default=1, help='number of scales')
    parser.add_argument('-mr', '--markers_path', help='path to markers files')
    parser.add_argument('-ms', '--masks_path', help='path to mask files')
    parser.add_argument('-sp', '--save_patches_as_images', action='store_true', help='save patches a images')

    args = parser.parse_args()

    create_dir(args.output_path)

    if args.mode == 'patches':
        image_files = [f for f in listdir(args.images_path) if isfile(join(args.images_path, f)) and splitext(basename(f))[1] == '.jpg']
        image_files = [join(args.images_path, f) for f in image_files]
        make_patches(image_files,
                     args.patch_size,
                     args.patch_overlap,
                     np.linspace(args.scale_min, args.scale_max, num=args.scale_num),
                     args.output_path)
    elif args.mode == 'counts':
        make_patches_with_counts(args.images_path,
                                 args.markers_path,
                                 args.patch_size,
                                 args.patch_overlap,
                                 np.linspace(args.scale_min, args.scale_max, num=args.scale_num),
                                 args.output_path,
                                 args.save_patches_as_images)
    elif args.mode == 'masks':
        make_patches_with_masks(args.images_path,
                                args.masks_path,
                                args.patch_size,
                                args.patch_overlap,
                                np.linspace(args.scale_min, args.scale_max, num=args.scale_num),
                                args.output_path,
                                args.save_patches_as_images)

