from __future__ import print_function
from model.crowdnet import CrowdNet
from keras.preprocessing.image import ImageDataGenerator
from utils.npy_iterator import NpyDirectoryIterator
from utils.explorer import create_dir, get_files_in_dir
import argparse
from os.path import join
from datetime import datetime
from keras.optimizers import SGD
import numpy as np
from skimage.io import imread
import os


# for python 2 & 3 compatibility
try:
    from itertools import izip
except ImportError:
    izip = zip


def read_images_from(dir, img_shape):
    """

    :param dir:
    :param img_shape:
    :return:
    """
    img_files = get_files_in_dir(dir, '.jpg')
    num_imgs = len(img_files)
    images = np.zeros((num_imgs,) + img_shape, dtype=float)
    as_grey = img_shape[2] == 1
    for i, f in enumerate(img_files):
        img = imread(os.path.join(dir, f), as_grey=as_grey)
        images[i] = img
    images = np.divide(images, 255.)
    return images


def read_npy_from_dir(dir, shape):
    """

    :param dir:
    :param shape:
    :return:
    """
    npy_files = get_files_in_dir(dir, '.npy')
    num_arrays = len(npy_files)
    arrays = np.zeros((num_arrays,) + shape, dtype=float)
    for i, f in enumerate(npy_files):
        arrays[i] = np.load(os.path.join(dir, f))
    return arrays


def train_on_generators(images_path, density_maps_path, input_shape, epochs, verbosity, batch_size, learning_rate, workers):
    """

    :param images_path: string
    :param density_maps_path: string
    :param input_shape: (int, int)
    :param epochs: int
    :param verbosity: int
    :param batch_size: int
    :param learning_rate: float
    :param workers: int
    :return:
    """
    if verbosity > 0:
        print('Creating model...')

    crowdnet = CrowdNet()
    model = crowdnet.model_for_training(input_shape)

    if verbosity > 0:
        print('Compiling model...')

    optimizer = SGD(lr=learning_rate, momentum=0.9, decay=0)

    model.compile(optimizer=optimizer,
                  loss='mean_squared_error')

    if verbosity > 0:
        print('Creating data generators...')

    image_datagen = ImageDataGenerator(rescale=1./255)
    density_map_datagen = ImageDataGenerator()

    seed = 1

    image_generator = image_datagen.flow_from_directory(
        images_path,
        target_size=input_shape[0:2],
        color_mode='rgb',
        class_mode=None,
        shuffle=True,
        seed=seed,
        batch_size=batch_size
    )

    density_map_generator = NpyDirectoryIterator(
        density_maps_path,
        density_map_datagen,
        target_size=input_shape[0:2],
        shuffle=True,
        seed=seed,
        batch_size=batch_size
    )

    train_generator = (([x, x], y) for x, y in izip(image_generator, density_map_generator))

    if verbosity > 0:
        print('starting training...')

    model.fit_generator(
        train_generator,
        steps_per_epoch=50,
        epochs=epochs,
        verbose=verbosity,
        workers=workers,
        pickle_safe=workers > 1
    )

    # save model after training
    create_dir('out')
    out_path = join('out', '{}_{}.h5'.format(crowdnet.name, datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
    model.save(out_path)
    if verbosity > 0:
        print('model has been saved to {}'.format(out_path))


def train_in_memory(images_path, density_maps_path, input_shape, epochs, verbosity, batch_size, learning_rate):
    """

    :param images_path:
    :param density_maps_path:
    :param input_shape:
    :param epochs:
    :param verbosity:
    :param batch_size:
    :param learning_rate: float
    :return:
    """
    if verbosity > 0:
        print('Reading images from {}'.format(images_path))
    x = read_images_from(images_path, input_shape)

    if verbosity > 0:
        print('Reading density maps from {}'.format(density_maps_path))
    y = read_npy_from_dir(density_maps_path, input_shape[0:2])
    y = np.expand_dims(y, axis=3)
    x = [x, x]

    if verbosity > 0:
        print('Creating model...')
    crowdnet = CrowdNet()
    model = crowdnet.model_for_training(input_shape)

    if verbosity > 0:
        print('Compiling model...')
    optimizer = SGD(lr=learning_rate, momentum=0.9, decay=0)
    model.compile(optimizer=optimizer,
                  loss='mean_squared_error')

    if verbosity > 0:
        print('Starting training...')
    model.fit(x, y,
              epochs=epochs,
              verbose=verbosity,
              batch_size=batch_size,
              shuffle=True)

    # save model after training
    create_dir('out')
    out_path = join('out', '{}_{}.h5'.format(crowdnet.name, datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
    model.save(out_path)
    if verbosity > 0:
        print('model has been saved to {}'.format(out_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('images_path', help='path to the folder with training images')
    parser.add_argument('density_maps_path', help='path to the folder with corresponding density maps')
    parser.add_argument('input_shape', nargs='+', type=int, help='size of input data, expected as WIDTH HEIGHT CHANNELS')

    parser.add_argument('-e', '--epochs', type=int, default=50, help='number of training epochs, 50 by default')
    parser.add_argument('-v', '--verbosity', action='count', default=0, help='verbosity level, expected to be between 0 and 2')
    parser.add_argument('-b', '--batch_size', default=32, type=int, help='number of images in a training batch')
    parser.add_argument('-w', '--workers', default=4, type=int, help="maximum number of processes to spin up, default is 4")
    parser.add_argument('-m', '--in_memory', action='store_true', help="load training data into memory")
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.00001, help='learning rate')

    args = parser.parse_args()

    if args.in_memory:
        if args.verbosity > 0:
            print('Training model in memory')
        train_in_memory(args.images_path,
                        args.density_maps_path,
                        tuple(args.input_shape),
                        args.epochs,
                        args.verbosity,
                        args.batch_size,
                        args.learning_rate)
    else:
        if args.verbosity > 0:
            print('Training model using generators')
        train_on_generators(args.images_path,
                            args.density_maps_path,
                            tuple(args.input_shape),
                            args.epochs,
                            args.verbosity,
                            args.batch_size,
                            args.learning_rate,
                            args.workers)