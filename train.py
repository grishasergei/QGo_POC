from __future__ import print_function
from model.crowdnet import CrowdNet
from keras.preprocessing.image import ImageDataGenerator
from utils.npy_iterator import NpyDirectoryIterator
from utils.explorer import create_dir
import argparse
from os.path import join
from datetime import datetime

# for python 2 & 3 compatibility
try:
    from itertools import izip
except ImportError:
    izip = zip


def train(images_path, density_maps_path, input_shape, epochs, verbosity, batch_size, workers):
    """

    :param images_path: string
    :param density_maps_path: string
    :param input_shape: (int, int)
    :param epochs: int
    :param verbosity: int
    :param batch_size: int
    :param workers: int
    :return:
    """
    if verbosity > 0:
        print('Creating model...')

    crowdnet = CrowdNet()
    model = crowdnet.model_for_training(input_shape)

    if verbosity > 0:
        print('Compiling model...')

    model.compile(optimizer='adam',
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('images_path', help='path to the folder with training images')
    parser.add_argument('density_maps_path', help='path to the folder with corresponding density maps')
    parser.add_argument('input_shape', nargs='+', type=int, help='size of input data, expected as WIDTH HEIGHT CHANNELS')

    parser.add_argument('-e', '--epochs', type=int, default=50, help='number of training epochs, 50 by default')
    parser.add_argument('-v', '--verbosity', action='count', default=0, help='verbosity level, expected to be between 0 and 2')
    parser.add_argument('-b', '--batch_size', default=32, type=int, help='number of images in a training batch')
    parser.add_argument('-w', '--workers', default=4, type=int, help="maximum number of processes to spin up, default is 4")

    args = parser.parse_args()

    train(args.images_path,
          args.density_maps_path,
          tuple(args.input_shape),
          args.epochs,
          args.verbosity,
          args.batch_size,
          args.workers)