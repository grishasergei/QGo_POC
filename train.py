from model.crowdnet import crowdnet
from model.mini import qgo_mini
from keras.preprocessing.image import ImageDataGenerator
from utils.npy_iterator import NpyDirectoryIterator
from itertools import izip
import argparse


def train(images_path, density_maps_path, input_shape, epochs, verbosity):
    """

    :param images_path:
    :param density_maps_path:
    :param input_shape:
    :param epochs:
    :param verbosity:
    :return:
    """
    model = crowdnet(input_shape)
    model.compile(optimizer='SGD',
                  loss='mean_squared_error')

    image_datagen = ImageDataGenerator(rescale=1./255)
    density_map_datagen = ImageDataGenerator()

    seed = 1

    image_generator = image_datagen.flow_from_directory(
        images_path,
        target_size=input_shape[0:2],
        color_mode='rgb',
        class_mode=None,
        shuffle=True,
        seed=seed
    )

    density_map_generator = NpyDirectoryIterator(
        density_maps_path,
        density_map_datagen,
        target_size=input_shape[0:2],
        shuffle=True,
        seed=seed
    )

    train_generator = (([x, x], y) for x, y in izip(image_generator, density_map_generator))

    model.fit_generator(
        train_generator,
        steps_per_epoch=47,
        epochs=epochs,
        verbose=verbosity
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('images_path', help='path to the folder with training images')
    parser.add_argument('density_maps_path', help='path to the folder with corresponding density maps')
    parser.add_argument('input_shape', nargs='+', type=int, help='size of input data, expected as WIDTH HEIGHT CHANNELS')

    parser.add_argument('--epochs', type=int, default=50, help='number of training epochs, 50 by default')
    parser.add_argument('-v', '--verbosity', action='count', default=0, help='verbosity level, expected to be between 0 and 2')

    args = parser.parse_args()

    train(args.images_path,
          args.density_maps_path,
          tuple(args.input_shape),
          args.epochs,
          args.verbosity)