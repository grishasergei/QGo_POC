from __future__ import print_function
from model import get_model
from utils.explorer import create_dir, empty_dir
import argparse
from os.path import join
from datetime import datetime
import numpy as np
from keras.callbacks import TensorBoard, ModelCheckpoint, TerminateOnNaN, LearningRateScheduler, ReduceLROnPlateau
from sklearn.model_selection import train_test_split


# for python 2 & 3 compatibility
try:
    from itertools import izip
except ImportError:
    izip = zip


def train_in_memory(model_name, x, y, epochs, verbosity, batch_size, learning_rate,
                    tensorboard, checkpoint, validation_split):
    seed = 7
    np.random.seed(seed)

    # create model object
    model_obj = get_model(model_name)

    if verbosity > 0:
        print('Creating {} model...'.format(model_obj.name))

    if verbosity > 0:
        print('Compiling model...')

    model = model_obj.model_for_training(x.shape[1:], 'mean_squared_error', 'adam')

    # callbacks
    if verbosity > 0:
        print('Creating callbacks...')

    callbacks = []

    # termintate training on NAN
    callbacks.append(TerminateOnNaN())

    # learning rate scheduler
    # callbacks.append(LearningRateScheduler(lr_scheduler))

    if tensorboard:
        log_dir = join('out', 'logs', 'tensorboard')
        empty_dir(log_dir)
        callbacks.append(TensorBoard(log_dir=log_dir,
                                     histogram_freq=1,
                                     write_graph=False,
                                     write_grads=False,
                                     write_images=False))

    if checkpoint:
        checkpoint_path = join('out', 'checkpoints')
        create_dir(checkpoint_path)
        empty_dir(checkpoint_path)

        checkpointer = ModelCheckpoint(join(checkpoint_path, 'checkpoint.{epoch:02d}.hdf5'), verbose=0)
        callbacks.append(checkpointer)

        if verbosity > 0:
            print('Checkpoints will be saved to {}'.format(checkpoint_path))

    if verbosity > 0:
        print('Starting training...')

    if validation_split:
        x_train, x_validation, y_train, y_validation = train_test_split(x, y, test_size=validation_split, random_state=seed)
        validation_data = (x_validation, y_validation)
    else:
        if verbosity > 0:
            print('No validation split')
        x_train = x
        y_train = y
        validation_data = None

    if verbosity > 0:
        print('Shape before oversampling:')
        print('x: {}'.format(x_train.shape))
        print('y: {}'.format(y_train.shape))

    # oversample patches of medium density
    indices = np.where(y_train <= 20)
    indices = np.where(y_train[indices] > 5)
    y_train = np.concatenate((y_train, y_train[indices]))
    x_train = np.concatenate((x_train, x_train[indices]))

    if verbosity > 0:
        print('Shape after oversampling:')
        print('x: {}'.format(x_train.shape))
        print('y: {}'.format(y_train.shape))

    if verbosity > 1:
        model.summary()

    model.fit(x_train, y_train,
              validation_data=validation_data,
              batch_size=batch_size,
              epochs=epochs,
              verbose=verbosity,
              shuffle=True,
              callbacks=callbacks)

    # save model after training
    create_dir('out')
    out_path = join('out', '{}_{}.h5'.format(model_obj.name, datetime.now().strftime("%Y-%m-%d-%H-%M-%S")))
    model.save(out_path)
    if verbosity > 0:
        print('model has been saved to {}'.format(out_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('model', help='model name')
    parser.add_argument('x_path', help='full path to the x array')
    parser.add_argument('y_path', help='full path to the y array')

    parser.add_argument('-e', '--epochs', type=int, default=50, help='number of training epochs, 50 by default')
    parser.add_argument('-v', '--verbosity', action='count', default=0, help='verbosity level, expected to be between 0 and 2')
    parser.add_argument('-b', '--batch_size', default=32, type=int, help='number of images in a training batch')
    parser.add_argument('-w', '--workers', default=4, type=int, help="maximum number of processes to spin up, default is 4")
    parser.add_argument('-lr', '--learning_rate', type=float, default=0.00001, help='learning rate')
    parser.add_argument('-tb', '--tensorboard', action='store_true', help='activate tensorboard visualization, log is written to ./out/logs/tensorboard')
    parser.add_argument('-cp', '--check_point', action='store_true', help='save model checkpoints')
    parser.add_argument('-vs', '--validation_split', type=float, default=0, help='percentage of data to be used as validation set')

    args = parser.parse_args()

    x = np.load(args.x_path)
    x = x / 255.
    y = np.load(args.y_path)

    if args.verbosity > 0:
        print('Training model in memory')
    train_in_memory(args.model,
                    x,
                    y,
                    args.epochs,
                    args.verbosity,
                    args.batch_size,
                    args.learning_rate,
                    args.tensorboard,
                    args.check_point,
                    args.validation_split)
