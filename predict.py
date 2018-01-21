from __future__ import print_function
import numpy as np
from model import get_model
import argparse


def predict(x, model_name, weights_path):
    model_obj = get_model(model_name)

    x = x / 255.
    if x.ndim == 3:
        x = np.expand_dims(x, axis=0)

    model = model_obj.model_for_prediction(x[0].shape)

    model.load_weights(weights_path)

    predictions = model.predict(x)

    return predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('x_path', help='full path to the x array')
    parser.add_argument('model', help='model name')
    parser.add_argument('weights_path', help='full path to the model weights file')
    parser.add_argument('-v', '--verbose', action='store_true', help='verbose')

    args = parser.parse_args()
    x = np.load(args.x_path)
    predictions = predict(x, args.model, args.weights_path)

    if args.verbose:
        for i, prediction in enumerate(predictions):
            print('{}: {}'.format(i, int(round(prediction[0]))))

