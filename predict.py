from __future__ import print_function
from skimage.io import imread
import numpy as np
from os.path import join, basename
from model import get_model
import glob


model_name = 'regression'
grayscale = False
if grayscale:
    channels = 1
else:
    channels = 3

print('Creating {} object...'.format(model_name))
model_obj = get_model(model_name)
model = model_obj.model_for_prediction((256, 256, channels))

print('Loading weights...')
model.load_weights('./checkpoints/regression.hdf5')

img_folder = join('data', 'test', 'patches')
image_names = [basename(x) for x in glob.glob(join(img_folder, '*.jpg'))]

for img_name in image_names:
    img = imread(join(img_folder, img_name), as_grey=grayscale)
    if not grayscale:
        img = np.divide(img, 255.)
    img = np.expand_dims(img, axis=0)
    if grayscale:
        img = np.expand_dims(img, axis=3)

    prediction = model.predict(img)
    print("{}: {}".format(img_name, prediction.item()))