from __future__ import division
from __future__ import print_function
from skimage.io import imread
import matplotlib.pyplot as plt
import numpy as np
from os.path import join, basename
from utils.explorer import create_dir, empty_dir
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
model.load_weights('./out/checkpoints/checkpoint.99.hdf5')
# model.load_weights('./out/CrowdNetMini_2017-07-18-20-55-47.h5')

img_folder = join('data', 'test')
image_names = [basename(x) for x in glob.glob(join(img_folder, '*.jpg'))]


out_path = join('out', 'predictions')
create_dir(out_path)
empty_dir(out_path)

for img_name in image_names:
    print('Making prediction for {}...'.format(img_name))
    img = imread(join(img_folder, img_name), as_grey=grayscale)
    if not grayscale:
        img = np.divide(img, 255.)
    img = np.expand_dims(img, axis=0)
    if grayscale:
        img = np.expand_dims(img, axis=3)

    if model_obj.num_inputs > 1:
        img = [img, img]

    prediction = model.predict(img)
    print("{}: {}".format(img_name, prediction[0]))
    '''
    prediction = np.squeeze(prediction, axis=(0, 3))
    print('count: {}'.format(prediction.sum()))
    print('max: {}'.format(prediction.max()))
    fig = plt.figure(figsize=(18, 12))
    ax = fig.add_subplot(1, 1, 1)
    imgplot = plt.imshow(prediction, cmap='hot', interpolation='bicubic')
    plt.colorbar()
    plt.savefig(join(out_path, '{}_pred.jpg'.format(basename(img_name))), bbox_inches='tight')
    plt.close()
    '''