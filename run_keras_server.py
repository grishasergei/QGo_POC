import flask
from model import get_model
from prepare.patch import patches_generator
import numpy as np
from keras.preprocessing.image import img_to_array
from PIL import Image
import io
import argparse


app = flask.Flask(__name__)

# This approach is suitable ONLY for the default single threaded Flask server
model = None


@app.route("/predict", methods=["POST"])
def predict():
    data = {"success": False}

    if flask.request.method != "POST":
        return flask.jsonify(data)

    if not flask.request.files.get("image"):
        return flask.jsonify(data)

    # read image in PIL format
    image = flask.request.files["image"].read()
    image = Image.open(io.BytesIO(image))
    data["image"] = [{"width": image.width},
                     {"height": image.height}]

    data["patch"] = [{"width": 256},
                     {"height": 256}]

    patches = prepare_image(image)

    predictions = model.predict(patches)

    data["patches"] = []

    total_count = 0

    for i, prediction in enumerate(predictions):
        p = int(round(prediction[0]))
        r = {"patch": i, "count": p}
        data["patches"].append(r)
        total_count += p

    data["total_count"] = total_count
    data["success"] = True

    return flask.jsonify(data)


def load_model():
    """
    Loads a Keras model.
    :return: an instance of a Keras Model
    """
    # TODO make model name, input size and weights path configurable
    # TODO move to a separate module
    global model
    model_wrapper = get_model('qgodeep')
    model = model_wrapper.model_for_prediction((256, 256, 3))
    model.load_weights('./weights/qgodeep_2018-01-21-15-17-37.h5')


def prepare_image(image):
    """
    Converts an image into a numpy array of patches (256, 256) patches.
    Rescales to 0..1 range if needed.
    :param image: an image in PIL format
    :return: numpy array of image patches
    """
    # TODO make patch size configurable
    # TODO move to a separate module
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = img_to_array(image)

    if np.amax(image) > 1:
        image = image / 255.

    patches = np.array(list(patches_generator(image, (256, 256), 0, 0)))

    return patches


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--port', type=int, default=5000, help='server port')
    args = parser.parse_args()

    print('Loading a model...')
    load_model()

    print('Starting the server...')
    app.run(port=args.port)

