from keras.layers import Conv2D, MaxPooling2D, Input, AvgPool2D, Merge, UpSampling2D
from keras.models import Model


class QgoMini:

    def __init__(self):
        self.name = 'qgo_mini'
        self.num_inputs = 1

    def _model(self, input_shape):
        """

        :param input_shape: (int, int, int)
        :return: Keras sequential model
        """
        input_layer = Input(input_shape)

        # Block 1
        x = Conv2D(32, (5, 5), activation='relu', padding='same')(input_layer)
        x = Conv2D(32, (5, 5), activation='relu', padding='same')(x)
        x = AvgPool2D((5, 5), strides=(2, 2), padding='same')(x)

        # Block 2
        x = Conv2D(32, (5, 5), activation='relu', padding='same')(x)
        x = Conv2D(32, (5, 5), activation='relu', padding='same')(x)
        x = AvgPool2D((5, 5), strides=(2, 2), padding='same')(x)

        # Block 3
        x = Conv2D(32, (5, 5), activation='relu', padding='same')(x)
        x = Conv2D(32, (5, 5), activation='relu', padding='same')(x)
        x = AvgPool2D((5, 5), strides=(2, 2), padding='same')(x)

        # Block 5
        x = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(x)
        x = UpSampling2D(size=(8, 8))(x)

        model = Model(input_layer, x, name='qgo_mini')

        return model

    def model_for_training(self, input_shape):
        """

        :param input_shape:
        :return:
        """
        return self._model(input_shape)

    def model_for_prediction(self, input_shape):
        """

        :param input_shape:
        :return:
        """
        return self._model(input_shape)