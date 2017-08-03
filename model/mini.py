from .base import _ModelBase
from keras.layers import Conv2D, MaxPooling2D, Input, AvgPool2D, Merge, UpSampling2D
from keras.models import Model
from keras.regularizers import l2


class QgoMini(_ModelBase):

    def __init__(self):
        self.name = 'qgo_mini'

    def _model(self, input_shape):
        """

        :param input_shape: (int, int, int)
        :return: Keras sequential model
        """
        input_layer = Input(input_shape)

        regularizer_0 = l2(0.0001)

        x = Conv2D(24, (5, 5), activation='relu', padding='same', activity_regularizer=regularizer_0)(input_layer)
        x = AvgPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

        x = Conv2D(24, (5, 5), activation='relu', padding='same', activity_regularizer=regularizer_0)(x)
        x = AvgPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

        x = Conv2D(24, (5, 5), activation='relu', padding='same', activity_regularizer=regularizer_0)(x)
        x = AvgPool2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

        x = Conv2D(1, (1, 1), padding='same', activation='relu', activity_regularizer=regularizer_0)(x)
        x = UpSampling2D(size=(8, 8))(x)

        model = Model(input_layer, x, name='qgo_mini')

        return model

