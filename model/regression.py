from .base import _ModelBase
from keras.layers import Conv2D, MaxPooling2D, Input, AvgPool2D, Merge, UpSampling2D, Dense, Dropout, Flatten
from keras.models import Model
from keras.regularizers import l2


class Regression(_ModelBase):

    def __init__(self):
        self.name = 'regression'

    def _model(self, input_shape):
        """

        :param input_shape: (int, int, int)
        :return: Keras sequential model
        """
        input_layer = Input(input_shape)

        regularizer_0 = l2(0.0001)

        x = Conv2D(32, (3, 3), activation='relu', padding='same', activity_regularizer=regularizer_0)(input_layer)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

        x = Conv2D(32, (3, 3), activation='relu', padding='same', activity_regularizer=regularizer_0)(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

        x = Conv2D(32, (3, 3), activation='relu', padding='same', activity_regularizer=regularizer_0)(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

        # x = Conv2D(1, (1, 1), activation='relu', padding='same', activity_regularizer=regularizer_0)(x)

        x = Flatten()(x)

        x = Dense(1024, activation='relu')(x)
        x = Dropout(0.2)(x)

        x = Dense(256, activation='relu')(x)
        x = Dropout(0.2)(x)

        x = Dense(32, activation='relu')(x)
        x = Dropout(0.2)(x)

        x = Dense(8, activation='relu')(x)
        x = Dropout(0.2)(x)

        x = Dense(1)(x)

        model = Model(input_layer, x, name=self.name)

        return model

