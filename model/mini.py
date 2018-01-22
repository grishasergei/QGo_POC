from .base import _ModelBase
from keras.layers import Conv2D, MaxPooling2D, Input, Flatten, Dense, Dropout
from keras.models import Model


class QgoMini(_ModelBase):

    def __init__(self, *args, **kwargs):
        super(QgoMini, self).__init__(*args, **kwargs)
        self.name = 'qgomini'

    def _model(self, input_shape):
        input_layer = Input(input_shape)

        x = Conv2D(8, (5, 5), activation='relu')(input_layer)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

        x = Conv2D(16, (5, 5), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

        x = Conv2D(32, (5, 5), activation='relu')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

        x = Flatten()(x)
        x = Dropout(0.5)(x)

        x = Dense(128, activation='relu')(x)
        x = Dropout(0.5)(x)

        x = Dense(1)(x)

        model = Model(input_layer, x, name=self.name)

        return model

