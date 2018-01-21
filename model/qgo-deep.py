from .base import _ModelBase
from keras.layers import Conv2D, MaxPooling2D, Input, AvgPool2D, Merge, UpSampling2D, Dense, Dropout, Flatten, concatenate
from keras.models import Model


class QGoDeep(_ModelBase):

    def __init__(self, *args, **kwargs):
        super(QGoDeep, self).__init__(*args, **kwargs)
        self.name = 'qgodeep'

    def _model(self, input_shape):
        input_layer = Input(input_shape)

        x = Conv2D(32, (3, 3), strides=(2, 2), activation='relu', padding='same')(input_layer)
        x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

        x = Conv2D(64, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
        x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

        x = Conv2D(128, (3, 3), strides=(2, 2), activation='relu', padding='same')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same')(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

        x = Flatten()(x)

        x = Dense(30, activation='relu')(x)
        x = Dropout(0.3)(x)

        x = Dense(1)(x)

        model = Model(input_layer, x, name=self.name)

        return model

