from base import _ModelBase
from keras.layers import Conv2D, MaxPooling2D, Input, UpSampling2D, concatenate
from keras.models import Model
from keras.regularizers import l2


class QGo4B(_ModelBase):

    def __init__(self):
        self.name = 'QGo4B'

    def _model(self, input_shape):
        input_layer = Input(input_shape)

        channel_axis = 3

        regularizer = l2(0.00001)

        x = Conv2D(32, (9, 9), activation='relu', padding='same', activity_regularizer=regularizer)(input_layer)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

        branch9x9 = Conv2D(16, (9, 9), activation='relu', padding='same', activity_regularizer=regularizer)(x)
        branch7x7 = Conv2D(16, (7, 7), activation='relu', padding='same', activity_regularizer=regularizer)(x)
        branch5x5 = Conv2D(16, (5, 5), activation='relu', padding='same', activity_regularizer=regularizer)(x)
        branch3x3 = Conv2D(16, (3, 3), activation='relu', padding='same', activity_regularizer=regularizer)(x)

        x = concatenate([branch9x9, branch7x7, branch5x5, branch3x3], axis=channel_axis)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

        branch7x7 = Conv2D(16, (7, 7), activation='relu', padding='same', activity_regularizer=regularizer)(x)
        branch5x5 = Conv2D(16, (5, 5), activation='relu', padding='same', activity_regularizer=regularizer)(x)
        branch3x3 = Conv2D(16, (3, 3), activation='relu', padding='same', activity_regularizer=regularizer)(x)

        x = concatenate([branch7x7, branch5x5, branch3x3], axis=channel_axis)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

        x = Conv2D(100, (1, 1), padding='same', activation='relu', activity_regularizer=regularizer)(x)
        x = Conv2D(1, (1, 1), padding='same', activation='relu', activity_regularizer=regularizer)(x)
        x = UpSampling2D(size=(8, 8))(x)

        model = Model(input_layer, x, name='qgo_mini')

        return model

