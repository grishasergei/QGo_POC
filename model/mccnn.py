from .base import _ModelBase
from keras.layers import Conv2D, MaxPooling2D, Input, UpSampling2D, concatenate, Conv2DTranspose
from keras.models import Model
from keras.regularizers import l2


class McCNN(_ModelBase):

    def __init__(self):
        self.name = 'mccnn'

    def _model(self, input_shape):
        input_layer = Input(input_shape)

        channel_axis = 3

        regularizer = l2(0.0)

        # 9x9
        branch9x9 = Conv2D(16, (9, 9), activation='relu', padding='same', activity_regularizer=regularizer)(input_layer)
        branch9x9 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(branch9x9)
        branch9x9 = Conv2D(32, (7, 7), activation='relu', padding='same', activity_regularizer=regularizer)(branch9x9)
        branch9x9 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(branch9x9)
        branch9x9 = Conv2D(16, (7, 7), activation='relu', padding='same', activity_regularizer=regularizer)(branch9x9)
        branch9x9 = Conv2D(8, (7, 7), activation='relu', padding='same', activity_regularizer=regularizer)(branch9x9)

        # 7x7
        branch7x7 = Conv2D(20, (7, 7), activation='relu', padding='same', activity_regularizer=regularizer)(input_layer)
        branch7x7 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(branch7x7)
        branch7x7 = Conv2D(40, (5, 5), activation='relu', padding='same', activity_regularizer=regularizer)(branch7x7)
        branch7x7 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(branch7x7)
        branch7x7 = Conv2D(20, (5, 5), activation='relu', padding='same', activity_regularizer=regularizer)(branch7x7)
        branch7x7 = Conv2D(10, (5, 5), activation='relu', padding='same', activity_regularizer=regularizer)(branch7x7)

        # 5x5
        branch5x5 = Conv2D(24, (5, 5), activation='relu', padding='same', activity_regularizer=regularizer)(input_layer)
        branch5x5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(branch5x5)
        branch5x5 = Conv2D(48, (3, 3), activation='relu', padding='same', activity_regularizer=regularizer)(branch5x5)
        branch5x5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(branch5x5)
        branch5x5 = Conv2D(24, (3, 3), activation='relu', padding='same', activity_regularizer=regularizer)(branch5x5)
        branch5x5 = Conv2D(12, (3, 3), activation='relu', padding='same', activity_regularizer=regularizer)(branch5x5)

        x = concatenate([branch9x9, branch7x7, branch5x5], axis=channel_axis)

        x = UpSampling2D(size=(4, 4))(x)

        x = Conv2D(1, (1, 1), padding='same', activation='sigmoid')(x)

        model = Model(input_layer, x, name=self.name)

        return model

