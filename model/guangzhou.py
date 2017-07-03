from keras.layers import Conv2D, MaxPooling2D, Input, \
    UpSampling2D, concatenate, BatchNormalization, \
    Activation, AveragePooling2D
from keras.models import Model
from keras.regularizers import l2


class GuangzhouNet:
    """
    https://arxiv.org/pdf/1702.02359.pdf
    """

    def __init__(self):
        self.name = 'guangzhou_net'
        self.num_inputs = 1

    def _conv2d_bn(self, x, filters, conv_size, bn_axis):
        x = Conv2D(filters,
                   (conv_size, conv_size),
                   padding='same')(x)
        x = BatchNormalization(axis=bn_axis, scale=False)(x)
        x = Activation('relu')(x)
        return x

    def _model(self, input_shape):
        """

        :param input_shape: (int, int, int)
        :return: Keras sequential model
        """
        input_layer = Input(input_shape)
        channel_axis = 3

        # Block 1
        x = self._conv2d_bn(input_layer, 64, 9, channel_axis)

        # Mixed 1
        branch_9x9 = self._conv2d_bn(x, 16, 9, channel_axis)
        branch_7x7 = self._conv2d_bn(x, 16, 7, channel_axis)
        branch_5x5 = self._conv2d_bn(x, 16, 5, channel_axis)
        branch_3x3 = self._conv2d_bn(x, 16, 3, channel_axis)

        x = concatenate([branch_9x9, branch_7x7,
                         branch_5x5, branch_3x3], axis=channel_axis)

        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

        # Mixed 2
        branch_9x9 = self._conv2d_bn(x, 32, 9, channel_axis)
        branch_9x9 = self._conv2d_bn(branch_9x9, 32, 9, channel_axis)

        branch_7x7 = self._conv2d_bn(x, 32, 7, channel_axis)
        branch_7x7 = self._conv2d_bn(branch_7x7, 32, 7, channel_axis)

        branch_5x5 = self._conv2d_bn(x, 32, 5, channel_axis)
        branch_5x5 = self._conv2d_bn(branch_5x5, 32, 7, channel_axis)

        branch_3x3 = self._conv2d_bn(x, 32, 3, channel_axis)
        branch_3x3 = self._conv2d_bn(branch_3x3, 32, 3, channel_axis)

        x = concatenate([branch_9x9, branch_7x7,
                         branch_5x5, branch_3x3], axis=channel_axis)

        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(x)

        # Mixed 3
        branch_7x7 = self._conv2d_bn(x, 64, 7, channel_axis)
        branch_7x7 = self._conv2d_bn(branch_7x7, 64, 7, channel_axis)

        branch_5x5 = self._conv2d_bn(x, 64, 5, channel_axis)
        branch_5x5 = self._conv2d_bn(branch_5x5, 64, 7, channel_axis)

        branch_3x3 = self._conv2d_bn(x, 64, 3, channel_axis)
        branch_3x3 = self._conv2d_bn(branch_3x3, 64, 3, channel_axis)

        x = concatenate([branch_7x7,
                         branch_5x5, branch_3x3], axis=channel_axis)

        x = AveragePooling2D(pool_size=(5, 5), strides=(2, 2), padding='same')(x)

        x = self._conv2d_bn(x, 1000, 1, channel_axis)
        x = self._conv2d_bn(x, 1, 1, channel_axis)
        x = UpSampling2D(size=(8, 8))(x)

        model = Model(input_layer, x, name=self.name)

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