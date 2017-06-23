from keras.layers import Conv2D, MaxPooling2D, Input, AvgPool2D, Merge, UpSampling2D
from keras.models import Model, Sequential
from keras.utils.data_utils import get_file
from keras.applications.vgg16 import WEIGHTS_PATH_NO_TOP


class CrowdNet:
    """
    CorwdNet model
    https://arxiv.org/pdf/1608.06197.pdf
    """

    def __init__(self):
        self.name = 'CrowdNet'

    def _deep_branch(self, input_shape=None):
        """
        Deep part of the CrowdNet model based on VGG16 model
        :param input_shape: tuple of int, shape of input images
        :return: Sequential Keras model with ImageNet weights
        """

        input_layer = Input(input_shape)

        # Block 1
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='deep_block1_conv1')(input_layer)
        x = Conv2D(64, (3, 3), activation='relu', padding='same', name='deep_block1_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='deep_block1_pool')(x)

        # Block 2
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='deep_block2_conv1')(x)
        x = Conv2D(128, (3, 3), activation='relu', padding='same', name='deep_block2_conv2')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='deep_block2_pool')(x)

        # Block 3
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='deep_block3_conv1')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='deep_block3_conv2')(x)
        x = Conv2D(256, (3, 3), activation='relu', padding='same', name='deep_block3_conv3')(x)
        x = MaxPooling2D((2, 2), strides=(2, 2), padding='same', name='deep_block3_pool')(x)

        # Block 4
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='deep_block4_conv1')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='deep_block4_conv2')(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='deep_block4_conv3')(x)
        x = MaxPooling2D((3, 3), strides=(1, 1), padding='same', name='deep_block4_pool')(x)

        # Block 5
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='deep_block5_conv1', dilation_rate=2)(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='deep_block5_conv2', dilation_rate=2)(x)
        x = Conv2D(512, (3, 3), activation='relu', padding='same', name='deep_block5_conv3', dilation_rate=2)(x)

        model = Model(input_layer, x, name='crowdnet_deep')

        # load weights
        weights_path = get_file('vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                WEIGHTS_PATH_NO_TOP,
                                cache_subdir='cache')
        model.load_weights(weights_path)

        return model

    def _shallow_branch(self, input_shape=None):
        """
        Shallow part of the CorwdNet model
        :param input_shape: tuple of int, shape of input images
        :return: Keras Sequential model
        """

        input_layer = Input(input_shape)

        # Block 1
        x = Conv2D(24, (5, 5), activation='relu', padding='same', name='shallow_block1_conv1')(input_layer)
        x = AvgPool2D((5, 5), strides=(2, 2), padding='same', name='shallow_block1_pool')(x)

        # Block 2
        x = Conv2D(24, (5, 5), activation='relu', padding='same', name='shallow_block2_conv1')(x)
        x = AvgPool2D((5, 5), strides=(2, 2), padding='same', name='shallow_block2_pool')(x)

        # Block 3
        x = Conv2D(24, (5, 5), activation='relu', padding='same', name='shallow_block3_conv1')(x)
        x = AvgPool2D((5, 5), strides=(2, 2), padding='same', name='shallow_block3_pool')(x)

        model = Model(input_layer, x, name='crowdnet_shallow')

        return model

    def _merge(self, deep, shallow):
        """
        Merge deep and shallow networks into one combined model
        :param deep: Keras Sequential model
        :param shallow: Keras Sequential model
        :return: Keras Sequential model
        """
        model = Sequential()
        model.add(Merge([deep, shallow], mode='concat', name='top_merge'))
        model.add(Conv2D(1, (1, 1,), activation='sigmoid', padding='same', name='top_conv1'))
        model.add(UpSampling2D(size=(8, 8), name='top_upsampling'))

        return model

    def model_for_training(self, input_shape):
        """
        CrowdNet model configured for training. Deep part based of VGG16 is freezed
        :param input_shape: tuple of int, shape of input images
        :return: Keras Sequential model, not compiled
        """
        deep_part = self._deep_branch(input_shape)
        for layer in deep_part.layers:
            layer.trainable = False

        shallow_part = self._shallow_branch(input_shape)

        return self._merge(deep_part, shallow_part)

    def model_for_fine_tuning(self, input_shape):
        """
        CrowdNet model configured for training. Deep part based of VGG16 is freezed
        except the last convolution block
        :param input_shape: tuple of int, shape of input images
        :return: Keras Sequential model, not compiled
        """
        deep_part = self._deep_branch(input_shape)
        for layer in deep_part.layers[0:14]:
            layer.trainable = False

        shallow_part = self._shallow_branch(input_shape)

        return self._merge(deep_part, shallow_part)

