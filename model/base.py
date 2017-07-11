class _ModelBase(object):
    """
    Abstract base class. All models must inherit from it.
    """
    num_inputs = 1
    name = 'base'

    def _model(self, input_shape):
        raise NotImplementedError()

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