class _ModelBase(object):
    """
    Abstract base class. All models must inherit from it.
    """
    num_inputs = 1
    name = 'base'
    input_shape = None
    loss = None
    optimizer = None

    def __init__(self):
        pass

    def _model(self, input_shape):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.model_for_training(self.input_shape, self.loss, self.optimizer)

    def model_for_training(self, input_shape, loss, optimizer):
        """

        :param input_shape:
        :return:
        """
        model = self._model(input_shape)
        model.compile(loss=loss, optimizer=optimizer)
        return model

    def model_for_prediction(self, input_shape):
        """

        :param input_shape:
        :return:
        """
        return self._model(input_shape)