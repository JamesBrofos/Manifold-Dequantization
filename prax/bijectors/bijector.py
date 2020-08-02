import abc


class Bijector:
    def __init__(self):
        pass

    @abc.abstractmethod
    def forward(self, x, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def inverse(self, y, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def forward_log_det_jacobian(self, x, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def inverse_log_det_jacobian(self, y, **kwargs):
        raise NotImplementedError()
