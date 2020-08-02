import abc


class Distribution:
    def __init__(self):
        pass

    @abc.abstractmethod
    def log_prob(self, x, *args, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def rvs(self, key, shape, *args, **kwargs):
        raise NotImplementedError()
