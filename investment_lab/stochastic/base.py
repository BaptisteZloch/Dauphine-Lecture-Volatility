from abc import ABC, abstractmethod


class StochasticProcess(ABC):

    @abstractmethod
    def simulate(self, *args, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def calibrate(self, *args, **kwargs):
        raise NotImplementedError
