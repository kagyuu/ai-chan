from abc import ABCMeta, abstractmethod


class AbstractLearningRate(metaclass=ABCMeta):

    @abstractmethod
    def get_train_rate(self):
        pass


class StaticRate(AbstractLearningRate):

    def __init__(self, rate = 0.001):
        self.rate = rate

    def get_train_rate(self):
        return self.rate
