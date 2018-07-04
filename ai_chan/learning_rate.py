from abc import ABCMeta, abstractmethod


class AbstractLearningRate(metaclass=ABCMeta):

    @abstractmethod
    def get_train_rate(self):
        pass


class StaticRate(AbstractLearningRate):

    __rate = 0.001

    def __init__(self, rate):
        self.__rate = rate

    def get_train_rate(self):
        return self.__rate
