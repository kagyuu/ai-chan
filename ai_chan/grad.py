import numpy as np
from abc import ABCMeta, abstractmethod


class Grad(metaclass=ABCMeta):

    @abstractmethod
    def get_rate(self, dEdW, dEdB):
        pass


class Static(Grad):

    def __init__(self, rate=0.001):
        self.rate = rate
        self.dW = None
        self.dB = None

    def get_rate(self, dEdW, dEdB):
        if self.dW is None:
            self.dW = [None]
            self.dB = [None]
            for idx in range(1, len(dEdW)):
                self.dW.append(self.rate * np.ones_like(dEdW[idx]))
                self.dB.append(self.rate * np.ones_like(dEdB[idx]))

        return self.dW, self.dB
