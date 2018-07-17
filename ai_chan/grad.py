import numpy as np
from abc import ABCMeta, abstractmethod


class Grad(metaclass=ABCMeta):

    @abstractmethod
    def eta(self, dEdW, dEdB):
        """
        学習係数η(イータ)を返します
        :param dEdW: ∂E/∂W
        :param dEdB: ∂E/∂b
        :return: 学習係数
        """
        pass


class Static(Grad):
    """
    常に、コンストラクタで与えた係数を返します
    """

    def __init__(self, rate=0.001):
        self.rate = rate

    def eta(self, dEdW, dEdB):
        hw = [None]
        hb = [None]

        h = self.rate
        for idx in range(1, len(dEdW)):
            hw.append(h * np.ones_like(dEdW[idx]))
            hb.append(h * np.ones_like(dEdB[idx]))

        return hw, hb


class Shrink(Grad):
    """
    呼ばれた回数tを数えていて、コンストラクタで与えた係数/t+1 を返します
    """

    def __init__(self, rate=0.001):
        self.rate = rate
        self.cnt = 0.0

    def eta(self, dEdW, dEdB):
        hw = [None]
        hb = [None]

        self.cnt += 1.0
        h = self.rate / self.cnt # これは rate / (cnt+1)
        for idx in range(1, len(dEdW)):
            hw.append(h * np.ones_like(dEdW[idx]))
            hb.append(h * np.ones_like(dEdB[idx]))

        return hw, hb
