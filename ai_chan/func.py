from abc import ABCMeta, abstractmethod
import numpy as np


class ActivateFunction(metaclass=ABCMeta):
    """
    活性化関数の抽象クラス.
    """

    @abstractmethod
    def calc(self, x):
        """
        x に対する値を計算します
        """
        pass

    @abstractmethod
    def differential(self, x):
        """
        x に対する微分値を計算します
        """
        pass

    @abstractmethod
    def delta(self, d, y):
        """
        出力層にこの関数を使った場合のδ(L)を計算します
        """
        pass

    def name(self):
        """
        この関数の名称を返却します
        """
        pass


class IdentityMapping(ActivateFunction):
    """
    恒等写像
    """
    def calc(self, x):
        return x

    def differential(self, x):
        return np.ones_like(x)

    def delta(self, d,y):
        return y - d

    def name(self):
        return "恒等写像(Identity Mapping)"


class Sigmoid(ActivateFunction):
    """
    シグモイド関数
    """
    def calc(self, x):
        return 1.0 / (1.0 + np.exp(-1.0 * x))

    def differential(self, x):
        s = self.calc(x)
        return (1.0 - s) * s

    def delta(self, d,y):
        return d - y

    def name(self):
        return "Sigmoid"


class Tanh(ActivateFunction):
    """
    双曲線正接関数(tanh)
    """
    def calc(self, x):
        # return (np.exp(x) - np.exp(-1.0 * x)) / (np.exp(x) + np.exp(-1.0 * x))
        return np.tanh(x)

    def differential(self, x):
        #return 4.0 / np.power((np.exp(x) + np.exp(-1.0 * x)), 2)
        return 1.0 / (np.cosh(x) ** 2)

    def delta(self, d,y):
        return d - y

    def name(self):
        return "双曲線正接関数(tanh)"


class ReLu(ActivateFunction):
    """
    Rectified Linear Unit (正規化線形関数)
    """
    def calc(self, x):
        return np.maximum(0, x)

    def differential(self, x):
        return np.where(x > 0, 1, 0)

    def delta(self, d,y):
        return y - d

    def name(self):
        return "正規化線形関数(ReLu)"
