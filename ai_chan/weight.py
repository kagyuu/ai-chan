from abc import ABCMeta, abstractmethod
import numpy as np


class WeightDecay(metaclass=ABCMeta):
    """
    重み制限の抽象クラス.
    """

    @abstractmethod
    def r(self, dEdW, dEdB):
        """
        正則化項 rw,rbを返します
        :param w 重み行列
        :param b バイアス
        :return: 正則化項 rw,rb
        """
        pass


class NoDecay(WeightDecay):
    """
    0行列の正則化項を返します.
    (重みの正則化は行いません)
    """

    def __init__(self):
        pass

    def r(self, w, b):
        dRdW = [None]
        dRdB = [None]

        for idx in range(1, len(w)):
            dRdW.append(np.zeros_like(w[idx]))
            dRdB.append(np.zeros_like(b[idx]))

        return dRdW, dRdB


class L1Decay(WeightDecay):
    """
    L1正則化項を返します.
    ∂R(x)    ∂ (|W11| + |W22| + |W33| + ... + |Wnn|)    ∂(|Wij|)      1 (if Wij >  0)
    ------ = ---------------------------------------- = --------- =  -1 (if Wij <= 0)
    ∂Wij     ∂Wij                                       ∂Wij
    全ての重み Wij を、0方向へ rate だけ近づけます
    """

    def __init__(self, rate=0.1):
        self.rate = rate

    def r(self, w, b):
        dRdW = [None]
        dRdB = [None]

        for idx in range(1, len(w)):
            dRdW.append(self.rate * np.where(w[idx] > 0.0, 1.0, -1.0))
            dRdB.append(self.rate * np.where(b[idx] > 0.0, 1.0, -1.0))

        return dRdW, dRdB


class L2Decay(WeightDecay):
    """
    L2正則化項を返します.
    ∂R(x)    ∂ 1/2( W11^2 + W22^2 + W33^2 + ... + Wnn^2)    ∂ 1/2(Wij^2)
    ------ = -------------------------------------------- = ------------- =  Wij
    ∂Wij     ∂Wij                                           ∂Wij
    全ての重み Wij を、0方向へ rate * Wij だけ近づけます
    """

    def __init__(self, rate=0.1):
        self.rate = rate

    def r(self, w, b):
        dRdW = [None]
        dRdB = [None]

        for idx in range(1, len(w)):
            dRdW.append(self.rate * w[idx])
            dRdB.append(self.rate * b[idx])

        return dRdW, dRdB


class LmaxDecay(WeightDecay):
    """
    L∞正則化項を返します.
    ∂R(x)    ∂ 1/∞( W11^∞ + W22^∞ + W33^∞ + ... + Wnn^∞)   ∂ 1/∞(Wij^∞)     Wij (if Wij is abs_max(W))
    ------ = ------------------------------------------- = ------------- =  0   (otherwise)
    ∂Wij     ∂Wij                                          ∂Wij
    最も大きい重み Wij を、0方向へ rate * Wij だけ近づけます
    """

    def __init__(self, rate=0.1):
        self.rate = rate

    def r(self, w, b):
        dRdW = [None]
        dRdB = [None]

        for idx in range(1, len(w)):
            maxW = np.max(w[idx])
            maxB = np.max(b[idx])
            minW = np.min(w[idx])
            minB = np.min(b[idx])
            abs_maxW = maxW if maxW + minW > 0.0 else minW
            abs_maxB = maxB if maxB + minB > 0.0 else minB

            dRdW.append(self.rate * np.where(w[idx] == abs_maxW, abs_maxW, 0.0))
            dRdB.append(self.rate * np.where(b[idx] == abs_maxB, abs_maxB, 0.0))

        return dRdW, dRdB
