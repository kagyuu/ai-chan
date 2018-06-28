import numpy as np


def sigmoid(x):
    """
     Sigmoid (シグモイド関数)
    :param x:
    :return: sigmoid(x)
    """

    return 1.0 / (1.0 + np.exp(-1.0 * x))


def relu(x):
    """
     Rectified Linear Unit (正規化線形関数)
    :param x:
    :return: max(0,x)
    """
    return np.maximum(0, x)


def identity_mapping(x):
    """
     恒等写像です
    :param x:
    :return: xをそのまま返します
    """

    return x


def d_sigmoid(x):
    """
     Sigmoid (シグモイド関数)の導関数
    :param x:
    :return: sigmoid'(x)
    """

    dx = (1.0 - sigmoid(x)) * sigmoid(x)
    return dx


def d_relu(x):
    """
     ReLuの導関数
    :param x:
    :return: relu'(x)
    """

    return np.where(x > 0, 1, 0)


def least_square(d, y):
    """
     自乗誤差を求めます
    :param d:教師データ(expected)
    :param y:予想(actual)
    :return: 自乗誤差
    """

    return np.sum(np.square(d - y)) / 2
