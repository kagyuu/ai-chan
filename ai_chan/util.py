import numpy as np

def debug(msg, *args):
    print(msg.format(*args))

def least_square_average(d, y):
    """
     自乗誤差平均を求めます
    :param d:教師データ(expected)
    :param y:予想(actual)
    :return: 自乗誤差平均
    """
    return (np.sum(np.square(d - y)) / 2.0) / float(len(d))