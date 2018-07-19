import numpy as np
import matplotlib.pyplot as plt


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


def draw_hist(m, label, bins=50, min_max=(-10.0, 10.0)):
    """
    行列のリストを展開して、matplotlib の　histogram を作ります
    :param m: 行列のリスト
    :param label: データの名称
    :param bins: 区切り数(デフォルト50). Noneを指定すると自動
    :param min_max: データの幅(デフォルト-10~10). Noneを指定すると自動
    :return:
    """
    flat = []
    for idx in range(1, len(m)):
        flat.extend(m[idx].flatten().tolist())
    plt.hist(flat, label=label, bins=bins, range=min_max, alpha=0.5)
