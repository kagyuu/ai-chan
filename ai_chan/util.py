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


def histogram(m, bins=10):
    """
    ネットワークの重みのヒストグラムを作ります
    ```
    matplotlib の階段グラフに表示する場合には、返値を使って次のようにします
    plt.step(bin_edges[:-1], hist, where='post')
    plt.show()

    ※ 学習が進んだ時にネットワークの重みがどうなるかを見るために、複数のヒストグラムを階段グラフとして色を変えながら
    表示することを目的としています。
    ある学習状態のヒストグラムを見たい場合には plot.histogram(flat) を使ったほうが楽です
    ```
    :param m: 重み行列の配列
    :param bins: ヒストグラムの区間数 (デフォルト10)
    :return hist: 頻度データ
    :return bin_edges: 頻度の下限・上限
    """
    flat = []
    for idx in range(1, len(m)):
        flat.extend(m[idx].flatten().tolist())

    hist, bin_edges = np.histogram(flat, bins=bins)
    return hist, bin_edges
