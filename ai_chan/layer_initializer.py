import numpy as np


def init_seq_layer(in_size, out_size):
    """
     1レイヤ分をシーケンス値で初期化します.
    順伝搬・逆伝搬の検証用に使います
    :param in_size: 入力サイズ
    :param out_size: 出力サイズ
    :return w: 重み行列
    :return b: バイアス
    """

    w = np.array(range(0, out_size * in_size)).reshape(out_size, in_size)
    b = np.array(range(0, out_size)).reshape(1, out_size).T
    return w, b


def init_random_layer(in_size, out_size):
    """
     1レイヤ分をランダム(平均0, 分散1)に初期化します.
    :param in_size: 入力サイズ
    :param out_size: 出力サイズ
    :return w: 重み行列
    :return b: バイアス
    """
    w = np.random.normal(0, 1, (out_size, in_size))
    b = np.random.normal(0, 1, (1, out_size)).T
    return w, b


def init_xavier_layer(in_size, out_size):
    """
     1レイヤ分をランダム(平均0, 分散√in_size)に初期化します.
     各層の出力が、平均0 分散1 になるような初期ネットワークを作成します.
    :param in_size: 入力サイズ
    :param out_size: 出力サイズ
    :return w: 重み行列
    :return b: バイアス
    """
    w = np.random.normal(0, np.sqrt(in_size), (out_size, in_size))
    b = np.random.normal(0, np.sqrt(in_size), (1, out_size)).T
    return w, b


def init_he_layer(in_size, out_size):
    """
     1レイヤ分をランダム(平均0, 分散(√(in_size/2)に初期化します.
     各層の出力が、平均0 分散2 になるような初期ネットワークを作成します.
     ReLuでは、半分(x<0)が0になるので、有効な部分(0≧x)の分散を2倍する.
    :param in_size: 入力サイズ
    :param out_size: 出力サイズ
    :return w: 重み行列
    :return b: バイアス
    """
    w = np.random.normal(0, np.sqrt(in_size/2), (out_size, in_size))
    b = np.random.normal(0, np.sqrt(in_size/2), (1, out_size)).T
    return w, b


def init_normalize_layer(data, expect):
    """
    入力データを標準化スコアに変換するレイヤを作ります.
    標準化スコア(n,i) = (x(n,i) - avr_x(n) / σ(n) = x(n,i)/σ(n) - avr_x(n)/σ(n)
    なので、
    u = wx+b
    w = 1/σ
    b = -avr_x/σ
    としたとき、このニューラルネットは、標準化スコアを出力させることができます。

    ※ 計算上 分散σ(n) が、非常に小さくなるのを防ぐために σ(n) には 10e-7 を加えています。
    :param data: 入力データ
    :param expect: 期待出力
    :return w: 重み行列
    :return b: バイアス
    :return data: 入力データを標準化スコアにしたもの
    """
    sigma = np.sqrt(np.var(data, axis=1) ** 2 + 10e-7)
    average = np.mean(data, axis=1)

    # 対角成分が sigma[i] な対角行列を作ります(対角成分以外0)
    w = np.diag(1.0 / sigma)
    # b[i] = -average[i]/sigma[i] なバイアスベクトルb(N行1列)を作ります
    b = np.array([- average / sigma]).T

    return w, b, np.dot(w, data) + b