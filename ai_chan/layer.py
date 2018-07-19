import numpy as np
from abc import ABCMeta, abstractmethod


class LayerFactory(metaclass=ABCMeta):
    pass


class PreLayerFactory(LayerFactory):
    """
    入力層のレイヤを作ります.
    教師データの統計分析を行うレイヤを作ります.
    """
    @abstractmethod
    def create(self, x, y):
        pass


class MidLayerFactory(LayerFactory):
    """
    中間層及び出力層のレイヤを作ります
    """
    @abstractmethod
    def create(self, in_size, out_size):
        pass


class Seq(MidLayerFactory):

    def create(self, in_size, out_size):
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


class Random(MidLayerFactory):

    def create(self, in_size, out_size):
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


class Xavier(MidLayerFactory):

    def create(self, in_size, out_size):
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


class He(MidLayerFactory):

    def create(self, in_size, out_size):
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


class Normalize(PreLayerFactory):

    def create(self, x, y):
        """
        入力データを標準化スコアに変換するレイヤを作ります.
        標準化スコア(n,i) = (x(n,i) - avr_x(n) / σ(n) = x(n,i)/σ(n) - avr_x(n)/σ(n)
        なので、
        u = wx+b
        w = 1/σ
        b = -avr_x/σ
        としたとき、このニューラルネットは、標準化スコアを出力させることができます。

        このクラスで作られるレイヤは、活性化関数に ReLu を使った場合にすべての情報を
        次のレイヤに渡せるように、標準化スコアと標準化スコアの -1 倍を出力します.
        (ReLuは0以下を0に丸める活性化関数なので、標準化スコアと標準化スコアの-1倍
        を出力すれば、そのどちらかが次の層に渡ることになる)

        ※ 計算上 分散σ(n) が、非常に小さくなるのを防ぐために σ(n) には 10e-7 を加えています。
        :param x: 入力データ
        :param y: 期待出力
        :return w: 重み行列
        :return b: バイアス
        """
        sigma = np.sqrt(np.var(x, axis=1) ** 2 + 10e-7)
        average = np.mean(x, axis=1)

        # 対角成分が sigma[i] な対角行列を作ります(対角成分以外0)
        w = np.diag(1.0 / sigma)
        w = np.vstack((w, -1.0 * w))
        # b[i] = -average[i]/sigma[i] なバイアスベクトルb(N行1列)を作ります
        b = np.array([- average / sigma]).T
        b = np.vstack((b, -1.0 * b))

        return w, b