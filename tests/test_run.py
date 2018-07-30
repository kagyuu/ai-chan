import unittest
import numpy as np
from matplotlib import pyplot as plt
from sklearn import datasets

from ai_chan import nnet
from ai_chan import layer
from ai_chan import util
from ai_chan import grad
from ai_chan import func
from ai_chan import weight


class TestForward(unittest.TestCase):
    """
    新しく組み込んだモジュールの思考をするためのテストモジュール.
    Jupyter notebook を再起動するのめんどいので
    """
    def test(self):
        # データセットのロード
        # iris.data = [(がく片の長さ , がく片の幅 , 花びらの長さ , 花びらの幅)]
        iris = datasets.load_iris()
        # データをシャッフルする (ミソ!)
        np.random.shuffle(iris.data)

        x_vals = np.array([x[0:3] for x in iris.data])
        d_vals = np.array([x[3] for x in iris.data])

        data_size = len(x_vals)
        train_size = int(data_size * 0.8)
        test_size = data_size - train_size

        x_train = x_vals[0:train_size].T
        d_train = d_vals[0:train_size].T

        x_eval = x_vals[train_size:data_size].T
        d_eval = d_vals[train_size:data_size].T

        print("total:{} = train:{} + test:{}".format(data_size, len(x_train[0]), len(x_eval[0])))

        net = nnet.SimpleNet()

        # 入力データを(初期状態で)標準化スコアに変換する前処理層
        # 3→6
        net.add_pre_layer(layer.Normalize(), x=x_vals.T)

        # 中間層は 6→60
        net.add_layer(60)

        # 出力層は 60→1
        # net.add_layer(1)
        net.add_post_layer(x_train, d_train)

        # 学習係数は 0.001固定 (呼び出し回数で減衰させるとうまく収束しない)
        # net.set_learning_rate(grad.Shrink())

        # 正則化（重み減衰)
        # net.set_weight_decay(weight.L1Decay())
        net.set_weight_decay(weight.L2Decay())
        # net.set_weight_decay(weight.LmaxDecay())

        # 訓練誤差
        tx = []
        te = []
        # 汎化誤差
        gx = []
        ge = []

        for cnt in range(0, 10000):
            # 順伝搬 (評価用)
            gy = net.forward(x_eval)

            # 誤差評価 (評価用)
            gx.append(cnt)
            error_eval = util.least_square_average(d_eval, gy)
            ge.append(error_eval)

            # 順伝搬 (訓練用)
            y = net.forward(x_train)

            # 誤差評価 (訓練用)
            tx.append(cnt)
            error_train = util.least_square_average(d_train, y)
            te.append(error_train)

            # 逆伝搬
            dEdW, dEdB = net.backward(d_train, y)
            # パラメータ修正
            net.adjust_network(dEdW, dEdB)

            if 0 == cnt % 1000:
                print("訓練誤差 {}\t 汎化誤差{}".format(error_train, error_eval))

