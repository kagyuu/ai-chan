import unittest
import numpy as np
import cupy as cp
import time
from matplotlib import pyplot as plt
from sklearn import datasets
from ai_chan import nnet, layer, util, weight, func, gpu, grad


class TestForward(unittest.TestCase):
    """
    新しく組み込んだモジュールの思考をするためのテストモジュール.
    Jupyter notebook を再起動するのめんどいので
    """
    def test_gpu(self):
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

        print("GPU total:{} = train:{} + test:{}".format(data_size, len(x_train[0]), len(x_eval[0])))

        net = gpu.GPUNet()

        # 入力データを(初期状態で)標準化スコアに変換する前処理層
        # 3→6
        net.add_pre_layer(layer.Normalize(), activate_function=gpu.ReLu(), x=x_vals.T)

        # 中間層は 6→60
        net.add_layer(60, activate_function=gpu.ReLu())

        # 出力層は 60→1
        net.add_layer(1, layer_factory=layer.Random(), activate_function=gpu.IdentityMapping())

        # 学習係数は 0.001固定 (呼び出し回数で減衰させるとうまく収束しない)
        net.set_learning_rate(gpu.Static())

        # 正則化（重み減衰)
        # net.set_weight_decay(weight.L1Decay())
        net.set_weight_decay(gpu.L2Decay())
        # net.set_weight_decay(weight.LmaxDecay())
        # net.set_weight_decay(gpu.NoDecay())

        # 精度を変えても収束の影響なし
        cp.cuda.set_allocator(cp.cuda.MemoryPool().malloc)
        gpu_x_eval = cp.asarray(x_eval, dtype=gpu.FLOAT_PRECISION)
        gpu_d_eval = cp.asarray(d_eval, dtype=gpu.FLOAT_PRECISION)
        gpu_x_train = cp.asarray(x_train, dtype=gpu.FLOAT_PRECISION)
        gpu_d_train = cp.asarray(d_train, dtype=gpu.FLOAT_PRECISION)

        net.to_gpu()

        start = time.time()
        for cnt in range(0, 10000):
            # 順伝搬 (評価用)
            gpu_gy = net.forward(gpu_x_eval)

            # 誤差評価 (評価用)
            #error_train = util.least_square_average(d_eval, gpu_gy.get())

            # 順伝搬 (訓練用)
            gpu_y = net.forward(gpu_x_train)

            # 誤差評価 (訓練用)
            #error_eval = util.least_square_average(d_train, gpu_y.get())

            # 逆伝搬
            gpu_dEdW, gpu_dEdB = net.backward(gpu_d_train, gpu_y)
            # パラメータ修正
            net.adjust_network(gpu_dEdW, gpu_dEdB)

            # print("訓練誤差 {}\t 汎化誤差{}".format(error_train, error_eval))

        end = time.time()
        print("GPU 10000 epoch = {}".format(end-start))

        net.to_cpu()


    def test_cpu(self):
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

        print("CPU total:{} = train:{} + test:{}".format(data_size, len(x_train[0]), len(x_eval[0])))

        net = nnet.SimpleNet()

        # 入力データを(初期状態で)標準化スコアに変換する前処理層
        # 3→6
        net.add_pre_layer(layer.Normalize(), x=x_vals.T)

        # 中間層は 6→60
        net.add_layer(60)

        # 出力層は 60→1
        net.add_layer(1, layer_factory=layer.Random(), activate_function=func.IdentityMapping())

        # 学習係数は 0.001固定 (呼び出し回数で減衰させるとうまく収束しない)
        net.set_learning_rate(grad.Static())

        # 正則化（重み減衰)
        # net.set_weight_decay(weight.L1Decay())
        net.set_weight_decay(weight.L2Decay())
        # net.set_weight_decay(weight.LmaxDecay())

        start = time.time()
        for cnt in range(0, 10000):
            # 順伝搬 (評価用)
            gy = net.forward(x_eval)

            # 誤差評価 (評価用)
            #error_train = util.least_square_average(d_eval, gy)

            # 順伝搬 (訓練用)
            y = net.forward(x_train)

            # 誤差評価 (訓練用)
            #error_eval = util.least_square_average(d_train, y)

            # 逆伝搬
            dEdW, dEdB = net.backward(d_train, y)
            # パラメータ修正
            net.adjust_network(dEdW, dEdB)

            # print("訓練誤差 {}\t 汎化誤差{}".format(error_train, error_eval))

        end = time.time()
        print("CPU 10000 epoch = {}".format(end-start))

if __name__ == '__main__':
    unittest.main()