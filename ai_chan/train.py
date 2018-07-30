from ai_chan import nnet, util
import numpy as np
import sys


class NetTrainer:
    """
    学習管理クラス
    """

    def __init__(self, nnet, x, d, divide=5):
        """
        コンストラクタ
        :param nnet: ニューラルネット
        :param x: 入力データ
        :param d: 教師値
        :param divide: 教師データをいくつのミニバッチに分割するか(デフォルト5)
        """
        self.nnet = nnet
        # 訓練誤差
        self.tx = []
        self.te = []
        # 汎化誤差
        self.gx = []
        self.ge = []
        # 入力データ
        self.x = np.hsplit(x, divide)
        # 教師値データ
        self.d = np.hsplit(d, divide)
        # 訓練用ミニバッチ数
        self.train_size = divide - 1
        # 評価用データの添字
        self.eval_data = self.train_size
        # 初期状態と終了状態のパラメータ
        self.start_w = None
        self.start_b = None
        self.finish_w = None
        self.finish_b = None

    def train(self, loop):
        """
        学習を行います.
        学習の途中経過は、このクラスのインスタンス変数を参照してください
        :param loop: 学習回数
        :return: 最小エラー
        """
        # 初期の重みをとっておく
        self.start_w = np.copy(self.nnet.w)
        self.start_b = np.copy(self.nnet.b)

        min_error = sys.float_info.max

        for cnt in range(0, loop):
            # 順伝搬 (評価用)
            gy = self.nnet.forward(self.x[self.eval_data])

            # 誤差評価 (評価用)
            self.gx.append(cnt)
            error = util.least_square_average(self.d[self.eval_data], gy)
            self.ge.append(error)

            # 最小エラー値の更新
            min_error = min_error if min_error < error else error

            # 今回の学習セット番号
            current_batch = cnt % self.train_size

            # 順伝搬 (訓練用)
            y = self.nnet.forward(self.x[current_batch])

            # 誤差評価 (訓練用)
            self.tx.append(cnt)
            error = util.least_square_average(self.d[current_batch], y)
            self.te.append(error)

            # 逆伝搬
            dEdW, dEdB = self.nnet.backward(self.d[current_batch], y)
            # パラメータ修正
            self.nnet.adjust_network(dEdW, dEdB)

        # TODO: 最後の重みではなく、最も汎化誤差が小さい w と b をとっておくようにする
        # 最後の重みをとっておく
        self.finish_w = np.copy(self.nnet.w)
        self.finish_b = np.copy(self.nnet.b)

        return min_error

    def eval(self):
        """
        学習結果を返します.
        :return d_train: 訓練データの教師値のlist
        :return y_train: 訓練データの予測値のlist
        :return d_eval: 評価用データの教師値
        :return y_eval: 評価用データの予測値
        """
        d_train = []
        y_train = []
        for dataset in range(0, self.train_size):
            d_train.append(self.d[dataset])
            y_train.append(self.nnet.forward(self.x[dataset]))

        d_eval = self.d[self.eval_data]
        y_eval = self.nnet.forward(self.x[self.eval_data])

        return d_train, y_train, d_eval, y_eval

    # TODO 訓練データを返すメソッドを作る。出力層で線形回帰を行うため

