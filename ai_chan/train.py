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
        # ミニバッチ数
        self.batch_size = divide - 1
        # 評価用データの添字
        self.eval_data = self.batch_size
        # 重みのヒストグラム
        self.hist_w_start = None
        self.edge_w_start = None
        self.hist_b_start = None
        self.edge_b_start = None
        self.hist_w_finish = None
        self.edge_w_finish = None
        self.hist_b_finish = None
        self.edge_b_finish = None


    def train(self, loop):
        """
        学習を行います.
        学習の途中経過は、このクラスのインスタンス変数を参照してください
        :param loop: 学習回数
        :return: 最小エラー
        """
        # 初期の重みの頻度
        hist_w, bin_edges_w = util.histogram(self.nnet.w)
        hist_b, bin_edges_b = util.histogram(self.nnet.b)
        self.hist_w_start = hist_w
        self.edge_w_start = bin_edges_w
        self.hist_b_start = hist_b
        self.edge_b_start = bin_edges_b

        min_error = sys.float_info.max

        for cnt in range(0, loop):
            # 順伝搬 (評価用)
            gy = self.nnet.forward(self.x[self.eval_data])

            # 誤差評価 (評価用)
            self.gx.append(cnt)
            error = util.least_square_average(self.d[self.eval_data], gy)
            self.ge.append(error)

            current_batch = cnt % self.batch_size

            # 順伝搬 (訓練用)
            y = self.nnet.forward(self.x[current_batch])

            # 誤差評価 (訓練用)
            self.tx.append(cnt)
            error = util.least_square_average(self.d[current_batch], y)
            self.te.append(error)

            # 最小エラー値の更新
            min_error = min_error if min_error < error else error

            # 逆伝搬
            dEdW, dEdB = self.nnet.backward(self.d[current_batch], y)
            # パラメータ修正
            self.nnet.adjust_network(dEdW, dEdB)

        # 学習後の重みの頻度
        hist_w, bin_edges_w = util.histogram(self.nnet.w)
        hist_b, bin_edges_b = util.histogram(self.nnet.b)
        self.hist_w_finish = hist_w
        self.edge_w_finish = bin_edges_w
        self.hist_b_finish = hist_b
        self.edge_b_finish = bin_edges_b

        return min_error
