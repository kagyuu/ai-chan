from abc import ABCMeta, abstractmethod
import numpy as np
from ai_chan import func
from ai_chan import grad
from ai_chan import layer
from ai_chan import weight


class AbstractNet(metaclass=ABCMeta):
    """
    ニューラルネットワークの抽象クラス.
    """

    def __init__(self):
        """
        コンストラクタ.
        インスタンス変数
        w 重み行列 (配列添え字と、一般的な教科書と層番号を合わせるため 第0層 にNoneを設定)
        b バイアス (配列添え字と、一般的な教科書と層番号を合わせるため 第0層 にNoneを設定)
        learning_flag W,b を更新するかのフラグ 1.0=学習する 0.0=学習しない
        f 活性化関数 (配列添え字と、一般的な教科書と層番号を合わせるため 第0層 にNoneを設定)
        learning_rate 学習率 (初期値は、学習率0.001固定)
        u_memento 順伝搬時のuの記録用リスト (逆伝搬で使う)
        z_memento 順伝搬時のzの記録用リスト (逆伝搬で使う)
        d 重み減衰アルゴリズム (Weight Decay)
        """
        self.w = [None]
        self.b = [None]
        self.learning_flag = [None]
        self.f: [func.ActivateFunction] = [None]
        self.g = grad.Static()
        self.u_memento = []
        self.z_memento = []
        self.d = weight.NoDecay()

    @abstractmethod
    def add_pre_layer(self, layer_factory, activate_function, x, fix_parameter):
        """
        ニューラルネットワークに (初期状態で) 統計的な前処理を行うレイヤを追加します.
        活性化関数はReLu関数になります.
        :param layer_factory: レイヤを初期化する関数
        :param activate_function: 活性化関数
        :param x: 入力データ
        :param y: 教師値
        :param fix_parameter: この層の W,b を固定するかどうかのフラグ
        :return: x をこの層で変換したときの出力
        """
        pass

    @abstractmethod
    def add_layer(self, *units, layer_factory, activate_function, fix_parameter):
        """
        ニューラルネットワークに 中間層を追加します
        :param *units: 中間層のサイズを可変引数で指定します
        :param layer_factory: レイヤを初期化する関数
        :param activate_function: 活性化関数
        :param fix_parameter: この層の W,b を固定するかどうかのフラグ
        """
        pass

    @abstractmethod
    def add_post_layer(self, x, y, layer_factory, activate_function, fix_parameter):
        """
        ニューラルネットワークに 出力層を追加します
        :param x: 入力データ
        :param y: 出力データ
        :param layer_factory: レイヤを初期化する関数 (省略時:Random)
        :param activate_function: 活性化関数 (省略時:恒等写像)
        :param fix_parameter: この層の W,b を固定するかどうかのフラグ
        """
        pass

    @abstractmethod
    def set_learning_rate(self, g):
        """
        学習係数の管理を行うクラスを登録します
        :param g: 学習計数管理クラス
        """
        pass

    @abstractmethod
    def set_weight_decay(self, d):
        """
        重み減衰の管理を行うクラスを登録します
        :param d: 重み減衰管理クラス
        """
        pass

    @abstractmethod
    def forward(self, x, xp=np):
        """
        順伝搬
        :param x: 入力データ
        :param xp: numpy or cupy
        :return: 予測値
        """
        pass

    @abstractmethod
    def backward(self, d, y, xp=np):
        """
        逆伝搬
        :param d: 教師値
        :param y: 予測値
        :param xp: numpy or cupy
        :return:
        """
        pass

    @abstractmethod
    def adjust_network(self, dEdW, dEdB, xp=np):
        """
        :param xp: numpy or cupy
        :return:
        """
        pass


class SimpleNet(AbstractNet):

    def add_pre_layer(self, layer_factory, activate_function=func.ReLu(), x=None, fix_parameter=False):
        w, b = layer_factory.create(x)
        self.w.append(w)
        self.b.append(b)
        self.f.append(activate_function)
        self.learning_flag.append(0.0 if fix_parameter else 1.0)

        u = np.dot(w, x) + b
        z = activate_function.calc(u)
        return z

    def add_layer(self, *units, layer_factory=layer.He(), activate_function=func.ReLu(), fix_parameter=False):
        # すでに前の層があるのであれば、その層の出力を入力として units[0] 個の出力を行う層を作る
        if len(self.w) > 1:
            in_size = self.b[-1].shape[0]
            out_size = units[0]
            self.__append_layer(in_size, out_size, layer_factory, activate_function, fix_parameter)

        for layer in range(0, len(units) - 1):
            in_size = units[layer]
            out_size = units[layer + 1]
            self.__append_layer(in_size, out_size, layer_factory, activate_function, fix_parameter)

    def add_post_layer(self, x, y, layer_factory=layer.LeastSquare(), activate_function=func.IdentityMapping()
                      , fix_parameter=False):
        w, b = layer_factory.create(self.forward(x), activate_function.inv(y))
        self.w.append(w)
        self.b.append(b)
        self.f.append(activate_function)
        self.learning_flag.append(0.0 if fix_parameter else 1.0)

    def __append_layer(self, in_size, out_size, layer_factory, activate_function, fix_parameter):
        w, b = layer_factory.create(in_size, out_size)
        self.w.append(w)
        self.b.append(b)
        self.f.append(activate_function)
        self.learning_flag.append(0.0 if fix_parameter else 1.0)

    def set_learning_rate(self, g):
        self.g = g

    def set_weight_decay(self, d):
        self.d = d

    def forward(self, x, xp=np):
        """
         順伝搬します.
         z(0) = x
         for l = 1 to L {
           u(l) = W(l) ・ z(l-1) + b(l)
           z(l) = func1( u(l) )
         }
         y = func2 ( z(L) )
        :param x: 入力データ　(複数のデータを同時に投入できる)
        :param xp: numpy or cupy
        :return: 出力
        """

        # uとzの記録用リストの初期化
        self.u_memento = []
        self.z_memento = []

        # u[0] はNone
        # z[0] は入力データ
        z = x
        self.u_memento.append(None)

        for layer in range(1, len(self.w)):
            self.z_memento.append(z)
            # w・z はn行m列の行列、bはn行1列のベクトル
            # numpy の 行列計算の broadcast 規則 により
            # b が 列方向に m 個コピーされた n行m列の行列として計算される
            u = xp.dot(self.w[layer], z) + self.b[layer]
            self.u_memento.append(u)
            # 活性化関数
            z = self.f[layer].calc(u)

        # 最終的なzは出力y
        y = z
        self.z_memento.append(y)

        return y

    def backward(self, d, y, xp=np):

        delta = self.f[-1].delta(d, y)

        # dE/dW , dE/db の格納領域。第L層 から 第1層にむけて 逆順に登録するので
        # 最後に 第0層の None を追加して reverse して返却する
        dEdW = []
        dEdB = []

        # delta の列数が、バッチサイズ
        batch_size = delta.shape[1]

        for l in range(len(self.w) - 1, 0, -1):
            # dEdW = δ[l] (z[l-1].T) の各要素をバッチサイズで割ったもの
            # dEdB = δ[l] の各行平均
            dEdW.append(xp.dot(delta, self.z_memento[l - 1].T) / batch_size)
            dEdB.append(xp.array([xp.mean(delta, axis=1)]).T)

            # 誤差逆伝搬 δ[l-1] = δ[l] W[l] f'(u[l-1])
            # layer=1 のとき、次は 入力層(layer=0) なので、もう逆伝搬する
            # 必要ない (実装上は u[0] が None なのでエラーになる)
            if l > 1:
                delta = self.f[l - 1].differential(self.u_memento[l - 1]) * xp.dot(self.w[l].T, delta)

        # 第0層(入力層)のダミー微分値
        dEdW.append(None)
        dEdB.append(None)

        # 第L層 から 第0層にむけて 逆順に登録したので、反転して返却
        dEdW.reverse()
        dEdB.reverse()

        return dEdW, dEdB

    def adjust_network(self, dEdW, dEdB, xp=np):
        # ネットワークの重みの調整
        hw, hb = self.g.eta(dEdW, dEdB)
        dRdW, dRdB = self.d.r(self.w, self.b)

        for idx in range(1, len(self.w)):
            # 微分値が正 → Wijを大きくしたら誤差Eが大きくなるんでWijを少し小さくする
            # 微分値が負 → Wjiを大きくしたら誤差Eが小さくなるんでWijを少し大きくする
            self.w[idx] = self.w[idx] - self.learning_flag[idx] * hw[idx] * (dEdW[idx] + dRdW[idx])
            self.b[idx] = self.b[idx] - self.learning_flag[idx] * hb[idx] * (dEdB[idx] + dRdB[idx])
            # 極大に発散するのを防ぐため重みの上限は 10e2
            self.w[idx] = xp.maximum(-10e2, xp.minimum(10e2, self.w[idx]))
            self.b[idx] = xp.maximum(-10e2, xp.minimum(10e2, self.b[idx]))
