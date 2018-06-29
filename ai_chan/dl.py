import numpy as np


def create_random_layer(in_size, out_size):
    """
     1レイヤ分をランダムに初期化します.
    :param in_size: 入力サイズ
    :param out_size: 出力サイズ
    :return w: 重み行列
    :return b: バイアス
    """

    w = np.random.rand(out_size, in_size)
    b = np.random.rand(1, out_size).T
    return w, b


def create_seq_layer(in_size, out_size):
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


def create_network(*units, create_layer=create_random_layer):
    """
     nレイヤ分のネットワークを作成します.
    :param *units: 中間層のサイズを可変引数で指定します
    :param create_layer: 1層分の w,b を作る関数を指定します。デフォルト値は、ランダム初期化
    :return w: 重み行列
    :return b: バイアス
    """

    w_lst = []
    b_lst = []
    # ネットワークの一般的な層番号(1はじまり)に合わせるため 0 層には None を入れる
    w_lst.append(None)
    b_lst.append(None)

    for layer in range(0, len(units) - 1):
        in_size = units[layer]
        out_size = units[layer + 1]

        w, b = create_layer(in_size, out_size)

        w_lst.append(w)
        b_lst.append(b)
    return w_lst, b_lst


def forward(x, w, b, func1, func2):
    """
     順伝搬します.
     z(0) = x
     for l = 1 to L {
     u(l) = W(l) ・ z(l-1) + b(l)
     z(l) = func1( u(l) )
     }
     y = func2 ( z(L) )
    :param x: 入力データ　(複数のデータを同時に投入できる)
    :param w: 重み行列
    :param b: バイアス
    :param func1: 中間層の活性化関数
    :param func2: 出力層の活性化関数
    :return u: 中間層の計算結果
    :return z: 中間層の出力 z[l]=func1(u[l]) (ただし z[0]=x z{L}=y)
    :return y: 出力
    """

    # uとzの記録用リスト (逆伝搬で使う)
    u_memento = []
    z_memento = []

    # u[0] はNone
    # z[0] は入力データ
    z = x
    u_memento.append(None)

    for layer in range(1, len(w)):
        z_memento.append(z)
        # w・z はn行m列の行列、bはn行1列のベクトル
        # numpy の 行列計算の broadcast 規則 により
        # b が 列方向に m 個コピーされた n行m列の行列として計算される
        u = np.dot(w[layer], z) + b[layer]
        u_memento.append(u)
        # 中間層の活性化関数
        z = func1(u)

    # 出力層の活性化関数
    y = func2(u)
    z_memento.append(y)

    return u_memento, z_memento, y


def backward(w, b, u, z, delta, func):
    """
     逆伝搬をします
    :param w: 重み行列
    :param b: バイアス
    :param u: 中間層の計算結果
    :param z: 中間層の出力 z[l]=func1(u[l]) (ただし z[0]=x z{L}=y)
    :param delta: ネットワークの出力の誤差 δ(L)
    :param func: 中間層の活性化関数の導関数
    :return dEdW: Wij で E を偏微分した結果
    :return dEdB: bj で E を偏微分した結果
    """

    # dE/dW , dE/db の格納領域。第L層 から 第1層にむけて 逆順に登録するので
    # 最後に 第0層の None を追加して reverse して返却する
    dEdW = []
    dEdB = []

    # delta の列数が、バッチサイズ
    batch_size = delta.shape[1]

    for layer in range(len(w) - 1, 0, -1):
        # dEdW = δ[l] (z[l-1].T)
        # dEdB = δ[l] の各行平均
        dEdW.append(np.dot(delta, z[layer - 1].T) / batch_size)
        dEdB.append(np.array([np.mean(delta, axis=1)]).T)
        # ※ delta は参照(pointer)なので、通常 delta を変えつつ list に
        # ※ 追記する場合には list.append(copy.deepcopy(delta)) する必要あり
        # ※ 今回は下記の逆伝搬処理で delta が別のオブジェクトになるため
        # ※ deepcopy は不要

        # 誤差逆伝搬 δ[l-1] = δ[l] W[l] f'(u[l-1])
        # layer=1 のとき、次は 入力層(layer=0) なので、もう逆伝搬する
        # 必要ない (実装上は u[0] が None なのでエラーになる)
        if layer > 1:
            delta = func(u[layer - 1]) * np.dot(w[layer].T, delta)

    # 第0層(入力層)のダミー微分値
    dEdW.append(None)
    dEdB.append(None)

    # 第L層 から 第0層にむけて 逆順に登録したので、反転して返却
    dEdW.reverse()
    dEdB.reverse()

    return dEdW, dEdB


def adjust_network(w, b, dEdW, dEdB, learning_rate=0.05):
    """
     ネットワークのパラメタ―調整を行います.
    :param w: ネットワークの重み行列
    :param b: ネットワークのバイアス
    :param dEdW: 誤差のWijに対する偏微分値
    :param dEdB: 誤差のBjに対する偏微分値
    :param learning_late: 学習率(デフォルト値0.05)
    :return w: ネットワークの重み行列
    :return b: ネットワークのバイアス

    """

    for idx in range(1, len(w)):
        # 微分値が正 → Wijを大きくしたら誤差Eが大きくなるんでWijを少し小さくする
        # 微分値が負 → Wjiを大きくしたら誤差Eが小さくなるんでWijを少し大きくする
        # 少し = 学習率 ここでは 微分値の 0.05 倍
        w[idx] = w[idx] - learning_rate * dEdW[idx]
        b[idx] = b[idx] - learning_rate * dEdB[idx]

    return w, b
