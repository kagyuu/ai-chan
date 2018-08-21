import functools
import types
import inspect
import cupy as cp
from ai_chan import func, grad, nnet, weight


def cupy_decorator(func):
    """
    数値演算関数の cupy/numpy 対応デコレーション.
    第二引数が　numpy.ndarray であれば、可変引数 xp の numpy を指定します。
    cupy.ndarray であれば、可変引数 xp に cupy を指定します。
    ※ 第一引数は self です
    :param func: function
    :return: cupy対応
    """

    @functools.wraps(func)
    def __func(*args, **kwargs):
        kwargs["xp"] = cp.get_array_module(args[1])
        return func(*args, **kwargs)

    return __func


def cupy_decorator2(func):
    """
    数値演算関数の cupy/numpy 対応デコレーション.
    第二引数が　[numpy.ndarray] であれば、可変引数 xp の numpy を指定します。
    [cupy.ndarray] であれば、可変引数 xp に cupy を指定します。
    ※ 第一引数は self です
    :param func: function
    :return: cupy対応
    """

    @functools.wraps(func)
    def __func(*args, **kwargs):
        kwargs["ap"] = cp.get_array_module(args[1][1])
        return func(*args, **kwargs)

    return __func


def GPU(clazz):
    """
    引数に xp または ap を持つ function を cupy_decorator で デコレートする
    xp の場合は第一引数に cupy.ndarray または numpy.ndarray が来ると期待する.
    ap の場合には第一引数に [cupy.ndarray] または [numpy.ndarray] が来ると期待する
    :param clazz: クラス
    :return: 処理済みクラス
    """
    for property_name in dir(clazz):
        attr = getattr(clazz, property_name)
        if type(attr) is not types.FunctionType:
            continue

        args = inspect.signature(attr).parameters.keys()
        if 'xp' in args:
            setattr(clazz, property_name, cupy_decorator(attr))
        elif 'ap' in args:
            setattr(clazz, property_name, cupy_decorator2(attr))

    return clazz


@GPU
class IdentityMapping(func.IdentityMapping):
    pass


@GPU
class Sigmoid(func.Sigmoid):
    pass


@GPU
class Tanh(func.Tanh):
    pass


@GPU
class ReLu(func.ReLu):
    pass


@GPU
class Static(grad.Static):
    pass


@GPU
class Shrink(grad.Shrink):
    pass


@GPU
class NoDecay(weight.NoDecay):
    pass


@GPU
class L1Decay(weight.L1Decay):
    pass


@GPU
class L2Decay(weight.L2Decay):
    pass


@GPU
class LmaxDecay(weight.LmaxDecay):
    pass


@GPU
class GPUNet(nnet.SimpleNet):

    def to_gpu(self):
        self.__to_gpu(self.w)
        self.__to_gpu(self.b)
        self.__to_gpu(self.learning_flag)

    def to_cpu(self):
        self.__to_cpu(self.w)
        self.__to_cpu(self.b)
        self.__to_cpu(self.learning_flag)

    @staticmethod
    def __to_gpu(list):
        for cnt in range(0, len(list)):
            list[cnt] = cp.asarray(list[cnt], dtype=cp.float16)

    @staticmethod
    def __to_cpu(list):
        for cnt in range(0, len(list)):
            list[cnt] = cp.asnumpy(list[cnt])
