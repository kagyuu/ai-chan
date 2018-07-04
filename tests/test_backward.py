import unittest
import numpy as np
import numpy.testing as npt
from ai_chan import layer_initializer
from ai_chan import math


class TestBackward(unittest.TestCase):

    def test_backward_SingleBatch(self):
        """
        1データを　backward できることを検証します.
        """
        # L1
        #  W1    Z0=X   B1    U1             Z1
        # |0 1| | 1| + |0| = |-1| - ReLu -> | 0|
        # |2 3| |-1|   |1|   | 0|           | 0|
        # |4 5|        |2|   | 1|           | 1|

        # L2
        #  W2       Z1    B2    U2             Z2=Y
        # |0 1 2| | 0| + |0| = | 2| - 恒等 -> | 2|
        #         | 0|                         D
        #         | 1|                        | 0|
        w, b = layer_initializer.create_network(2, 3, 1, layer_factory=layer_initializer.create_layer_seq)
        x = np.array([
            [1],
            [-1]
        ])
        d = np.array([[0]])
        u, z, y = layer_initializer.forward(x, w, b, math.relu, math.identity_mapping)

        dEdW, dEdB = layer_initializer.backward(w, b, u, z, y - d, math.d_relu)

        # δ2 = Y - D = |2|
        #
        # dE/dW2 = δ2・Z1.T = |0 0 2|
        # dE/dB2 = δ2 = |2|
        #
        # δ1 = f'(u1)⦿(W2.T・δ2) = |0| |0| = |0|
        #                          |0|⦿|2|   |0|
        # ⦿ Hadamard product       |1| |4|   |4|
        #
        # dE/dW1 = δ1・Z0.T = |0| |1 -1| = |0  0|
        #                     |0|          |0  0|
        #                     |4|          |4 -4|
        # dE/dB1 = δ1 = |0|
        #               |0|
        #               |4|

        self.assertEqual(3, len(dEdW))
        self.assertEqual(3, len(dEdB))

        npt.assert_array_equal([[0, 0, 2]], dEdW[2])
        npt.assert_array_equal([[2.]], dEdB[2])

        npt.assert_array_equal([
            [0,  0],
            [0,  0],
            [4, -4]
        ], dEdW[1])

        npt.assert_array_equal([
            [0.],
            [0.],
            [4.]
        ], dEdB[1])

    def test_backward_SingleCycle(self):
        """
        1データを繰り返しパラメータ調整を行うことで、誤差が小さくなることを検証します
        """
        w, b = layer_initializer.create_network(2, 3, 1, layer_factory=layer_initializer.create_layer_seq)
        x = np.array([
            [1],
            [-1]
        ])
        d = np.array([[0]])

        last_error = np.finfo(float).max
        for cnt in range(0, 10):
            # 順伝搬
            u, z, y = layer_initializer.forward(x, w, b, math.relu, math.identity_mapping)

            # 誤差評価 (前回より誤差が小さくなることを確認する)
            error =  math.least_square(d, y)
            # print("LOOP={} ERROR={}".format(cnt,error))
            self.assertLess(error, last_error)
            last_error = error

            # 逆伝搬
            dEdW, dEdB = layer_initializer.backward(w, b, u, z, y - d, math.d_relu)
            # パラメータ修正
            w, b = layer_initializer.adjust_network(w, b, dEdW, dEdB)

        # print(w)
        # print(b)

    def test_backward_MultiBatch(self):
        """
        複数データを　backward できることを検証します.
        """
        # L1
        #  W1    Z0=X       B1      U1               Z1
        # |0 1| | 1 -1| + |0 0| = |-1 1| - ReLu -> | 0 1|
        # |2 3| |-1  1|   |1 1|   | 0 2|           | 0 2|
        # |4 5|           |2 2|   | 1 3|           | 1 3|

        # L2
        #  W2        Z1    B2        U2             Z2=Y
        # |0 1 2| | 0 1| + |0 0| = | 2 8| - 恒等 -> | 2  8|
        #         | 0 2|                             D
        #         | 1 3|                            | 0 10|

        w, b = layer_initializer.create_network(2, 3, 1, layer_factory=layer_initializer.create_layer_seq)
        x = np.array([
            [ 1,-1],
            [-1,1]
        ])
        d = np.array([[0, 10]])
        u, z, y = layer_initializer.forward(x, w, b, math.relu, math.identity_mapping)

        dEdW, dEdB = layer_initializer.backward(w, b, u, z, y - d, math.d_relu)

        # δ2 = Y - D = |2 -2|
        #
        # dE/dW2 = δ2・Z1.T ÷ 2 = |-2 -4 -4| ÷ 2
        # dE/dB2 = δ2 = |avr(2 -2)| = |0|
        #
        # δ1 = f'(u1)⦿(W2.T・δ2) = |0 1| |0  0| = | 0  0|
        #                          |0 1|⦿|2 -2|   | 0 -2|
        # ⦿ Hadamard product       |1 1| |4 -4|   | 4 -4|
        #
        # dE/dW1 = δ1・Z0.T ÷ 2 = | 0  0| | 1 -1| ÷2 = |0  0| ÷ 2
        #                         | 0 -2| |-1  1|      |2 -2|
        #                         | 4 -4|              |8 -8|
        # dE/dB1 = δ1 = |avr(0  0)| = | 0|
        #               |avr(0 -2)|   |-1|
        #               |avr(4 -4)|   | 0|

        self.assertEqual(3, len(dEdW))
        self.assertEqual(3, len(dEdB))

        npt.assert_array_equal([[-1., -2., -2.]], dEdW[2])
        npt.assert_array_equal([[0.]], dEdB[2])

        npt.assert_array_equal([
            [0.,  0],
            [1., -1.],
            [4., -4.]
        ], dEdW[1])

        npt.assert_array_equal([
            [ 0.],
            [-1.],
            [ 0.]
        ], dEdB[1])

    def test_backward_MultiCycle(self):
        """
        複数データを繰り返しパラメータ調整を行うことで、誤差が小さくなることを検証します
        """
        w, b = layer_initializer.create_network(2, 3, 1, layer_factory=layer_initializer.create_layer_seq)
        x = np.array([
            [ 1,-1],
            [-1,1]
        ])
        d = np.array([[0, 10]])

        last_error = np.finfo(float).max
        for cnt in range(0, 10):
            # 順伝搬
            u, z, y = layer_initializer.forward(x, w, b, math.relu, math.identity_mapping)

            # 誤差評価 (前回より誤差が小さくなることを確認する)
            error =  math.least_square(d, y)
            # print("LOOP={} ERROR={}".format(cnt,error))
            self.assertLess(error, last_error)
            last_error = error

            # 逆伝搬
            dEdW, dEdB = layer_initializer.backward(w, b, u, z, y - d, math.d_relu)
            # パラメータ修正
            w, b = layer_initializer.adjust_network(w, b, dEdW, dEdB)

        # print(w)
        # print(b)
if __name__ == '__main__':
    unittest.main()
