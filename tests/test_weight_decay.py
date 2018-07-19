import unittest
import numpy as np
import numpy.testing as npt
import matplotlib.pyplot as plt
from ai_chan import weight
from ai_chan import util


class TestWeight(unittest.TestCase):

    def setUp(self):
        w = [None]
        b = [None]

        w.append(np.array([
                     [-1.0, 2.0, 3.0],
                     [-4.0, 5.0, 6.0],
                     [-7.0, -8.0, -9.0]
                 ]))
        b.append(np.array([
                     [11.0],
                     [12.0],
                     [-13.0]
                 ]))

        w.append(np.array([
                     [-21.0, 22.0],
                     [-23.0, -24.0]
                 ]))
        b.append(np.array([
                     [31.0],
                     [-32.0]
                 ]))

        w.append(np.array([
                     [41.0]
                 ]))
        b.append(np.array([
                     [51.0]
                 ]))

        self.w = w
        self.b = b

    def test_no_decay(self):
        d = weight.NoDecay()

        dRdW, dRdB = d.r(self.w, self.b)

        self.assertIsNone(dRdW[0])
        self.assertIsNone(dRdB[0])

        npt.assert_allclose(dRdW[1], [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])
        npt.assert_allclose(dRdB[1], [
            [0],
            [0],
            [0]
        ])

        npt.assert_allclose(dRdW[2], [
            [0, 0],
            [0, 0]
        ])
        npt.assert_allclose(dRdB[2], [
            [0],
            [0]
        ])

        npt.assert_allclose(dRdW[3], [
            [0]
        ])
        npt.assert_allclose(dRdB[3], [
            [0]
        ])

    def test_L1_decay(self):
        d = weight.L1Decay()

        dRdW, dRdB = d.r(self.w, self.b)

        self.assertIsNone(dRdW[0])
        self.assertIsNone(dRdB[0])

        npt.assert_allclose(dRdW[1], [
            [-0.1, 0.1, 0.1],
            [-0.1, 0.1, 0.1],
            [-0.1, -0.1, -0.1]
        ])
        npt.assert_allclose(dRdB[1], [
            [0.1],
            [0.1],
            [-0.1]
        ])

        npt.assert_allclose(dRdW[2], [
            [-0.1, 0.1],
            [-0.1, -0.1]
        ])
        npt.assert_allclose(dRdB[2], [
            [0.1],
            [-0.1]
        ])

        npt.assert_allclose(dRdW[3], [
            [0.1]
        ])
        npt.assert_allclose(dRdB[3], [
            [0.1]
        ])

    def test_L2_decay(self):
        # npt.assert_array_equal() は、厳密に一致することを検証する
        # npt.assert_allclose() は、10e-7 の相対誤差を許容する (デフォルト atol=0, rtol=10e-7)
        # L2正則化を行う際、係数λ=0.1をかけると(2進数の)丸め誤差の分期待値と実測値がずれる
        d = weight.L2Decay()

        dRdW, dRdB = d.r(self.w, self.b)

        self.assertIsNone(dRdW[0])
        self.assertIsNone(dRdB[0])

        npt.assert_allclose(dRdW[1], [
            [-0.1, 0.2, 0.3],
            [-0.4, 0.5, 0.6],
            [-0.7, -0.8, -0.9]
        ])
        npt.assert_allclose(dRdB[1], [
            [1.1],
            [1.2],
            [-1.3]
        ])

        npt.assert_allclose(dRdW[2], [
            [-2.1, 2.2],
            [-2.3, -2.4]
        ])
        npt.assert_allclose(dRdB[2], [
            [3.1],
            [-3.2]
        ])

        npt.assert_allclose(dRdW[3], [
            [4.1]
        ])
        npt.assert_allclose(dRdB[3], [
            [5.1]
        ])

    def test_Linf_decay(self):
        d = weight.LmaxDecay()

        dRdW, dRdB = d.r(self.w, self.b)

        self.assertIsNone(dRdW[0])
        self.assertIsNone(dRdB[0])

        npt.assert_allclose(dRdW[1], [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, -0.9]
        ])
        npt.assert_allclose(dRdB[1], [
            [0.0],
            [0.0],
            [-1.3]
        ])

        npt.assert_allclose(dRdW[2], [
            [0.0, 0.0],
            [0.0, -2.4]
        ])
        npt.assert_allclose(dRdB[2], [
            [0.0],
            [-3.2]
        ])

        npt.assert_allclose(dRdW[3], [
            [4.1]
        ])
        npt.assert_allclose(dRdB[3], [
            [5.1]
        ])


if __name__ == '__main__':
    unittest.main()
