import unittest
import numpy.testing as npt
from ai_chan import nnet
from ai_chan import layer


class TestCreateNetwork(unittest.TestCase):

    def test_createNetwork(self):
        """
        3層のネットワークを作成して、重み行列、バイアスの行列の形を検証します
        """
        # L0:None(添え字を層番号と揃えるためのダミー)
        # L1:in 3→ out 4
        # L2:in 4→ out 5
        # L3:in 5→ out 1
        net = nnet.SimpleNet()
        net.add_layer(3, 4, 5)
        net.add_layer(1)

        self.assertEqual(4, len(net.w))
        self.assertEqual(4, len(net.b))
        # 一般的な層番号と添え字をそろえるため 0層目のw,bには None が入る
        self.assertEqual(None, net.w[0])
        self.assertEqual(None, net.b[0])
        # 1層目は 入力3 出力4
        # |u1|   |w11 w12 w13| |z1|
        # |u2| = |w21 w22 w23| |z2|
        # |u3|   |w31 w32 w33| |z3|
        # |u4|   |w41 w42 w43|
        self.assertEqual((4,3), net.w[1].shape)
        self.assertEqual((4,1), net.b[1].shape)
        # 2層目は 入力4 出力5
        self.assertEqual((5,4), net.w[2].shape)
        self.assertEqual((5,1), net.b[2].shape)
        # 3層目は 入力5 出力1
        self.assertEqual((1,5), net.w[3].shape)
        self.assertEqual((1,1), net.b[3].shape)

    def test_createSimpleNetwork(self):
        """
        2層のネットワークを作成して、重み行列、バイアスの要素値を検証します
        """
        # L0:None(添え字を層番号と揃えるためのダミー)
        # L1:in 2→ out 3
        # L2:in 3→ out 1
        net = nnet.SimpleNet()
        net.add_layer(2, 3, layer_factory=layer.Seq())
        net.add_layer(1, layer_factory=layer.Seq())

        self.assertEqual(3, len(net.w))
        self.assertEqual(3, len(net.b))

        # 一般的な層番号と添え字をそろえるため 0層目のw,bには None が入る
        self.assertEqual(None, net.w[0])
        self.assertEqual(None, net.b[0])
        # 1層目は 入力2 出力3
        #    W      x  +   b
        # | 0 1 | |x1| + | 0 |
        # | 2 3 | |x2|   | 1 |
        # | 4 5 |        | 2 |
        npt.assert_array_equal([
            [0, 1],
            [2, 3],
            [4, 5]
        ], net.w[1])
        npt.assert_array_equal([
            [0],
            [1],
            [2]
        ], net.b[1])
        # 2層目は 入力3 出力1
        #    W      z  +   b
        # | 0 1 2 | |z1| + | 0 |
        #           |z2|
        #           |z3|
        npt.assert_array_equal([
            [0, 1, 2]
        ], net.w[2])
        npt.assert_array_equal([
            [0]
        ], net.b[2])


if __name__ == '__main__':
    unittest.main()
