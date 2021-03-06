import unittest
import numpy as np
import numpy.testing as npt
from ai_chan import nnet
from ai_chan import layer


class TestForward(unittest.TestCase):

    def test_forward_SingleBatch(self):
        """
        1データを　forward できることを検証します.
        """
        # L0:None(添え字を層番号と揃えるためのダミー)
        # L1:in 2→ out 3
        # L2:in 3→ out 1
        net = nnet.SimpleNet()
        net.add_layer(2, 3, layer_factory=layer.Seq())
        net.add_layer(1, layer_factory=layer.Seq())

        x = np.array([
            [1],
            [-1]
        ])

        y = net.forward(x)

        # L0 : u[0] は None、z[0] は 入力X
        self.assertIsNone(net.u_memento[0])
        npt.assert_array_equal([
            [1],
            [-1]
        ], net.z_memento[0])

        # L1
        #  W1    Z0=X   B1    U1             Z1
        # |0 1| | 1| + |0| = |-1| - ReLu -> | 0|
        # |2 3| |-1|   |1|   | 0|           | 0|
        # |4 5|        |2|   | 1|           | 1|
        npt.assert_array_equal([
            [-1],
            [0],
            [1]
        ], net.u_memento[1])
        npt.assert_array_equal([
            [0],
            [0],
            [1]
        ], net.z_memento[1])

        # L2
        #  W2       Z1    B2    U2             Z2=Y
        # |0 1 2| | 0| + |0| = | 2| - 恒等 -> | 2|
        #         | 0|
        #         | 1|
        npt.assert_array_equal([
            [2]
        ], net.u_memento[2])
        npt.assert_array_equal([
            [2]
        ], net.z_memento[2])

        # Y
        npt.assert_array_equal([
            [2]
        ], y)

    def test_forward_MultipleBatch(self):
        """
        複数データを一度に　forward できることを検証します.
        """
        # L0:None(添え字を層番号と揃えるためのダミー)
        # L1:in 2→ out 3
        # L2:in 3→ out 1
        net = nnet.SimpleNet()
        net.add_layer(2, 3, layer_factory=layer.Seq())
        net.add_layer(1, layer_factory=layer.Seq())

        x = np.array([
            [ 1, -1],
            [-1,  1]
        ])

        y = net.forward(x)

        # L0 : u[0] は None、z[0] は 入力X
        self.assertIsNone(net.u_memento[0])
        npt.assert_array_equal([
            [ 1, -1],
            [-1,  1]
        ], net.z_memento[0])

        # L1
        #  W1    Z0=X       B1      U1               Z1
        # |0 1| | 1 -1| + |0 0| = |-1 1| - ReLu -> | 0 1|
        # |2 3| |-1  1|   |1 1|   | 0 2|           | 0 2|
        # |4 5|           |2 2|   | 1 3|           | 1 3|
        npt.assert_array_equal([
            [-1, 1],
            [ 0, 2],
            [ 1, 3]
        ], net.u_memento[1])
        npt.assert_array_equal([
            [0, 1],
            [0, 2],
            [1, 3]
        ], net.z_memento[1])

        # L2
        #  W2        Z1    B2        U2             Z2=Y
        # |0 1 2| | 0 1| + |0 0| = | 2 8| - 恒等 -> | 2 8|
        #         | 0 2|
        #         | 1 3|
        npt.assert_array_equal([
            [2, 8]
        ], net.u_memento[2])
        npt.assert_array_equal([
            [2, 8]
        ], net.z_memento[2])

        # Y
        npt.assert_array_equal([
            [2, 8]
        ], y)


if __name__ == '__main__':
    unittest.main()
