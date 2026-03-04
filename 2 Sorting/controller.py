"""
Florin Leon, Modular Neural Computer, 2026
https://github.com/florinleon/ModularNeuralComputer
"""

import numpy as np
from mlp import MLP
from config import *


class Controller:
    """Controller MLP.

    Input:
        x = (i, p, z)
        i = current pair index
        p = current pass limit
        z = constant zero read from memory

    Output:
        y = (g_process, g_next_pass, g_stop, r1, r2, r3, w1, w2, w3)

    Exact map on legal execution states:
        if i < p, then
            y = (1, 0, 0, i, i + 1, AddrCur, i, i + 1, AddrCur)

        if i = p and p > 1, then
            y = (0, 1, 0, AddrPass, AddrZero, AddrZero, AddrPass, AddrCur, AddrZero)

        if i = p and p = 1, then
            y = (0, 0, 1, AddrZero, AddrZero, AddrZero, AddrRunning, AddrZero, AddrZero)

    Representative instances:
        x = (1, 5, 0) -> y = (1, 0, 0, 1,   2,   200, 1,   2,   200)
        x = (4, 5, 0) -> y = (1, 0, 0, 4,   5,   200, 4,   5,   200)
        x = (5, 5, 0) -> y = (0, 1, 0, 201, 202, 202, 201, 200, 202)
        x = (1, 1, 0) -> y = (0, 0, 1, 202, 202, 202, 203, 202, 202)

    The controller is a pure MLP. It emits both the exact one-hot phase gates
    and the three read addresses plus three write addresses needed by the active
    sorting module. No symbolic branch logic is used after the forward pass.
    """

    def __init__(self):
        self.net = MLP(layer_sizes=[3, 7, 5, 9], hidden_activation="relu", output_activation="linear")

        w1 = np.zeros((3, 7), dtype=np.float32)
        b1 = np.zeros(7, dtype=np.float32)

        # h0 = ReLU(p - i)
        w1[0, 0] = -1.0
        w1[1, 0] = 1.0

        # h1 = ReLU(p - i - 1)
        w1[0, 1] = -1.0
        w1[1, 1] = 1.0
        b1[1] = -1.0

        # h2 = ReLU(p - 1)
        w1[1, 2] = 1.0
        b1[2] = -1.0

        # h3 = ReLU(p - 2)
        w1[1, 3] = 1.0
        b1[3] = -2.0

        # h4 = ReLU(i)
        w1[0, 4] = 1.0

        # h5 = ReLU(i + 1)
        w1[0, 5] = 1.0
        b1[5] = 1.0

        # h6 = ReLU(1)
        b1[6] = 1.0

        self.net.weights[0] = w1
        self.net.biases[0] = b1

        w2 = np.zeros((7, 5), dtype=np.float32)
        b2 = np.zeros(5, dtype=np.float32)

        # u0 = g_process = ReLU(h0 - h1)
        w2[0, 0] = 1.0
        w2[1, 0] = -1.0

        # u1 = g_next_pass = ReLU(-h0 + h1 + h2 - h3)
        w2[0, 1] = -1.0
        w2[1, 1] = 1.0
        w2[2, 1] = 1.0
        w2[3, 1] = -1.0

        # u2 = g_stop = ReLU(1 - h2 + h3)
        w2[2, 2] = -1.0
        w2[3, 2] = 1.0
        w2[6, 2] = 1.0

        # u3 = g_process * i
        # On legal states, i is in [1, MaxN]. When g_process = 1, this unit
        # returns i exactly. When g_process = 0, the preactivation is at most 0.
        w2[0, 3] = MaxN
        w2[1, 3] = -MaxN
        w2[4, 3] = 1.0
        b2[3] = -MaxN

        # u4 = g_process * (i + 1)
        w2[0, 4] = MaxN + 1
        w2[1, 4] = -(MaxN + 1)
        w2[5, 4] = 1.0
        b2[4] = -(MaxN + 1)

        self.net.weights[1] = w2
        self.net.biases[1] = b2

        w3 = np.zeros((5, 9), dtype=np.float32)
        b3 = np.zeros(9, dtype=np.float32)

        # Gates.
        w3[0, 0] = 1.0
        w3[1, 1] = 1.0
        w3[2, 2] = 1.0

        # Read addresses.
        # r1 = i during pair processing, AddrPass during next-pass, AddrZero on stop.
        w3[3, 3] = 1.0
        w3[1, 3] = AddrPass
        w3[2, 3] = AddrZero

        # r2 = i + 1 during pair processing, AddrZero otherwise.
        w3[4, 4] = 1.0
        w3[1, 4] = AddrZero
        w3[2, 4] = AddrZero

        # r3 = AddrCur during pair processing, AddrZero otherwise.
        w3[0, 5] = AddrCur
        w3[1, 5] = AddrZero
        w3[2, 5] = AddrZero

        # Write addresses.
        # w1 = i during pair processing, AddrPass during next-pass, AddrRunning on stop.
        w3[3, 6] = 1.0
        w3[1, 6] = AddrPass
        w3[2, 6] = AddrRunning

        # w2 = i + 1 during pair processing, AddrCur during next-pass, AddrZero on stop.
        w3[4, 7] = 1.0
        w3[1, 7] = AddrCur
        w3[2, 7] = AddrZero

        # w3 = AddrCur during pair processing, AddrZero otherwise.
        w3[0, 8] = AddrCur
        w3[1, 8] = AddrZero
        w3[2, 8] = AddrZero

        self.net.weights[2] = w3
        self.net.biases[2] = b3


    def __call__(self, i, p, z):
        """Input:  (i, p, z)
        Output: (g_process, g_next_pass, g_stop, r1, r2, r3, w1, w2, w3)

        The third input is present so the controller has the same three-scalar
        control interface used elsewhere in this project, even though sorting does
        not require the zero value for branch discrimination.
        """
        x = np.asarray([i, p, z], dtype=np.float32)
        out = self.net.predict(x)
        return tuple(float(v) for v in out)
