"""
Florin Leon, Modular Neural Computer, 2026
https://github.com/florinleon/ModularNeuralComputer
"""

import numpy as np
from mlp import MLP
from config import *


class Controller:
    """
    Controller MLP.
    
    Input:
        x = (i, n, z)
        i = current array index
        n = array length
        z = constant zero read from memory
    
    Output:
        y = (g_init, g_update, g_stop, r1, r2, r3, w1, w2)
    
    Exact map on legal execution states:
        if i = 1, then
            y = (1, 0, 0, AddrFirst, AddrIdx, AddrIdx, AddrMin, AddrIdx)
    
        if 2 <= i <= n, then
            y = (0, 1, 0, AddrMin, i, AddrIdx, AddrMin, AddrIdx)
    
        if i = n + 1, then
            y = (0, 0, 1, AddrMin, i, AddrIdx, AddrRunning, AddrMin)
    
    Representative instances:
        x = (1, 5, 0) -> y = (1, 0, 0, 1,   121, 121, 120, 121)
        x = (2, 5, 0) -> y = (0, 1, 0, 120, 2,   121, 120, 121)
        x = (5, 5, 0) -> y = (0, 1, 0, 120, 5,   121, 120, 121)
        x = (6, 5, 0) -> y = (0, 0, 1, 120, 6,   121, 123, 120)
    
    The third input is present so the controller has a homogeneous fixed-size
    input interface. The output addresses are produced directly by the MLP, not
    by any symbolic post-processing step.
    """

    def __init__(self):
        self.net = MLP(layer_sizes=[3, 6, 8], hidden_activation="relu", output_activation="linear")

        w1 = np.zeros((3, 6), dtype=np.float32)
        b1 = np.zeros(6, dtype=np.float32)

        # h0 = ReLU(1.5 - i)
        w1[0, 0] = -1.0
        b1[0] = 1.5

        # h1 = ReLU(0.5 - i)
        w1[0, 1] = -1.0
        b1[1] = 0.5

        # h2 = ReLU(i - n - 0.5)
        w1[0, 2] = 1.0
        w1[1, 2] = -1.0
        b1[2] = -0.5

        # h3 = ReLU(i - n - 1.5)
        w1[0, 3] = 1.0
        w1[1, 3] = -1.0
        b1[3] = -1.5

        # h4 = ReLU(i)
        w1[0, 4] = 1.0

        # h5 = ReLU(1)
        b1[5] = 1.0

        self.net.weights[0] = w1
        self.net.biases[0] = b1

        w2 = np.zeros((6, 8), dtype=np.float32)
        b2 = np.zeros(8, dtype=np.float32)

        # g_init = 2*h0 - 2*h1
        w2[0, 0] = 2.0
        w2[1, 0] = -2.0

        # g_update = h5 - 2*h0 + 2*h1 - 2*h2 + 2*h3
        w2[0, 1] = -2.0
        w2[1, 1] = 2.0
        w2[2, 1] = -2.0
        w2[3, 1] = 2.0
        w2[5, 1] = 1.0

        # g_stop = 2*h2 - 2*h3
        w2[2, 2] = 2.0
        w2[3, 2] = -2.0

        # r1 = AddrMin + (AddrFirst - AddrMin) * g_init
        w2[0, 3] = 2.0 * (AddrFirst - AddrMin)
        w2[1, 3] = -2.0 * (AddrFirst - AddrMin)
        w2[5, 3] = AddrMin

        # r2 = i + (AddrIdx - 1) * g_init
        w2[0, 4] = 2.0 * (AddrIdx - 1)
        w2[1, 4] = -2.0 * (AddrIdx - 1)
        w2[4, 4] = 1.0

        # r3 = AddrIdx
        w2[5, 5] = AddrIdx

        # w1 = AddrIdx + (AddrRunning - AddrMin) * g_stop
        w2[2, 6] = 2.0 * (AddrRunning - AddrMin)
        w2[3, 6] = -2.0 * (AddrRunning - AddrMin)
        w2[5, 6] = AddrMin

        # w2 = AddrIdx + (AddrMin - AddrIdx) * g_stop
        w2[2, 7] = 2.0 * (AddrMin - AddrIdx)
        w2[3, 7] = -2.0 * (AddrMin - AddrIdx)
        w2[5, 7] = AddrIdx

        self.net.weights[1] = w2
        self.net.biases[1] = b2


    def __call__(self, i, n, z):
        """
        Input:  (i, n, z)
        Output: (g_init, g_update, g_stop, r1, r2, r3, w1, w2)
        
        This is a pure forward pass through the handcrafted MLP. The caller reads
        the fixed control cells, passes the three values in, and receives both the
        branch gates and the memory addresses directly from the network output.
        """
        x = np.asarray([i, n, z], dtype=np.float32)
        out = self.net.predict(x)
        return tuple(float(v) for v in out)
