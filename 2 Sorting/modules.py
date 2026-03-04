"""
Florin Leon, Modular Neural Computer, 2026
https://github.com/florinleon/ModularNeuralComputer
"""

import numpy as np
from mlp import MLP


class ProcessPairModule:
    """
    Pair-processing module.

    Input:
        x = (g, x1, x2, x3)
        g  = process-pair gate
        x1 = left value a_i
        x2 = right value a_{i + 1}
        x3 = current pair index i

    Output:
        y = (y1, y2, y3)

    Exact map:
        if g = 0, then y = (0, 0, 0)
        if g = 1, then y = (min(x1, x2), max(x1, x2), x3 + 1)

    Representative instances:
        x = (1, 5.0, 3.0, 2)   -> y = (3.0, 5.0, 3)
        x = (1, 3.0, 5.0, 2)   -> y = (3.0, 5.0, 3)
        x = (1, -1.0, -4.0, 7) -> y = (-4.0, -1.0, 8)
        x = (0, 5.0, 3.0, 2)   -> y = (0.0, 0.0, 0.0)

    This module is the neural equivalent of one adjacent compare-and-swap step.
    The two data outputs rewrite the pair in sorted order, and the third output
    advances the current pair index for the next comparison.
    """

    def __init__(self):
        self.net = MLP(layer_sizes=[4, 8, 3], hidden_activation="relu", output_activation="linear")

        w1 = np.zeros((4, 8), dtype=np.float32)
        b1 = np.zeros(8, dtype=np.float32)

        # Positive and negative parts of the two compared values.
        w1[1, 0] = 1.0
        w1[1, 1] = -1.0
        w1[2, 2] = 1.0
        w1[2, 3] = -1.0

        # Positive parts of the pairwise differences.
        w1[1, 4] = 1.0
        w1[2, 4] = -1.0
        w1[1, 5] = -1.0
        w1[2, 5] = 1.0

        # Current index and constant one.
        w1[3, 6] = 1.0
        b1[7] = 1.0

        self.net.weights[0] = w1
        self.net.biases[0] = b1

        w2 = np.zeros((8, 3), dtype=np.float32)
        b2 = np.zeros(3, dtype=np.float32)

        # y1 = min(x1, x2)
        w2[0, 0] = 0.5
        w2[1, 0] = -0.5
        w2[2, 0] = 0.5
        w2[3, 0] = -0.5
        w2[4, 0] = -0.5
        w2[5, 0] = -0.5

        # y2 = max(x1, x2)
        w2[0, 1] = 0.5
        w2[1, 1] = -0.5
        w2[2, 1] = 0.5
        w2[3, 1] = -0.5
        w2[4, 1] = 0.5
        w2[5, 1] = 0.5

        # y3 = x3 + 1
        w2[6, 2] = 1.0
        w2[7, 2] = 1.0

        self.net.weights[1] = w2
        self.net.biases[1] = b2


    def __call__(self, g, x1, x2, x3):
        """
        Input:  (g, x1, x2, x3)
        Output: (g * min(x1, x2), g * max(x1, x2), g * (x3 + 1))

        Only the active module contributes non-zero writes. The gate is applied
        inside the module interface so inhibition remains local to the module.
        """
        x = np.asarray([g, x1, x2, x3], dtype=np.float32)
        out = self.net.predict(x)
        return float(g * out[0]), float(g * out[1]), float(g * out[2])


class NextPassModule:
    """
    Next-pass module.

    Input:
        x = (g, x1, x2, x3)
        g  = next-pass gate
        x1 = current pass limit p
        x2 = unused in this module
        x3 = unused in this module

    Output:
        y = (y1, y2, y3)

    Exact map:
        if g = 0, then y = (0, 0, 0)
        if g = 1, then y = (x1 - 1, 1, 0)

    Representative instances:
        x = (1, 5, 0, 0) -> y = (4, 1, 0)
        x = (1, 2, 7, 9) -> y = (1, 1, 0)
        x = (0, 5, 0, 0) -> y = (0, 0, 0)

    The first output shortens the active sorting pass by one position. The second
    output resets the current pair index to the first array location.
    """

    def __init__(self):
        self.net = MLP(layer_sizes=[4, 3], hidden_activation="relu", output_activation="linear")

        w = np.zeros((4, 3), dtype=np.float32)
        b = np.zeros(3, dtype=np.float32)

        w[1, 0] = 1.0
        b[0] = -1.0

        b[1] = 1.0

        self.net.weights[0] = w
        self.net.biases[0] = b


    def __call__(self, g, x1, x2, x3):
        """
        Input:  (g, x1, x2, x3)
        Output: (g * (x1 - 1), g * 1, g * 0)
        """
        x = np.asarray([g, x1, x2, x3], dtype=np.float32)
        out = self.net.predict(x)
        return float(g * out[0]), float(g * out[1]), float(g * out[2])


class StopModule:
    """
    Stop module.

    Input:
        x = (g, x1, x2, x3)
        g  = stop gate
        x1 = unused in this module
        x2 = unused in this module
        x3 = unused in this module

    Output:
        y = (y1, y2, y3)

    Exact map:
        if g = 0, then y = (0, 0, 0)
        if g = 1, then y = (-1, 0, 0)

    Representative instances:
        x = (1, 0, 0, 0) -> y = (-1, 0, 0)
        x = (0, 7, 8, 9) -> y = (0, 0, 0)

    The first output turns the running flag off. The remaining outputs are neutral
    values written to the constant zero cell so the final step remains homogeneous.
    """

    def __init__(self):
        self.net = MLP(layer_sizes=[4, 3], hidden_activation="relu", output_activation="linear")

        w = np.zeros((4, 3), dtype=np.float32)
        b = np.zeros(3, dtype=np.float32)

        b[0] = -1.0

        self.net.weights[0] = w
        self.net.biases[0] = b


    def __call__(self, g, x1, x2, x3):
        """
        Input:  (g, x1, x2, x3)
        Output: (g * -1, g * 0, g * 0)
        """
        x = np.asarray([g, x1, x2, x3], dtype=np.float32)
        out = self.net.predict(x)
        return float(g * out[0]), float(g * out[1]), float(g * out[2])
