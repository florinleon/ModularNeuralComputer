"""
Florin Leon, Modular Neural Computer, 2026
https://github.com/florinleon/ModularNeuralComputer
"""

import numpy as np
from mlp import MLP


class InitMinModule:
    """
    Init module.
    
    Input:
        x = (g, x1, x2, x3)
        g  = init gate
        x1 = value read from AddrFirst when init is active
        x2 = value read from AddrIdx   when init is active
        x3 = third read value, unused in this module
    
    Output:
        y = (y1, y2)
    
    Exact map:
        if g = 0, then y = (0, 0)
        if g = 1, then y = (x1, x2 + 1)
    
    Representative instances:
        x = (1,  7.3, 1, 121) -> y = (7.3, 2)
        x = (0,  7.3, 1, 121) -> y = (0,   0)
    
    The affine subnetwork computes the useful init update. The final gate then
    inhibits the module by forcing both outputs to zero whenever g = 0.
    """

    def __init__(self):
        self.net = MLP(layer_sizes=[4, 2], hidden_activation="relu", output_activation="linear")

        w = np.zeros((4, 2), dtype=np.float32)
        b = np.zeros(2, dtype=np.float32)

        w[1, 0] = 1.0
        w[2, 1] = 1.0
        b[1] = 1.0

        self.net.weights[0] = w
        self.net.biases[0] = b


    def __call__(self, g, x1, x2, x3):
        """
        Input:  (g, x1, x2, x3)
        Output: (g * x1, g * (x2 + 1))
        
        The gate is carried explicitly in the module interface. Only the active
        module produces a non-zero write pair.
        """
        x = np.asarray([g, x1, x2, x3], dtype=np.float32)
        out = self.net.predict(x)
        return float(g * out[0]), float(g * out[1])


class UpdateMinModule:
    """
    Update module.
    
    Input:
        x = (g, x1, x2, x3)
        g  = update gate
        x1 = current running minimum
        x2 = current candidate value
        x3 = current array index
    
    Output:
        y = (y1, y2)
    
    Exact map:
        if g = 0, then y = (0, 0)
        if g = 1, then y = (min(x1, x2), x3 + 1)
    
    Representative instances:
        x = (1,  5.0,  3.0, 2) -> y = (3.0, 3)
        x = (1, -2.0, -7.0, 8) -> y = (-7.0, 9)
        x = (0,  5.0,  3.0, 2) -> y = (0.0, 0)
    
    The hidden layer builds the positive-part terms needed for the exact minimum.
    The gate then suppresses the whole module when the update branch is inactive.
    """

    def __init__(self):
        self.net = MLP(layer_sizes=[4, 8, 2], hidden_activation="relu", output_activation="linear")

        w1 = np.zeros((4, 8), dtype=np.float32)
        b1 = np.zeros(8, dtype=np.float32)

        w1[1, 0] = 1.0
        w1[1, 1] = -1.0
        w1[2, 2] = 1.0
        w1[2, 3] = -1.0
        w1[1, 4] = 1.0
        w1[2, 4] = -1.0
        w1[1, 5] = -1.0
        w1[2, 5] = 1.0
        w1[3, 6] = 1.0
        b1[7] = 1.0

        self.net.weights[0] = w1
        self.net.biases[0] = b1

        w2 = np.zeros((8, 2), dtype=np.float32)
        b2 = np.zeros(2, dtype=np.float32)

        w2[0, 0] = 0.5
        w2[1, 0] = -0.5
        w2[2, 0] = 0.5
        w2[3, 0] = -0.5
        w2[4, 0] = -0.5
        w2[5, 0] = -0.5

        w2[6, 1] = 1.0
        w2[7, 1] = 1.0

        self.net.weights[1] = w2
        self.net.biases[1] = b2


    def __call__(self, g, x1, x2, x3):
        """
        Input:  (g, x1, x2, x3)
        Output: (g * min(x1, x2), g * (x3 + 1))
        """
        x = np.asarray([g, x1, x2, x3], dtype=np.float32)
        out = self.net.predict(x)
        return float(g * out[0]), float(g * out[1])


class StopModule:
    """
    Stop module.
    
    Input:
        x = (g, x1, x2, x3)
        g  = stop gate
        x1 = current running minimum
        x2 = unused in this module
        x3 = unused in this module
    
    Output:
        y = (y1, y2)
    
    Exact map:
        if g = 0, then y = (0, 0)
        if g = 1, then y = (-1, x1)
    
    Representative instances:
        x = (1,  2.5, 0, 0) -> y = (-1, 2.5)
        x = (0,  2.5, 0, 0) -> y = (0,  0)
    
    The first output turns the running flag off. The second preserves the final
    minimum in its working-memory slot.
    """

    def __init__(self):
        self.net = MLP(layer_sizes=[4, 2], hidden_activation="relu", output_activation="linear")

        w = np.zeros((4, 2), dtype=np.float32)
        b = np.zeros(2, dtype=np.float32)

        b[0] = -1.0
        w[1, 1] = 1.0

        self.net.weights[0] = w
        self.net.biases[0] = b


    def __call__(self, g, x1, x2, x3):
        """
        Input:  (g, x1, x2, x3)
        Output: (g * -1, g * x1)
        """
        x = np.asarray([g, x1, x2, x3], dtype=np.float32)
        out = self.net.predict(x)
        return float(g * out[0]), float(g * out[1])
