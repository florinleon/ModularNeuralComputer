"""
Florin Leon, Modular Neural Computer, 2026
https://github.com/florinleon/ModularNeuralComputer
"""

import numpy as np
from config import *


class Memory:
    """
    Scalar associative memory with almost-discrete addressing.
    
    Each address is represented by a one-hot key vector. Reads and writes use the
    same temperature-controlled attention rule, so integer addresses behave like
    ordinary memory slots and non-integer addresses interpolate smoothly between
    neighbouring slots. Values remain scalar because value_dim = 1.
    """

    def __init__(self, size, value_dim):
        """
        size is the number of addressable slots.
        value_dim is the dimensionality of each stored value. The current program
        keeps it equal to one, which makes every slot a scalar cell.
        """
        self.size = size
        self.value_dim = value_dim
        self.K = np.eye(size, dtype=np.float32)
        self.V = np.zeros((size, value_dim), dtype=np.float32)
        self.temperature = 0.0001
        self.alpha = 1.0
        self.file = open("memlog.txt", "w")


    def __del__(self):
        """Close the trace file when the memory object goes out of scope."""
        self.file.close()


    def read(self, key):
        """
        Input:  key = a scalar address.
        Output: y = the scalar stored at that address.
        
        If key is an integer, the read is effectively hard because the temperature
        is extremely small. If key is fractional, the read interpolates between
        adjacent slots, which keeps the addressing interface continuous.
        """
        key_vector = self.to_one_hot(key)
        scores = np.dot(self.K, key_vector)
        weights = self.temperature_softmax(scores, self.temperature)
        result = np.dot(weights, self.V)

        self.file.write(f"{key} -> {result.item()}\n")
        self.file.flush()

        return result


    def write(self, key, value):
        """
        Input:  key = a scalar address, value = a scalar or length-one vector.
        Output: none. The addressed slot is overwritten through soft attention.
        
        With alpha = 1 and near-discrete attention, writing to an integer address
        behaves like an ordinary hard overwrite.
        """
        key_vector = self.to_one_hot(key)
        value_vector = np.asarray(value, dtype=np.float32).reshape(self.value_dim)

        self.file.write(f"{key} <- {value_vector.item()}\n")
        self.file.flush()

        scores = np.dot(self.K, key_vector)
        weights = self.temperature_softmax(scores, self.temperature)

        for i in range(self.size):
            self.V[i] = self.alpha * weights[i] * value_vector + (1.0 - self.alpha * weights[i]) * self.V[i]


    def delete(self, key):
        """
        Soft delete. Highly attended slots are pushed toward zero and the rest are
        left essentially unchanged.
        """
        key_vector = self.to_one_hot(key)
        scores = np.dot(self.K, key_vector)
        weights = self.temperature_softmax(scores, self.temperature)
        self.V *= (1.0 - weights[:, np.newaxis])


    def to_one_hot(self, key):
        """
        Convert a scalar address into the shared key space.
        
        Integer addresses select a single slot. Fractional addresses split mass
        linearly between neighbours, which gives the continuous addressing rule
        used by the neural controller.
        """
        one_hot = np.zeros(self.size, dtype=np.float32)

        if isinstance(key, (int, np.integer)):
            one_hot[int(key)] = 1.0
            return one_hot

        key = float(key)
        lower = int(np.floor(key))
        upper = int(np.ceil(key))

        if lower == upper:
            one_hot[lower] = 1.0
            return one_hot

        proportion = key - lower
        one_hot[lower] = 1.0 - proportion
        one_hot[upper] = proportion
        return one_hot


    def temperature_softmax(self, scores, temperature):
        """
        Standard temperature-scaled softmax with the usual max-subtraction step
        for numerical stability.
        """
        shifted = scores - np.max(scores)
        exp_scores = np.exp(shifted / temperature)
        return exp_scores / np.sum(exp_scores)


    def display(self):
        """Print only non-zero cells."""
        print("Memory contents:")
        for i, value in enumerate(self.V):
            if np.any(value != 0.0):
                print(f"{i}: {value.item()}")
        print()
