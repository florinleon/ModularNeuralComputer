"""
Florin Leon, Modular Neural Computer, 2026
https://github.com/florinleon/ModularNeuralComputer
"""

import numpy as np
from collections import OrderedDict
from mlp import MLP


def canonical_key(values):
    """Convert a vector of numeric values into a stable integer tuple. The fixed A*
    instance uses only integer-valued quantities at runtime, even though the memory
    stores them as floats. Canonical integer tuples therefore provide a clean exact
    domain for lookup compilation."""
    return tuple(int(round(float(value))) for value in values)


def deduplicate_examples(input_vectors, output_vectors):
    """Remove duplicate examples while preserving order. Repeated inputs must map to
    identical outputs; otherwise the symbolic specification would be inconsistent."""
    table = OrderedDict()
    for input_vector, output_vector in zip(input_vectors, output_vectors):
        key = canonical_key(input_vector)
        output_key = tuple(float(value) for value in output_vector)
        if key in table:
            if table[key] != output_key:
                raise ValueError("Inconsistent lookup table for input %s" % (key,))
        else:
            table[key] = output_key

    dedup_inputs = [np.array(key, dtype=np.float32) for key in table.keys()]
    dedup_outputs = [np.array(value, dtype=np.float32) for value in table.values()]
    return dedup_inputs, dedup_outputs


def build_exact_lookup_mlp(input_vectors, output_vectors):
    """Compile an exact finite lookup table into a ReLU MLP.

    The construction uses two hidden layers.

    Layer 1 computes all coordinatewise absolute differences |x_j - t_j| for every
    stored example t. Each absolute value is represented as
    ReLU(x_j - t_j) + ReLU(t_j - x_j).

    Layer 2 sums these absolute differences for each example and applies
    ReLU(1 - sum_j |x_j - t_j|). Because all valid runtime inputs are integers,
    this yields the exact indicator 1 for the matching example and 0 for every
    other stored example.

    The output layer then forms the exact table value as a linear combination of
    these one-hot indicators."""
    dedup_inputs, dedup_outputs = deduplicate_examples(input_vectors, output_vectors)

    if len(dedup_inputs) == 0:
        raise ValueError("At least one example is required")

    input_dim = int(dedup_inputs[0].shape[0])
    output_dim = int(dedup_outputs[0].shape[0])
    num_examples = len(dedup_inputs)

    hidden_1_dim = 2 * input_dim * num_examples
    hidden_2_dim = num_examples

    net = MLP(
        layer_sizes=[input_dim, hidden_1_dim, hidden_2_dim, output_dim],
        hidden_activation="relu",
        output_activation="linear"
    )

    w1 = np.zeros((input_dim, hidden_1_dim), dtype=np.float32)
    b1 = np.zeros(hidden_1_dim, dtype=np.float32)

    for example_index, input_vector in enumerate(dedup_inputs):
        for input_index in range(input_dim):
            base = 2 * (example_index * input_dim + input_index)
            target = float(input_vector[input_index])

            w1[input_index, base] = 1.0
            b1[base] = -target

            w1[input_index, base + 1] = -1.0
            b1[base + 1] = target

    w2 = np.zeros((hidden_1_dim, hidden_2_dim), dtype=np.float32)
    b2 = np.ones(hidden_2_dim, dtype=np.float32)

    for example_index in range(num_examples):
        start = 2 * example_index * input_dim
        end = start + 2 * input_dim
        w2[start:end, example_index] = -1.0

    w3 = np.zeros((hidden_2_dim, output_dim), dtype=np.float32)
    b3 = np.zeros(output_dim, dtype=np.float32)

    for example_index, output_vector in enumerate(dedup_outputs):
        w3[example_index, :] = output_vector

    net.weights[0] = w1
    net.biases[0] = b1
    net.weights[1] = w2
    net.biases[1] = b2
    net.weights[2] = w3
    net.biases[2] = b3

    return net
