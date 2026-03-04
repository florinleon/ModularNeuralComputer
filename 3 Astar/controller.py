"""
Florin Leon, Modular Neural Computer, 2026
https://github.com/florinleon/ModularNeuralComputer
"""

import numpy as np
from config import *
from exact_lookup import build_exact_lookup_mlp
from specification import collect_lookup_examples


class Controller:
    """The runtime controller is a pure MLP. Its input is the six-scalar control state
    (phase, scan_node, best_node, current_state, action_index, next_free_node).
    Its output concatenates:
      - 6 exact one-hot module gates;
      - 10 functional read addresses;
      - 10 write addresses.

    On every reachable execution state of the compiled A* instance, the network
    returns the exact controller decision."""

    def __init__(self):
        examples = collect_lookup_examples()
        self.net = build_exact_lookup_mlp(examples["controller_inputs"], examples["controller_outputs"])


    def read_state(self, memory):
        """Read the fixed controller input vector from memory."""
        return np.array([
            float(memory.read(AddrPhase).item()),
            float(memory.read(AddrScanNode).item()),
            float(memory.read(AddrBestNode).item()),
            float(memory.read(AddrCurrentState).item()),
            float(memory.read(AddrActionIndex).item()),
            float(memory.read(AddrNextFreeNode).item()),
        ], dtype=np.float32)


    def step(self, memory):
        """Emit the module gates together with the functional read and write addresses."""
        output_vector = self.net.predict(self.read_state(memory).reshape(1, -1))[0]

        gates = output_vector[:NumModules]
        read_start = NumModules
        read_end = read_start + NumReadHeads
        read_addresses = output_vector[read_start:read_end].tolist()
        write_addresses = output_vector[read_end:read_end + NumWriteHeads].tolist()

        return gates, read_addresses, write_addresses
