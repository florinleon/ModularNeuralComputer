"""
Florin Leon, Modular Neural Computer, 2026
https://github.com/florinleon/ModularNeuralComputer
"""

import numpy as np
from config import *
from exact_lookup import build_exact_lookup_mlp
from specification import collect_lookup_examples


def zero_outputs():
    """Return the neutral ten-dimensional write vector. Inactive modules contribute
    exactly zero, so the componentwise sum over all module outputs equals the unique
    non-zero output of the active module."""
    return np.zeros(NumWriteHeads, dtype=np.float32)


# Shared cache of the symbolic examples used to compile the pure runtime modules.
CompiledExamples = None


def get_compiled_examples():
    """Collect the active-state input/output pairs for all modules once. The symbolic
    trace is used only at compile time; runtime evaluation uses only MLP forward passes."""
    global CompiledExamples
    if CompiledExamples is None:
        CompiledExamples = collect_lookup_examples()
    return CompiledExamples


class InitRootModule:
    """Input-output map on active states:
    x = (start_state, h(start), 0, 1, running, solution, 0, 0, 0, 0)
    y = (state, parent, action, G, F, open, next_free, phase, running, solution)
      = (start_state, 0, 0, 0, h(start), 1, 2, PhaseStartOpenScan, 1, 0)

    The full map is compiled into an MLP from the symbolic reference
    execution of the fixed A* instance."""

    def __init__(self):
        examples = get_compiled_examples()
        self.net = build_exact_lookup_mlp(examples["module_inputs"][0], examples["module_outputs"][0])


    def __call__(self, gate, values):
        output = self.net.predict(np.array(values, dtype=np.float32).reshape(1, -1))[0]
        return float(gate) * output


class StartOpenScanModule:
    """Input-output map on active states:
    x = (INF, 1, 0, 0, 0, 0, 0, 0, 0, 0)
    y = (scan_node, best_node, best_f, phase, 0, 0, 0, 0, 0, 0)
      = (1, 0, INF, PhaseScanOpenNode, 0, 0, 0, 0, 0, 0)

    The map is exact on every active-state input encountered in the compiled
    finite A* instance."""

    def __init__(self):
        examples = get_compiled_examples()
        self.net = build_exact_lookup_mlp(examples["module_inputs"][1], examples["module_outputs"][1])


    def __call__(self, gate, values):
        output = self.net.predict(np.array(values, dtype=np.float32).reshape(1, -1))[0]
        return float(gate) * output


class ScanOpenNodeModule:
    """Input-output map on active states:
    x = (scan_node, next_free, open_flag(scan_node), F(scan_node), best_f,
         best_node, 1, phase, 0, 0)
    y = (new_best_f, new_best_node, next_scan_node, next_phase, 0, 0, 0, 0, 0, 0)

    On the fixed problem instance this phase has a finite set of reachable input
    tuples. The exact finite map over those tuples is compiled into an MLP."""

    def __init__(self):
        examples = get_compiled_examples()
        self.net = build_exact_lookup_mlp(examples["module_inputs"][2], examples["module_outputs"][2])


    def __call__(self, gate, values):
        output = self.net.predict(np.array(values, dtype=np.float32).reshape(1, -1))[0]
        return float(gate) * output


class FinishOpenScanModule:
    """Input-output map on active states:
    x = (best_node, best_state, best_G, running, solution, open_flag(best_node),
         phase, 0, 0, 0)
    y = (running', solution', current_node, current_state, current_G,
         open(best_node)', phase', 0, 0, 0)

    The module is compiled as an exact MLP over the finite set of best-node
    situations reachable in the fixed A* search."""

    def __init__(self):
        examples = get_compiled_examples()
        self.net = build_exact_lookup_mlp(examples["module_inputs"][3], examples["module_outputs"][3])


    def __call__(self, gate, values):
        output = self.net.predict(np.array(values, dtype=np.float32).reshape(1, -1))[0]
        return float(gate) * output


class GoalTestModule:
    """Input-output map on active states:
    x = (current_state, goal_state, current_node, running, solution, 1, phase, 0, 0, 0)
    y = (running', solution', action_index, phase', 0, 0, 0, 0, 0, 0)

    Successful and unsuccessful goal tests are both represented in the
    compiled lookup realized by the MLP."""

    def __init__(self):
        examples = get_compiled_examples()
        self.net = build_exact_lookup_mlp(examples["module_inputs"][4], examples["module_outputs"][4])


    def __call__(self, gate, values):
        output = self.net.predict(np.array(values, dtype=np.float32).reshape(1, -1))[0]
        return float(gate) * output


class ExpandActionModule:
    """Input-output map on active states:
    x = (current_state, current_node, current_G, action_index, successor_state,
         edge_cost, heuristic(successor), next_free_node, phase, 0)
    y = (child_state, child_parent, child_action, child_G, child_F, child_open,
         next_free_node', action_index', phase', 0)

    The fixed graph yields a finite set of reachable expansion situations. The
    child-generation map over those situations is compiled into the MLP."""

    def __init__(self):
        examples = get_compiled_examples()
        self.net = build_exact_lookup_mlp(examples["module_inputs"][5], examples["module_outputs"][5])


    def __call__(self, gate, values):
        output = self.net.predict(np.array(values, dtype=np.float32).reshape(1, -1))[0]
        return float(gate) * output
