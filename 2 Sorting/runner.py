"""
Florin Leon, Modular Neural Computer, 2026
https://github.com/florinleon/ModularNeuralComputer
"""

import random
from associative_memory import Memory
from config import *
from controller import Controller
from modules import ProcessPairModule, NextPassModule, StopModule


def init_system(memory, input_data):
    """
    Load one sorting instance into scalar memory.

    Memory layout:
        AddrLen      stores n
        1..n         store the input array
        AddrCur      stores the current pair index i
        AddrPass     stores the current pass limit p
        AddrZero     stores a permanent zero cell
        AddrRunning  stores the running flag

    The initial sorting state is i = 1 and p = n. This corresponds to the first
    left-to-right pass of adjacent compare-and-swap operations.
    """
    n = int(input_data[0])
    memory.write(AddrLen, float(n))

    for i in range(1, n + 1):
        memory.write(i, float(input_data[i]))

    memory.write(AddrCur, 1.0)
    memory.write(AddrPass, float(n))
    memory.write(AddrZero, 0.0)
    memory.write(AddrRunning, 1.0)


def run_step(memory, controller, modules):
    """
    Execute one neural sorting step with no symbolic dispatch.

    Stage 1. Fixed control reads.
        c = (i, p, z) = (memory[AddrCur], memory[AddrPass], memory[AddrZero])

    Stage 2. Controller forward pass.
        (g_process, g_next_pass, g_stop, r1, r2, r3, w1, w2, w3) = Controller(c)

    Stage 3. Functional reads.
        x = (x1, x2, x3) = (memory[r1], memory[r2], memory[r3])

    Stage 4. Homogeneous gated set of modules.
        ProcessPairModule(g_process,   x1, x2, x3) -> (u1_p, u2_p, u3_p)
        NextPassModule  (g_next_pass, x1, x2, x3) -> (u1_n, u2_n, u3_n)
        StopModule      (g_stop,      x1, x2, x3) -> (u1_s, u2_s, u3_s)

    Stage 5. Add the module outputs and write them back.
        The controller gates are one-hot on legal execution states, so exactly one
        module contributes a non-zero triple at each step.
    """
    i = float(memory.read(AddrCur).item())
    p = float(memory.read(AddrPass).item())
    z = float(memory.read(AddrZero).item())

    g_process, g_next_pass, g_stop, r1, r2, r3, w1, w2, w3 = controller(i, p, z)

    x1 = float(memory.read(r1).item())
    x2 = float(memory.read(r2).item())
    x3 = float(memory.read(r3).item())

    gates = [g_process, g_next_pass, g_stop]
    output_triples = []
    for gate, module in zip(gates, modules):
        output_triples.append(module(gate, x1, x2, x3))

    y1 = sum(triple[0] for triple in output_triples)
    y2 = sum(triple[1] for triple in output_triples)
    y3 = sum(triple[2] for triple in output_triples)

    memory.write(w1, y1)
    memory.write(w2, y2)
    memory.write(w3, y3)


def run_system(input_data):
    """
    Run the fixed-depth neural sorting program.

    The symbolic routine performs exactly

        sum_{p=2}^{n} p + 1 = n (n + 1) / 2

    steps:
        p - 1 pair-processing steps and one pass-transition step for each pass
        p = n, n - 1, ..., 2, followed by one final stop step at p = 1.

    The outer loop therefore has a closed-form horizon and does not inspect the
    running flag, although the running cell is still part of the program state.
    """
    memory = Memory(size=MemSize, value_dim=ValueDim)
    controller = Controller()
    modules = [ProcessPairModule(), NextPassModule(), StopModule()]

    init_system(memory, input_data)

    n = int(input_data[0])
    steps = n * (n + 1) // 2
    for _ in range(steps):
        run_step(memory, controller, modules)

    result = [round(float(memory.read(i).item()), 3) for i in range(1, n + 1)]
    return memory, result, steps


def run_system_once():
    """Build one random instance, execute the neural sorting program, and print the result."""
    n = random.randint(20, 30)
    vector = [int(random.uniform(-9.99, 9.99) * 1000) / 1000.0 for _ in range(n)]
    input_data = [n] + vector

    print(f"Input vector: {vector}\n")
    print(f"True sorted: {sorted(vector)}\n")

    memory, result, steps = run_system(input_data)

    print(f"Steps taken: {steps}")
    print(f"Neural sort: {result}")
    #print()
    #memory.display()


if __name__ == "__main__":
    run_system_once()
