"""
Florin Leon, Modular Neural Computer, 2026
https://github.com/florinleon/ModularNeuralComputer
"""

import random
from associative_memory import Memory
from config import *
from controller import Controller
from modules import InitMinModule, UpdateMinModule, StopModule


def init_system(memory, input_data):
    """
    Load one minimum-search instance into scalar memory.
    
    Memory layout:
        AddrLen      stores n
        1..n         store the input array
        AddrMin      stores the running minimum
        AddrIdx      stores the current index
        AddrZero     stores a permanent zero cell
        AddrRunning  stores the running flag
    """
    n = int(input_data[0])
    memory.write(AddrLen, float(n))

    for i in range(1, n + 1):
        memory.write(i, float(input_data[i]))

    memory.write(AddrMin, 0.0)
    memory.write(AddrIdx, 1.0)
    memory.write(AddrZero, 0.0)
    memory.write(AddrRunning, 1.0)


def run_step(memory, controller, modules):
    """
    Execute one neural program step with no symbolic branch dispatch.
    
    Stage 1. Fixed control reads.
        c = (i, n, z) = (memory[AddrIdx], memory[AddrLen], memory[AddrZero])
    
    Stage 2. Controller forward pass.
        (g_init, g_update, g_stop, r1, r2, r3, w1, w2) = Controller(c)
    
    Stage 3. Functional reads.
        x = (x1, x2, x3) = (memory[r1], memory[r2], memory[r3])
    
    Stage 4. Homogeneous gated module bank.
        InitMinModule   (g_init,   x1, x2, x3) -> (u1_init,   u2_init)
        UpdateMinModule (g_update, x1, x2, x3) -> (u1_update, u2_update)
        StopModule      (g_stop,   x1, x2, x3) -> (u1_stop,   u2_stop)
    
    Stage 5. Add the module outputs and write them back.
        Because the controller gates are one-hot on legal states, only one module
        contributes a non-zero pair at each step.
    """
    i = float(memory.read(AddrIdx).item())
    n = float(memory.read(AddrLen).item())
    z = float(memory.read(AddrZero).item())

    g_init, g_update, g_stop, r1, r2, r3, w1, w2 = controller(i, n, z)

    x1 = float(memory.read(r1).item())
    x2 = float(memory.read(r2).item())
    x3 = float(memory.read(r3).item())

    gates = [g_init, g_update, g_stop]
    output_pairs = []
    for gate, module in zip(gates, modules):
        output_pairs.append(module(gate, x1, x2, x3))

    y1 = sum(pair[0] for pair in output_pairs)
    y2 = sum(pair[1] for pair in output_pairs)

    memory.write(w1, y1)
    memory.write(w2, y2)


def run_system(input_data):
    """
    Run the fixed-depth neural program.
    
    The minimum routine always needs exactly n + 1 steps:
        1 init step
        n - 1 update steps
        1 stop step
    
    The outer loop therefore does not inspect the running flag. The running cell
    is still written by the stop module because it remains part of the program state.
    """
    memory = Memory(size=MemSize, value_dim=ValueDim)
    controller = Controller()
    modules = [InitMinModule(), UpdateMinModule(), StopModule()]

    init_system(memory, input_data)

    steps = int(input_data[0]) + 1
    for _ in range(steps):
        run_step(memory, controller, modules)

    result = float(memory.read(AddrMin).item())
    return memory, result, steps


def run_system_once():
    """Build one random instance, execute the neural program, and print the result."""
    n = random.randint(90, 99)
    vector = [int(random.uniform(-9.99, 9.99) * 1000) / 1000.0 for _ in range(n)]
    input_data = [n] + vector

    print(f"Input vector: {vector}\n")
    print(f"True minimum: {min(vector):.3f}\n")

    memory, result, steps = run_system(input_data)

    print(f"Steps taken: {steps}")
    print(f"Neural min: {result:.3f}")
    #print()
    #memory.display()


if __name__ == "__main__":
    run_system_once()
