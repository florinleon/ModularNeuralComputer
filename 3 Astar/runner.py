"""
Florin Leon, Modular Neural Computer, 2026
https://github.com/florinleon/ModularNeuralComputer
"""

from associative_memory import Memory
from config import *
from controller import Controller
from modules import *


def write_problem_instance(memory):
    """Write the fixed finite graph instance into memory. This part is intentionally
    non-neural: the graph, costs, and heuristic values define the problem instance
    and are present in memory before the modular neural search starts."""
    memory.write(AddrStartState, float(StateS))
    memory.write(AddrGoalState, float(StateG))
    memory.write(AddrInf, 1000000.0)
    memory.write(AddrZero, 0.0)
    memory.write(AddrOne, 1.0)

    for state_id, descriptor in GraphDescription.items():
        successors = descriptor["successors"]

        memory.write(addr_graph_heuristic(state_id), float(descriptor["heuristic"]))
        memory.write(addr_graph_action_count(state_id), float(len(successors)))

        if len(successors) >= 1:
            memory.write(addr_graph_succ_1(state_id), float(successors[0][0]))
            memory.write(addr_graph_cost_1(state_id), float(successors[0][1]))

        if len(successors) >= 2:
            memory.write(addr_graph_succ_2(state_id), float(successors[1][0]))
            memory.write(addr_graph_cost_2(state_id), float(successors[1][1]))


def init_system(memory):
    """Initialize the control region for the compiled A* program. Search nodes are left
    at zero because the memory starts blank and node creation is handled by modules."""
    write_problem_instance(memory)

    memory.write(AddrRunning, 1.0)
    memory.write(AddrPhase, float(PhaseInitRoot))
    memory.write(AddrNextFreeNode, 1.0)
    memory.write(AddrScanNode, 0.0)
    memory.write(AddrBestNode, 0.0)
    memory.write(AddrBestF, 0.0)
    memory.write(AddrCurrentNode, 0.0)
    memory.write(AddrCurrentState, 0.0)
    memory.write(AddrCurrentG, 0.0)
    memory.write(AddrActionIndex, 0.0)
    memory.write(AddrSolutionNode, 0.0)


def read_functional_values(memory, read_addresses):
    """Read all controller-selected functional inputs for the current step."""
    return [float(memory.read(address).item()) for address in read_addresses]


def run_step(memory, controller, modules):
    """Execute one full step of the modular neural program. The controller emits exact
    gates and addresses, all modules receive the same read values, inactive modules
    return zero, and the active module contributes the unique non-zero write vector."""
    gates, read_addresses, write_addresses = controller.step(memory)
    values = read_functional_values(memory, read_addresses)

    total_output = zero_outputs()
    for module_index in range(len(modules)):
        module_output = modules[module_index](float(gates[module_index]), values)
        total_output += module_output

    for head_index in range(NumWriteHeads):
        memory.write(write_addresses[head_index], float(total_output[head_index]))


def reconstruct_solution_path(memory):
    """Recover the path of graph states from the final solution node by following the
    stored parent pointers back to the root of the generated search tree."""
    solution_node = int(round(float(memory.read(AddrSolutionNode).item())))
    if solution_node <= 0:
        return [], 0.0

    path = []
    current_node = solution_node
    while current_node > 0:
        state_id = int(round(float(memory.read(addr_node_state(current_node)).item())))
        path.append(StateNames[state_id])
        current_node = int(round(float(memory.read(addr_node_parent(current_node)).item())))

    path.reverse()
    total_cost = float(memory.read(addr_node_g(solution_node)).item())
    return path, total_cost


def display_search_nodes(memory):
    """Display the generated search nodes so the evolution of the open list and the
    chosen solution can be inspected directly from the scalar memory contents."""
    next_free = int(round(float(memory.read(AddrNextFreeNode).item())))
    print("Generated search nodes:")
    for node_index in range(1, next_free):
        state_id = int(round(float(memory.read(addr_node_state(node_index)).item())))
        parent = int(round(float(memory.read(addr_node_parent(node_index)).item())))
        action = int(round(float(memory.read(addr_node_action(node_index)).item())))
        g_value = float(memory.read(addr_node_g(node_index)).item())
        f_value = float(memory.read(addr_node_f(node_index)).item())
        open_flag = int(round(float(memory.read(addr_node_open(node_index)).item())))
        print(
            f"node {node_index}: state={StateNames.get(state_id, state_id)}, "
            f"parent={parent}, action={action}, G={g_value:.0f}, F={f_value:.0f}, open={open_flag}"
        )
    print()


def run_system_once():
    """Run the fixed finite A* instance to completion and print the recovered solution."""
    memory = Memory(size=MemSize, value_dim=ValueDim)
    init_system(memory)

    controller = Controller()
    modules = [
        InitRootModule(),
        StartOpenScanModule(),
        ScanOpenNodeModule(),
        FinishOpenScanModule(),
        GoalTestModule(),
        ExpandActionModule(),
    ]

    steps = 0
    max_steps = 200

    while float(memory.read(AddrRunning).item()) > 0.0 and steps < max_steps:
        run_step(memory, controller, modules)
        steps += 1

    path, total_cost = reconstruct_solution_path(memory)

    print(f"Steps taken: {steps}")
    print(f"Solution node: {int(round(float(memory.read(AddrSolutionNode).item())))}")
    print(f"Path: {' -> '.join(path) if path else 'no solution'}")
    print(f"Total cost: {total_cost:.0f}")
    print()
    display_search_nodes(memory)


if __name__ == "__main__":
    run_system_once()
