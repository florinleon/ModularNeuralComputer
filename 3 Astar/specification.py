"""
Florin Leon, Modular Neural Computer, 2026
https://github.com/florinleon/ModularNeuralComputer
"""

from associative_memory import Memory
from config import *


def write_problem_instance(memory):
    """Write the fixed graph instance into memory. This specification-side helper is
    shared by the table-construction code for the controller and the functional
    modules. The runtime program uses the same concrete graph."""
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


def init_spec_memory():
    """Initialize the symbolic execution state used to compile the pure runtime MLPs."""
    memory = Memory(size=MemSize, value_dim=ValueDim)
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
    return memory


def read_controller_state(memory):
    """Return the 6 controller input values used by the pure neural runtime."""
    return [
        float(memory.read(AddrPhase).item()),
        float(memory.read(AddrScanNode).item()),
        float(memory.read(AddrBestNode).item()),
        float(memory.read(AddrCurrentState).item()),
        float(memory.read(AddrActionIndex).item()),
        float(memory.read(AddrNextFreeNode).item()),
    ]


def oracle_controller_step(memory):
    """Explicit symbolic controller used only to generate exact lookup examples for the
    runtime controller MLP."""
    phase = int(round(float(memory.read(AddrPhase).item())))
    scan_node = int(round(float(memory.read(AddrScanNode).item())))
    best_node = int(round(float(memory.read(AddrBestNode).item())))
    current_state = int(round(float(memory.read(AddrCurrentState).item())))
    action_index = int(round(float(memory.read(AddrActionIndex).item())))
    next_free = int(round(float(memory.read(AddrNextFreeNode).item())))

    gates = [0.0] * NumModules
    if 1 <= phase <= NumModules:
        gates[phase - 1] = 1.0

    read_addresses = [AddrZero] * NumReadHeads
    write_addresses = [AddrZero] * NumWriteHeads

    if phase == PhaseInitRoot:
        start_state = int(round(float(memory.read(AddrStartState).item())))
        read_addresses = [
            AddrStartState,
            addr_graph_heuristic(start_state),
            AddrZero,
            AddrOne,
            AddrRunning,
            AddrSolutionNode,
            AddrZero,
            AddrZero,
            AddrZero,
            AddrZero,
        ]
        write_addresses = [
            addr_node_state(1),
            addr_node_parent(1),
            addr_node_action(1),
            addr_node_g(1),
            addr_node_f(1),
            addr_node_open(1),
            AddrNextFreeNode,
            AddrPhase,
            AddrRunning,
            AddrSolutionNode,
        ]

    elif phase == PhaseStartOpenScan:
        read_addresses = [
            AddrInf,
            AddrOne,
            AddrZero,
            AddrZero,
            AddrZero,
            AddrZero,
            AddrZero,
            AddrZero,
            AddrZero,
            AddrZero,
        ]
        write_addresses = [
            AddrScanNode,
            AddrBestNode,
            AddrBestF,
            AddrPhase,
            AddrZero,
            AddrZero,
            AddrZero,
            AddrZero,
            AddrZero,
            AddrZero,
        ]

    elif phase == PhaseScanOpenNode:
        if 1 <= scan_node < next_free:
            open_address = addr_node_open(scan_node)
            f_address = addr_node_f(scan_node)
        else:
            open_address = AddrZero
            f_address = AddrZero

        read_addresses = [
            AddrScanNode,
            AddrNextFreeNode,
            open_address,
            f_address,
            AddrBestF,
            AddrBestNode,
            AddrOne,
            AddrPhase,
            AddrZero,
            AddrZero,
        ]
        write_addresses = [
            AddrBestF,
            AddrBestNode,
            AddrScanNode,
            AddrPhase,
            AddrZero,
            AddrZero,
            AddrZero,
            AddrZero,
            AddrZero,
            AddrZero,
        ]

    elif phase == PhaseFinishOpenScan:
        if best_node > 0:
            state_address = addr_node_state(best_node)
            g_address = addr_node_g(best_node)
            open_address = addr_node_open(best_node)
        else:
            state_address = AddrZero
            g_address = AddrZero
            open_address = AddrZero

        read_addresses = [
            AddrBestNode,
            state_address,
            g_address,
            AddrRunning,
            AddrSolutionNode,
            open_address,
            AddrPhase,
            AddrZero,
            AddrZero,
            AddrZero,
        ]
        write_addresses = [
            AddrRunning,
            AddrSolutionNode,
            AddrCurrentNode,
            AddrCurrentState,
            AddrCurrentG,
            open_address,
            AddrPhase,
            AddrZero,
            AddrZero,
            AddrZero,
        ]

    elif phase == PhaseGoalTest:
        read_addresses = [
            AddrCurrentState,
            AddrGoalState,
            AddrCurrentNode,
            AddrRunning,
            AddrSolutionNode,
            AddrOne,
            AddrPhase,
            AddrZero,
            AddrZero,
            AddrZero,
        ]
        write_addresses = [
            AddrRunning,
            AddrSolutionNode,
            AddrActionIndex,
            AddrPhase,
            AddrZero,
            AddrZero,
            AddrZero,
            AddrZero,
            AddrZero,
            AddrZero,
        ]

    elif phase == PhaseExpandAction:
        if action_index == 1:
            succ_address = addr_graph_succ_1(current_state)
            cost_address = addr_graph_cost_1(current_state)
        elif action_index == 2:
            succ_address = addr_graph_succ_2(current_state)
            cost_address = addr_graph_cost_2(current_state)
        else:
            succ_address = AddrZero
            cost_address = AddrZero

        successor_state = int(round(float(memory.read(succ_address).item())))
        if successor_state > 0:
            heuristic_address = addr_graph_heuristic(successor_state)
        else:
            heuristic_address = AddrZero

        write_addresses = [
            addr_node_state(next_free),
            addr_node_parent(next_free),
            addr_node_action(next_free),
            addr_node_g(next_free),
            addr_node_f(next_free),
            addr_node_open(next_free),
            AddrNextFreeNode,
            AddrActionIndex,
            AddrPhase,
            AddrZero,
        ]
        read_addresses = [
            AddrCurrentState,
            AddrCurrentNode,
            AddrCurrentG,
            AddrActionIndex,
            succ_address,
            cost_address,
            heuristic_address,
            AddrNextFreeNode,
            AddrPhase,
            AddrZero,
        ]

    return gates, read_addresses, write_addresses


def oracle_module_output(phase, values):
    """Explicit symbolic module specification used only to generate exact lookup tables
    for the runtime module MLPs."""
    if phase == PhaseInitRoot:
        start_state = float(values[0])
        start_h = float(values[1])
        return [start_state, 0.0, 0.0, 0.0, start_h, 1.0, 2.0, float(PhaseStartOpenScan), 1.0, 0.0]

    if phase == PhaseStartOpenScan:
        inf_value = float(values[0])
        return [1.0, 0.0, inf_value, float(PhaseScanOpenNode), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    if phase == PhaseScanOpenNode:
        scan_node = int(round(float(values[0])))
        next_free = int(round(float(values[1])))
        open_flag = float(values[2])
        candidate_f = float(values[3])
        best_f = float(values[4])
        best_node = float(values[5])

        if scan_node >= next_free:
            return [best_f, best_node, float(scan_node), float(PhaseFinishOpenScan), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        new_best_f = best_f
        new_best_node = best_node
        if open_flag > 0.5 and candidate_f < best_f:
            new_best_f = candidate_f
            new_best_node = float(scan_node)

        return [new_best_f, new_best_node, float(scan_node + 1), float(PhaseScanOpenNode), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    if phase == PhaseFinishOpenScan:
        best_node = int(round(float(values[0])))
        best_state = float(values[1])
        best_g = float(values[2])
        running = float(values[3])
        solution = float(values[4])

        if best_node <= 0:
            return [-1.0, 0.0, 0.0, 0.0, 0.0, 0.0, float(PhaseStop), 0.0, 0.0, 0.0]

        return [running, solution, float(best_node), best_state, best_g, 0.0, float(PhaseGoalTest), 0.0, 0.0, 0.0]

    if phase == PhaseGoalTest:
        current_state = int(round(float(values[0])))
        goal_state = int(round(float(values[1])))
        current_node = float(values[2])
        running = float(values[3])
        solution = float(values[4])

        if current_state == goal_state:
            return [-1.0, current_node, 0.0, float(PhaseStop), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

        return [running, solution, 1.0, float(PhaseExpandAction), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    if phase == PhaseExpandAction:
        current_node = float(values[1])
        current_g = float(values[2])
        action_index = int(round(float(values[3])))
        successor_state = int(round(float(values[4])))
        edge_cost = float(values[5])
        successor_h = float(values[6])
        next_free = float(values[7])

        if successor_state <= 0:
            return [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, next_free, float(action_index), float(PhaseStartOpenScan), 0.0]

        child_g = current_g + edge_cost
        child_f = child_g + successor_h
        return [float(successor_state), current_node, float(action_index), child_g, child_f, 1.0, float(next_free + 1.0), float(action_index + 1), float(PhaseExpandAction), 0.0]

    raise ValueError("Unsupported phase %s" % phase)


def collect_lookup_examples():
    """Execute the symbolic specification once and collect every controller state and
    every active-module input/output pair encountered on the fixed A* instance."""
    memory = init_spec_memory()

    controller_inputs = []
    controller_outputs = []
    module_inputs = [[] for _ in range(NumModules)]
    module_outputs = [[] for _ in range(NumModules)]

    steps = 0
    max_steps = 200

    while float(memory.read(AddrRunning).item()) > 0.0 and steps < max_steps:
        controller_state = read_controller_state(memory)
        gates, read_addresses, write_addresses = oracle_controller_step(memory)

        controller_inputs.append(controller_state)
        controller_outputs.append(gates + [float(address) for address in read_addresses] + [float(address) for address in write_addresses])

        values = [float(memory.read(address).item()) for address in read_addresses]
        phase = int(round(controller_state[0]))
        module_index = phase - 1
        output_vector = oracle_module_output(phase, values)

        module_inputs[module_index].append(values)
        module_outputs[module_index].append(output_vector)

        for head_index in range(NumWriteHeads):
            memory.write(write_addresses[head_index], float(output_vector[head_index]))

        steps += 1

    return {
        "controller_inputs": controller_inputs,
        "controller_outputs": controller_outputs,
        "module_inputs": module_inputs,
        "module_outputs": module_outputs,
    }
