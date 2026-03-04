"""
Florin Leon, Modular Neural Computer, 2026
https://github.com/florinleon/ModularNeuralComputer
"""

AddrMin = 1
MemSize = 500
ValueDim = 1

# Global control cells.
AddrRunning = 10
AddrPhase = 11
AddrStartState = 12
AddrGoalState = 13
AddrInf = 14
AddrZero = 15
AddrOne = 16
AddrNextFreeNode = 17
AddrScanNode = 18
AddrBestNode = 19
AddrBestF = 20
AddrCurrentNode = 21
AddrCurrentState = 22
AddrCurrentG = 23
AddrActionIndex = 24
AddrSolutionNode = 25

PhaseStop = 0
PhaseInitRoot = 1
PhaseStartOpenScan = 2
PhaseScanOpenNode = 3
PhaseFinishOpenScan = 4
PhaseGoalTest = 5
PhaseExpandAction = 6
NumModules = 6

# Fixed graph description blocks.
GraphBase = 100
GraphStride = 10
GraphOffHeuristic = 0
GraphOffActionCount = 1
GraphOffSucc1 = 2
GraphOffCost1 = 3
GraphOffSucc2 = 4
GraphOffCost2 = 5

# Generated search-node records.
NodeBase = 200
NodeStride = 10
NodeOffState = 0
NodeOffParent = 1
NodeOffAction = 2
NodeOffG = 3
NodeOffF = 4
NodeOffOpen = 5

NumReadHeads = 10
NumWriteHeads = 10

StateS = 1
StateA = 2
StateB = 3
StateC = 4
StateD = 5
StateE = 6
StateG = 7

StateNames = {
    StateS: "S",
    StateA: "A",
    StateB: "B",
    StateC: "C",
    StateD: "D",
    StateE: "E",
    StateG: "G",
}

GraphDescription = {
    StateS: {"heuristic": 7.0, "successors": [(StateA, 2.0), (StateB, 4.0)]},
    StateA: {"heuristic": 6.0, "successors": [(StateC, 2.0), (StateD, 5.0)]},
    StateB: {"heuristic": 4.0, "successors": [(StateD, 1.0), (StateE, 6.0)]},
    StateC: {"heuristic": 7.0, "successors": [(StateG, 7.0)]},
    StateD: {"heuristic": 3.0, "successors": [(StateG, 3.0)]},
    StateE: {"heuristic": 2.0, "successors": [(StateG, 2.0)]},
    StateG: {"heuristic": 0.0, "successors": []},
}


def graph_base(state_id):
    return GraphBase + GraphStride * int(state_id)


def addr_graph_heuristic(state_id):
    return graph_base(state_id) + GraphOffHeuristic


def addr_graph_action_count(state_id):
    return graph_base(state_id) + GraphOffActionCount


def addr_graph_succ_1(state_id):
    return graph_base(state_id) + GraphOffSucc1


def addr_graph_cost_1(state_id):
    return graph_base(state_id) + GraphOffCost1


def addr_graph_succ_2(state_id):
    return graph_base(state_id) + GraphOffSucc2


def addr_graph_cost_2(state_id):
    return graph_base(state_id) + GraphOffCost2


def node_base(node_index):
    return NodeBase + NodeStride * int(node_index)


def addr_node_state(node_index):
    return node_base(node_index) + NodeOffState


def addr_node_parent(node_index):
    return node_base(node_index) + NodeOffParent


def addr_node_action(node_index):
    return node_base(node_index) + NodeOffAction


def addr_node_g(node_index):
    return node_base(node_index) + NodeOffG


def addr_node_f(node_index):
    return node_base(node_index) + NodeOffF


def addr_node_open(node_index):
    return node_base(node_index) + NodeOffOpen
