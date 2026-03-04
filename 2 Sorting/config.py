"""
Florin Leon, Modular Neural Computer, 2026
https://github.com/florinleon/ModularNeuralComputer
"""

MemSize = 1000
ValueDim = 1

AddrLen = 0
AddrFirst = 1
AddrCur = 900
AddrPass = 901
AddrZero = 902
AddrRunning = 903

NumControlReads = 3
NumReadHeads = 3
NumWriteHeads = 3

BranchProcess = 0
BranchNextPass = 1
BranchStop = 2

TrainMinN = 2
TrainMaxN = 40
TrainNumExamples = 5000
ValidNumExamples = 1000
MaxN = TrainMaxN
