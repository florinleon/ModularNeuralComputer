# Modular Neural Computer

This repository contains reference implementations for the **Modular Neural Computer (MNC)**, a memory-augmented neural architecture for **exact algorithmic computation** on variable-length inputs.

The code illustrates a modular neural execution model built from:
- an external associative memory of scalar cells,
- explicit read and write heads,
- a controller MLP,
- a homogeneous set of functional MLP modules.

Unlike end-to-end trained memory-augmented neural networks, the programs in this repository are **analytically constructed**. Each module implements an exact step of a known algorithm, and the controller emits exact gates and addresses on valid execution states. The goal is not to learn an algorithm from data, but to realize it exactly within a neural architecture.

## Included case studies

This repository currently includes three case studies:

- **Minimum of an array**: exact computation of the minimum value of a variable-length array.
- **Sorting an array**: exact in-place sorting through modular compare-and-swap phases.
- **A\*** on a fixed finite instance: exact execution of a compiled search procedure over an explicit problem instance stored in memory.

Each case study follows the same overall MNC design, while adapting the controller, module set, and memory layout to the task.

## Repository organization

A typical case-study directory contains files such as:

- `config.py` — memory layout, constants, and address definitions
- `controller.py` — pure MLP controller producing gates and memory addresses
- `modules.py` — task-specific functional MLP modules
- `runner.py` — execution loop for the case study
- `associative_memory.py` — shared scalar associative memory implementation
- `mlp.py` — shared general-purpose MLP implementation

## Citation

A detailed description of the architecture and examples can be found in this paper:

> Florin Leon, *Modular Neural Computer*, 2026, https://arxiv.org/abs/2603.13323.

## Note

The implementations are intended as reference programs rather than optimized systems. The programs are distributed in the hope that they will be useful, but without any warranty; without even the implied warranty of merchantability or fitness for a particular purpose.
