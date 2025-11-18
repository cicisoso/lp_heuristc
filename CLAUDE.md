# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview
Link prediction is a fundamental task in graph learning, inherently
shaped by the topology of the graph. While traditional heuristics
are grounded in graph topology, they encounter challenges in generalizing across diverse graphs. Recent research efforts have aimed
to leverage the potential of heuristics, yet a unified formulation
accommodating both local and global heuristics remains undiscovered. Drawing insights from the fact that both local and global
heuristics can be represented by adjacency matrix multiplications,
we propose a unified matrix formulation to accommodate and generalize various heuristics. We further propose the Heuristic Learning
Graph Neural Network (HL-GNN) to efficiently implement the formulation. HL-GNN adopts intra-layer propagation and inter-layer
connections, allowing it to reach a depth of around 20 layers with
lower time complexity than GCN. Extensive experiments on the
Planetoid, Amazon, and OGB datasets underscore the effectiveness and efficiency of HL-GNN. It outperforms existing methods
by a large margin in prediction performance. Additionally, HLGNN is several orders of magnitude faster than heuristic-inspired
methods while requiring only a few trainable parameters. The
case study further demonstrates that the generalized heuristics and
learned weights are highly interpretable.

## Development Commands
### Running Experiments
- **Main English experiments (HART dataset)**: `./main.sh` (Note: Currently misconfigured to run RAID experiments)
- **Alternative detector experiments**: `./main2.sh`  
- **Multilingual experiments (5 languages)**: `./langs.sh`
- **RAID dataset experiments**: `./raid.sh`
- **Wavelet-enhanced experiments**: `./test.sh`