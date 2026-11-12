# Geometric Attractor Steering in Non-Equilibrium Metabolic-Immune Networks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Repository](https://img.shields.io/badge/GitHub-sukuyoshi%2Fmetabolic__immune__dynamics-blue)](https://github.com/sukuyoshi/metabolic_immune_dynamics)

Unified dissipative systems framework for Type 1 and Type 2 diabetes remission, early warning screening, and predictive autonomous closed-loop control.

Companion codebase to the research manuscript:
> **Geometric Attractor Steering in Non-Equilibrium Metabolic-Immune Networks: A Unified Dissipative Systems Framework for Type 1 and Type 2 Diabetes Remission and Stabilization**  
> *Reiji Yoshikai (2026)*. Source: [https://github.com/sukuyoshi/metabolic_immune_dynamics](https://github.com/sukuyoshi/metabolic_immune_dynamics)

## Overview

This repository provides an open-source (MIT Licensed) implementation of a multi-scale dissipative systems framework integrating:
1. **Bistable Metabolic Manifold**: Modeling the non-linear coupled dynamics of glucose, insulin, and functional, dedifferentiated, and apoptotic beta-cell pools.
2. **Least Action Pulse Steering**: Solving optimal finite-duration interventions (caloric restriction, incretin pulses) based on Willems dissipative inequalities to steer states across the separatric saddle barrier into the basin of attraction of healthy homeostasis (Type 2 diabetes remission).
3. **Sturm-Liouville Gap Softening for Type 1 Diabetes**: Autonomous identification of spectral gap softening in pre-symptomatic autoimmune repertoire dynamics months to years prior to clinical hyperglycemia.
4. **Raychaudhuri Predictive Closed-Loop Control**: Geometry-driven early divergence diagnostics for artificial pancreas systems, anticipating acute postprandial and exercise spikes tens of minutes in advance.

## Architecture

- `src/metabolic_immune_dynamics.py`: Coupled non-linear differential equations governing metabolic-endocrine and autoimmune state variables.
- `simulate_t2d_remission.py`: Numerical phase-portrait simulation of successful versus failed attractor escape trajectories.
- `simulate_t1d_early_warning.py`: Pre-symptomatic Koopman generator identification and Sturm-Liouville gap softening analysis.
- `simulate_artificial_pancreas_raychaudhuri.py`: Comparison of geometry-informed predictive insulin infusion against standard PID/MPC loops.
- `results/`: Numerical benchmark CSVs and high-resolution trajectory visualizations.

## License

MIT License. Copyright 2026 Reiji Yoshikai.
