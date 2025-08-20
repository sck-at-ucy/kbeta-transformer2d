[![Test (dev install)](https://github.com/sck-at-ucy/kbeta-transformer2d/actions/workflows/ci.yml/badge.svg?branch=main&job=test-dev)](...)
[![Test (wheel install)](https://github.com/sck-at-ucy/kbeta-transformer2d/actions/workflows/ci.yml/badge.svg?branch=main&job=test-wheel)](...)

<p align="center">
  <img src="assets/MLX_Kourkoutas.png" width="300"/>
  <img src="assets/t_2dframes.png" width="300"/>
</p>

# kbeta-transformer2d – *2-D Heat-Diffusion Transformer trained with Kourkoutas-β*  🌞🦎🚀📈

> **Research companion code for the upcoming paper**  
> “Kourkoutas‑β – Soft‑max Momentum with Adaptive Variance for Mesh‑Accelerated Deep Learning.”  
> Published as [arXiv:2508.12996](http://arxiv.org/abs/2508.12996).
>
> This repository contains the full **2‑D data‑driven Transformer** workload that accompanies the optimiser  
> (see the separate [`kbeta`](https://github.com/sck-at-ucy/kbeta) repo), plus lightweight utilities for training,  
> evaluation and visualisation.

---

## Table of Contents
1. [Why a 2‑D Transformer?](#why-a-2d-transformer)
2. [Model highlights](#model-highlights)
3. [Project layout](#project-layout)
4. [Installation](#installation)
5. [Quick start](#quick-start)
6. [Command‑line interface (CLI)](#command-line-interface-cli)
7. [Training from scratch](#training-from-scratch)
8. [Using your own datasets](#using-your-own-datasets)
9. [Tests & linting](#tests--linting)
10. [Relation to Kourkoutas‑β](#relation-to-kourkoutas-β)
11. [Citation](#citation)
12. [License](#license)

---

## Why a 2‑D Transformer?
* **Spatial‑temporal diffusion** appears in countless engineering problems (heat flow, pollutant transport, …).  
* A *purely data‑driven* Transformer offers a clean stress‑test for the optimiser.  
* Solver‑free physics loss: we embed the heat‑equation residual as an analytic term, no back‑prop through external PDE solvers is required.  
* The model scales to **512 × 512 meshes on Apple Silicon** while remaining <2 M parameters; perfect for rapid experimentation.

---

## Model highlights
*(what’s special about HeatDiffusion‑Transformer‑2D)*

* **Patch‑wise attention on 2‑D grids**  
  The input tensor is reshaped into *(T × H × W)* patches, letting the model treat every spatial location symmetrically while still exploiting MX‑GPU tensor cores efficiently.

* **Dual masking modes**  
  *Causal* masks give an autoregressive model useful for long‑horizon rollout tests; *block* masks allow full‑context training when future frames are available.

* **RoPE (Rotary Positional Encoding) in the time dimension**  
  A single line swap lets you switch between vanilla sinusoidal encodings and RoPE, which markedly improves extrapolation beyond the training window.

* **Activation quantisation ready**  
  All dense / conv projections are implemented with `mlx.nn.quantize_lin`, giving you 8‑bit weights on Apple Silicon **without** code changes.

* **Tiny footprint – 2.3 M parameters**  
  Fits comfortably on a single M‑series GPU core at batch‑size 32, even in FP16.

* **One‑liner optimiser swap**  
  The model inherits its optimiser object, so comparing Adam vs Kourkoutas‑β is literally *one* YAML entry.

---

## Project layout
```text
kbeta-transformer2d
├── src/kbeta_transformer2d/
│   ├── data.py              # mesh generation + loaders
│   ├── model.py             # Transformer & loss
│   ├── optim_factory.py     # Kourkoutas‑β wiring
│   ├── train.py             # training / eval loops
│   ├── plot_utils.py        # visualisations
│   └── demo_heat2d.py       # CLI entry‑point
├── configs/
│   └── heat2d.yml           # default hyper‑params
│   └── paper.yml            # paper   hyper‑params
└── README.md                # you are here
```

---

## Installation

### Option 1: PyPI wheels (end‑users)
If you only want to run the Transformer benchmark with the latest `kbeta`:

```bash
pip install kbeta-transformer2d
```

For dev tools and tests:

```bash
pip install "kbeta-transformer2d[dev]"
```

For exact reproducibility of the paper (MLX 0.26.3 + pinned deps):

```bash
pip install "kbeta-transformer2d[repro]"
```

---

### Option 2: Cloning the repo (researchers / contributors)

```bash
git clone https://github.com/sck-at-ucy/kbeta-transformer2d.git
cd kbeta-transformer2d
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

This makes all configs and scripts editable for research use.

---

## Quick start
```bash
pytest -q   # ➜ all smoke‑tests should pass
```

---
