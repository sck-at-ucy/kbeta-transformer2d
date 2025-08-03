# kbeta‑transformer2d – *2‑D Heat‑Diffusion Transformer trained with Kourkoutas‑β*  🌞🦎🚀📈

[![CI (macOS arm64)](https://github.com/sck-at-ucy/kbeta-transformer2d/actions/workflows/ci.yml/badge.svg)](https://github.com/sck-at-ucy/kbeta-transformer2d/actions/workflows/ci.yml)

> **Research companion code for the upcoming paper> “Kourkoutas‑β – Soft‑max Momentum with Adaptive Variance for Mesh‑Accelerated Deep Learning.”**> > This repository contains the full **2‑D data‑driven Transformer** workload that accompanies the optimiser > (see the separate [`kbeta`](https://github.com/sck-at-ucy/kbeta) repo), plus lightweight utilities for training, > evaluation and visualisation.

---

## Table of Contents
1. [Why a 2‑D Transformer?](#why-a-2d-transformer)
2. [Project layout](#project-layout)
3. [Quick start](#quick-start)
4. [Training from scratch](#training-from-scratch)
5. [Using your own datasets](#using-your-own-datasets)
6. [Tests & linting](#tests--linting)
7. [Relation to Kourkoutas‑β](#relation-to-kourkoutas-β)
8. [Citation](#citation)
9. [License](#license)

---

## Why a 2‑D Transformer?

* **Spatial‑temporal diffusion** appears in countless engineering problems (heat flow, pollutant transport, …).  
* A *purely data‑driven* Transformer offers a clean stress‑test for the optimiser—no PDE loss terms, no hand‑tuned schedulers.  
* The model scales to **512 × 512 meshes on Apple Silicon** while remaining <2 M parameters; perfect for rapid experimentation.

---

## Project layout

```
kbeta-transformer2d
├── src/kbeta_transformer2d/
│   ├── __init__.py          # public API
│   ├── data.py              # mesh generation + loaders
│   ├── model.py             # Transformer & loss
│   ├── optim_factory.py     # Kourkoutas‑β wiring
│   ├── train.py             # training / eval loops
│   ├── plot_utils.py        # visualisations
│   └── demo_heat2d.py       # CLI entry‑point
├── configs/
│   └── heat2d.yml           # default hyper‑params
├── tests/                   # smoke tests
└── README.md                # you are here
```

---

## Quick start

```bash
# 1) clone & set up a fresh virtualenv (Apple Silicon, Python 3.11)
git clone https://github.com/sck-at-ucy/kbeta-transformer2d.git
cd kbeta-transformer2d
python -m venv .venv && source .venv/bin/activate

# 2) install this package *and* the private optimiser
pip install -e ".[dev]"                    # installs mlx, ruff, pytest, …
pip install "kbeta @ git+https://github.com/sck-at-ucy/kbeta.git@main"

# 3) verify everything works
pytest -q                                  # ➜ 2 tests should pass
```

---

## Training from scratch

```bash
python -m kbeta_transformer2d.demo_heat2d         configs/heat2d.yml                         --override model_params.epochs=30
```

The YAML file controls **mesh size, model depth, batch‑size, optimiser settings, plotting**, etc.  
Any key can be overridden on the CLI via `--override key.subkey=value`.

### Collected artefacts

* checkpoints every *n* epochs (`.npz`)
* optimiser state for restartability
* `.json` run config
* optional GIFs of predicted vs truth heat maps

---

## Using your own datasets

1. Provide a NumPy array shaped `[t, ny, nx]` with your temperature fields.  
2. Adjust `geometry.nx, geometry.ny` in `configs/heat2d.yml`.  
3. Replace the `generate_datasets()` stub in **data.py** with your loader.

That’s it—no code changes in the model or training loop are required.

---

## Tests & linting

```bash
pytest                 # unit + smoke tests
ruff check .           # style / quality gate
mypy src               # (optional) static typing
```

Our GitHub Action (macOS‑14, MLX back‑end) blocks any PR that fails the above.

---

## Relation to Kourkoutas‑β

`kbeta_transformer2d` **does not re‑implement the optimiser**; it *consumes* it:  
`optim_factory.py` instantiates `KourkoutasSoftmaxFlex` from the `kbeta` package and demonstrates adaptive‑β₂ in a challenging, mesh‑based setting.  
If you only need the optimiser, install **`kbeta`**.  If you want a ready‑made workload to benchmark against Adam, install **this** repo.

---

## Citation

```
@misc{Kassinos2025Transformer2D,
  title        = {Data‑Driven 2‑D Heat‑Diffusion Transformer – Companion Code},
  author       = {Stavros Kassinos and others},
  howpublished = {GitHub},
  year         = {2025},
  note         = {https://github.com/sck-at-ucy/kbeta-transformer2d}
}
```

Please also cite the main *Kourkoutas‑β* paper.

---

## License

Released under the **MIT License**.  See [`LICENSE`](LICENSE) for the full text.

Happy experimenting — and may your gradients be sunny 🌞🦎🚀📈

