# kbeta_transformer2d/io_utils.py
"""Filesystem helpers – zero ML logic, pure I/O utilities."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any, Tuple

from .utils import compare_dict_states, compare_list_states

__all__ = [
    "setup_save_directories",
    "setup_load_directories",
    "save_model_and_optimizer",
    "load_model_and_optimizer",
]

# ---------------------------------------------------------------------------
# 1) directory helpers
# ---------------------------------------------------------------------------
def _ensure_dir(p: Path) -> Path:
    """Create *p* (and parents) if it does not yet exist, then return *p*."""
    p.mkdir(parents=True, exist_ok=True)
    return p


def _root(out_root: str | Path | None = None) -> Path:
    """
    Resolve the root folder used for *all* artefacts.

    Priority  1. explicit ``out_root`` argument  
              2. ``$KTRANS_OUT`` env‑var (handy for cluster jobs)  
              3. current working directory.
    """
    if out_root is not None:
        return Path(out_root).expanduser().resolve()
    from os import getenv

    return Path(getenv("KTRANS_OUT", Path.cwd())).expanduser().resolve()


# main public helpers -------------------------------------------------------
def setup_save_directories(
    run_name: str,
    restart_epoch: int | None = None,
    *,
    out_root: str | Path | None = None,
) -> Tuple[Path, Path, Path, Path]:
    """
    Create/return four folders:

    * model checkpoints
    * saved NPZ datasets
    * individual frame plots
    * per‑timestep inference MSE

    They all live under ``<root>/OUTPUTS`` where *root* defaults to CWD.
    """
    if restart_epoch is not None:
        run_name = f"{run_name}_restart_epoch_{restart_epoch}"

    base = _ensure_dir(_root(out_root) / "OUTPUTS")
    save_dir      = _ensure_dir(base / f"Transformer_save_BeyondL_{run_name}")
    dataset_dir   = _ensure_dir(base / f"Datasets_save_BeyondL_{run_name}")
    frameplot_dir = _ensure_dir(base / f"Heatmaps_BeyondL_{run_name}")
    mse_dir       = _ensure_dir(base / f"InferenceMSE_BeyondL_{run_name}")

    return save_dir, dataset_dir, frameplot_dir, mse_dir


def setup_load_directories(
    run_name: str,
    checkpoint_epoch: int,
    *,
    out_root: str | Path | None = None,
) -> Tuple[Path, Path]:
    """Locate the two folders created by :pyfunc:`setup_save_directories`."""
    base = _root(out_root) / "OUTPUTS"
    load_dir    = base / f"Transformer_save_BeyondL_{run_name}"
    dataset_dir = base / f"Datasets_save_BeyondL_{run_name}"

    for p in (load_dir, dataset_dir):
        if not p.is_dir():
            raise FileNotFoundError(
                f"{p} does not exist – did you specify the correct run‑name?"
            )
    return load_dir, dataset_dir


# ---------------------------------------------------------------------------
# 2) (de)serialization helpers
# ---------------------------------------------------------------------------
def save_model_and_optimizer(
    model: Any,
    optimizer: Any,
    mx_random_state: Any,
    config: dict[str, Any],
    current_epoch: int,
    dir_path: Path,
    model_base_file_name: str,
    optimizer_base_file_name: str,
    hyper_base_file_name: str,
) -> None:
    """Persist weights, optimiser state, RNG & hyper‑params for *current_epoch*."""
    dir_path = _ensure_dir(Path(dir_path))

    model_file      = dir_path / f"{model_base_file_name}_epoch_{current_epoch}.pkl"
    weights_file    = dir_path / f"{model_base_file_name}_weights_epoch_{current_epoch}.safetensors"
    optimizer_file  = dir_path / f"{optimizer_base_file_name}_epoch_{current_epoch}.pkl"
    random_file     = dir_path / f"random_state_epoch_{current_epoch}.pkl"
    config_file     = dir_path / f"{hyper_base_file_name}_epoch_{current_epoch}.json"

    # --- write ----------------------------------------------------------------
    with model_file.open("wb") as fh:
        pickle.dump(model.parameters(), fh)
    model.save_weights(weights_file)

    with optimizer_file.open("wb") as fh:
        pickle.dump(optimizer.state, fh)

    with random_file.open("wb") as fh:
        pickle.dump(mx_random_state, fh)

    cfg = dict(config, current_epoch=current_epoch)
    config_file.write_text(json.dumps(cfg, indent=4))

    print(f"✅ Saved checkpoint #{current_epoch} to {dir_path.relative_to(_root())}")


def load_model_and_optimizer(
    model: Any,
    optimizer: Any,
    mx_random_state: Any,
    dir_path: Path,
    model_base_file_name: str,
    optimizer_base_file_name: str,
    hyper_base_file_name: str,
    checkpoint_epoch: int,
    *,
    comparison: bool = True,
) -> Tuple[int, dict[str, Any], Any, dict[str, Any], dict[str, Any]]:
    """Inverse of :pyfunc:`save_model_and_optimizer` (see docstring there)."""
    dir_path = Path(dir_path)

    model_file     = dir_path / f"{model_base_file_name}_epoch_{checkpoint_epoch}.pkl"
    weights_file   = dir_path / f"{model_base_file_name}_weights_epoch_{checkpoint_epoch}.safetensors"
    optimizer_file = dir_path / f"{optimizer_base_file_name}_epoch_{checkpoint_epoch}.pkl"
    random_file    = dir_path / f"random_state_epoch_{checkpoint_epoch}.pkl"
    config_file    = dir_path / f"{hyper_base_file_name}_epoch_{checkpoint_epoch}.json"

    # --- existence check ------------------------------------------------------
    mandatory = [model_file, optimizer_file, random_file, config_file, weights_file]
    if not all(f.exists() for f in mandatory):
        raise FileNotFoundError(
            "⛔ Some checkpoint artefacts are missing:\n"
            + "\n".join(str(p) for p in mandatory if not p.exists())
        )

    # --- load -----------------------------------------------------------------
    with model_file.open("rb") as fh:
        loaded_parameters = pickle.load(fh)

    with optimizer_file.open("rb") as fh:
        loaded_optimizer_state = pickle.load(fh)

    with random_file.open("rb") as fh:
        loaded_random_state = pickle.load(fh)

    loaded_config: dict[str, Any] = json.loads(config_file.read_text())
    start_epoch = int(loaded_config.get("current_epoch", 0))

    # --- optional consistency checks -----------------------------------------
    if comparison:
        if compare_dict_states(optimizer.state, loaded_optimizer_state, "optimizer"):
            print("✅ Optimizer state matches stored checkpoint.")
        if compare_list_states(mx_random_state, loaded_random_state, "random"):
            print("✅ Random state matches stored checkpoint.")
        if compare_dict_states(model.parameters(), loaded_parameters, "model"):
            print("✅ Model parameters match stored checkpoint.")

    return (
        start_epoch,
        loaded_optimizer_state,
        loaded_random_state,
        loaded_parameters,
        loaded_config,
    )
