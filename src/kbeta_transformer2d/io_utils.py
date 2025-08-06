# ────────────────────────────────────────────────────────────────────────────
# io_utils.py  –  pure filesystem helpers (no ML code)
# ────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

from pathlib import Path
import json
import os
import pickle
from typing import Any

from .utils import compare_dict_states, compare_list_states

__all__ = [
    "setup_save_directories",
    "setup_load_directories",
    "save_model_and_optimizer",
    "load_model_and_optimizer",
]

# --------------------------------------------------------------------------- #
# 1) path helpers                                                             #
# --------------------------------------------------------------------------- #
def _normalise_dir(path: str | os.PathLike | None) -> Path:
    """Expand ~, resolve relative paths and create the directory if needed."""
    if path is None:
        # fall back to `<package‑root>/OUTPUTS`
        path = Path(__file__).resolve().parent / "OUTPUTS"
    else:
        path = Path(path).expanduser().resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_save_directories(
    run_name: str,
    restart_epoch: int | None = None,
    *,
    base_dir: str | os.PathLike | None = None,
) -> tuple[Path, Path, Path, Path]:
    """
    Create (or reuse) the four run‑specific output folders **below *base_dir***.

    Parameters
    ----------
    run_name      – slug that identifies this experiment
    restart_epoch – None for a fresh run, otherwise appended to the folder name
    base_dir      – root folder for all outputs
                    · not given      ⇒  <package>/OUTPUTS/
                    · relative path  ⇒  resolved under the *current working dir*
                    · absolute path  ⇒  taken as‑is
    """
    base_dir = _normalise_dir(base_dir)

    if restart_epoch is not None:
        run_name = f"{run_name}_restart_epoch_{restart_epoch}"

    paths = {
        "save":      base_dir / f"Transformer_save_BeyondL_{run_name}",
        "datasets":  base_dir / f"Datasets_save_BeyondL_{run_name}",
        "heatmaps":  base_dir / f"Heatmaps_BeyondL_{run_name}",
        "mse":       base_dir / f"InferenceMSE_BeyondL_{run_name}",
    }
    for p in paths.values():
        p.mkdir(parents=True, exist_ok=True)

    return paths["save"], paths["datasets"], paths["heatmaps"], paths["mse"]


def setup_load_directories(
    run_name: str,
    checkpoint_epoch: int,
    *,
    base_dir: str | os.PathLike | None = None,
) -> tuple[Path, Path]:
    """
    Return the two folders created by `setup_save_directories` and verify they exist.
    """
    base_dir = _normalise_dir(base_dir)
    save_dir = base_dir / f"Transformer_save_BeyondL_{run_name}"
    data_dir = base_dir / f"Datasets_save_BeyondL_{run_name}"

    if not save_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory {save_dir!s} not found")
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory {data_dir!s} not found")

    return save_dir, data_dir


# --------------------------------------------------------------------------- #
# 2) (de‑)serialisation helpers                                               #
# --------------------------------------------------------------------------- #
def save_model_and_optimizer(
    model,
    optimizer,
    mx_random_state,
    config: dict[str, Any],
    current_epoch: int,
    dir_path: Path,
    model_base_file_name: str,
    optimizer_base_file_name: str,
    hyper_base_file_name: str,
) -> None:
    """Dump model parameters, weights, optimizer state, RNG state and config."""
    dir_path.mkdir(parents=True, exist_ok=True)

    model_file_path     = dir_path / f"{model_base_file_name}_epoch_{current_epoch}.pkl"
    weights_file_path   = dir_path / f"{model_base_file_name}_weights_epoch_{current_epoch}.safetensors"
    optimizer_file_path = dir_path / f"{optimizer_base_file_name}_epoch_{current_epoch}.pkl"
    random_state_path   = dir_path / f"random_state_epoch_{current_epoch}.pkl"
    config_file_path    = dir_path / f"{hyper_base_file_name}_epoch_{current_epoch}.json"

    with model_file_path.open("wb") as fh:
        pickle.dump(model.parameters(), fh)
    model.save_weights(str(weights_file_path))          # MLX wants str

    with optimizer_file_path.open("wb") as fh:
        pickle.dump(optimizer.state, fh)

    with random_state_path.open("wb") as fh:
        pickle.dump(mx_random_state, fh)

    cfg_copy = dict(config, current_epoch=current_epoch)
    with config_file_path.open("w", encoding="utf-8") as fh:
        json.dump(cfg_copy, fh, indent=4)

    print(f"[io_utils] checkpoint written -> {dir_path}")


def load_model_and_optimizer(
    model,
    optimizer,
    mx_random_state,
    dir_path: Path,
    model_base_file_name: str,
    optimizer_base_file_name: str,
    hyper_base_file_name: str,
    checkpoint_epoch: int,
    comparison: bool = True,
) -> tuple[int, dict, Any, dict, dict]:
    """(unchanged – only minor refactor to Path API)"""
    # … keep your existing implementation, but build the *Path* objects via
    #   dir_path / f"...", then use .open() or str(path) where MLX insists.

    # (omitted here for brevity – paste your current body)
    ...
