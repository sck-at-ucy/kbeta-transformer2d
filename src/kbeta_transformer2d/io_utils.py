# kbeta_transformer2d/io_utils.py
"""Generic path helpers – no ML logic."""

from __future__ import annotations

from pathlib import Path
import json
import pickle
from typing import Tuple, Any

from .utils import compare_dict_states, compare_list_states

__all__ = [
    "setup_save_directories",
    "setup_load_directories",
    "save_model_and_optimizer",
    "load_model_and_optimizer",
]


# -----------------------------------------------------------------------------#
# 1) directory helpers                                                         #
# -----------------------------------------------------------------------------#
def _base_output_dir() -> Path:
    """Resolve the *runtime* output directory (cwd/OUTPUTS)."""
    return Path.cwd() / "OUTPUTS"


def setup_save_directories(
    run_name: str, restart_epoch: int | None = None
) -> Tuple[Path, Path, Path, Path]:
    """
    Create (if needed) and return 4 sibling folders inside OUTPUTS/ :
      • Transformer weights & optimiser checkpoints
      • Datasets
      • Heat‑map frames
      • Inference‑time MSE plots
    """
    out = _base_output_dir()
    out.mkdir(exist_ok=True)

    if restart_epoch is not None:
        run_name = f"{run_name}_restart_epoch_{restart_epoch}"

    save_dir        = out / f"Transformer_save_BeyondL_{run_name}"
    dataset_dir     = out / f"Datasets_save_BeyondL_{run_name}"
    heatmaps_dir    = out / f"Heatmaps_BeyondL_{run_name}"
    inference_dir   = out / f"InferenceMSE_BeyondL_{run_name}"

    for p in (save_dir, dataset_dir, heatmaps_dir, inference_dir):
        p.mkdir(parents=True, exist_ok=True)

    return save_dir, dataset_dir, heatmaps_dir, inference_dir


def setup_load_directories(run_name: str, checkpoint_epoch: int) -> Tuple[Path, Path]:
    """
    Resolve folders created by `setup_save_directories` and sanity‑check they exist.
    """
    out          = _base_output_dir()
    save_dir     = out / f"Transformer_save_BeyondL_{run_name}"
    dataset_dir  = out / f"Datasets_save_BeyondL_{run_name}"

    if not save_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint directory {save_dir} not found.")
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory {dataset_dir} not found.")

    return save_dir, dataset_dir


# -----------------------------------------------------------------------------#
# 2) checkpoint helpers                                                        #
# -----------------------------------------------------------------------------#
def save_model_and_optimizer(
    model: Any,
    optimizer: Any,
    mx_random_state: Any,
    config: dict,
    current_epoch: int,
    dir_path: Path,
    model_base_file_name: str,
    optimizer_base_file_name: str,
    hyper_base_file_name: str,
) -> None:
    """
    Persist model parameters, optimiser state, RNG state and full config.
    """
    dir_path.mkdir(parents=True, exist_ok=True)

    model_file      = dir_path / f"{model_base_file_name}_epoch_{current_epoch}.pkl"
    weights_file    = dir_path / f"{model_base_file_name}_weights_epoch_{current_epoch}.safetensors"
    optimizer_file  = dir_path / f"{optimizer_base_file_name}_epoch_{current_epoch}.pkl"
    rng_file        = dir_path / f"random_state_epoch_{current_epoch}.pkl"
    cfg_file        = dir_path / f"{hyper_base_file_name}_epoch_{current_epoch}.json"

    # ---- dump ----------------------------------------------------------------
    with model_file.open("wb") as fh:
        pickle.dump(model.parameters(), fh)

    model.save_weights(str(weights_file))          # MLX wants str, not Path

    with optimizer_file.open("wb") as fh:
        pickle.dump(optimizer.state, fh)

    with rng_file.open("wb") as fh:
        pickle.dump(mx_random_state, fh)

    config["current_epoch"] = current_epoch
    with cfg_file.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=4)

    print(f"[io_utils] checkpoint written -> {dir_path}")


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
) -> tuple[int, dict, Any, dict, dict]:
    """
    Restore everything saved by `save_model_and_optimizer`.
    Returns (start_epoch, opt_state, rng_state, parameters, loaded_cfg)
    """
    model_file     = dir_path / f"{model_base_file_name}_epoch_{checkpoint_epoch}.pkl"
    weights_file   = dir_path / f"{model_base_file_name}_weights_epoch_{checkpoint_epoch}.safetensors"
    optimizer_file = dir_path / f"{optimizer_base_file_name}_epoch_{checkpoint_epoch}.pkl"
    rng_file       = dir_path / f"random_state_epoch_{checkpoint_epoch}.pkl"
    cfg_file       = dir_path / f"{hyper_base_file_name}_epoch_{checkpoint_epoch}.json"

    if not all(p.exists() for p in (model_file, weights_file, optimizer_file, rng_file, cfg_file)):
        print(f"[io_utils] no full checkpoint for epoch {checkpoint_epoch}; starting fresh.")
        return 0, {}, mx_random_state, {}, {}

    with model_file.open("rb") as fh:
        loaded_parameters = pickle.load(fh)

    with optimizer_file.open("rb") as fh:
        loaded_optimizer_state = pickle.load(fh)

    with rng_file.open("rb") as fh:
        loaded_random_state = pickle.load(fh)

    with cfg_file.open(encoding="utf-8") as fh:
        loaded_config = json.load(fh)

    start_epoch = loaded_config.get("current_epoch", 0)

    # ---- optional consistency checks ----------------------------------------
    if comparison:
        if compare_dict_states(optimizer.state, loaded_optimizer_state, "optimizer"):
            print("✓ optimiser state matches")
        if compare_list_states(mx_random_state, loaded_random_state, "random state"):
            print("✓ RNG state matches")
        if compare_dict_states(model.parameters(), loaded_parameters, "model params"):
            print("✓ model parameters match")

    return (
        start_epoch,
        loaded_optimizer_state,
        loaded_random_state,
        loaded_parameters,
        loaded_config,
    )
