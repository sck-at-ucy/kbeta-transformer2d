# ────────────────────────────────────────────────────────────────────────────
# io_utils.py  –  pure filesystem helpers (no ML code)
# ────────────────────────────────────────────────────────────────────────────
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
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
        "save": base_dir / f"Transformer_save_BeyondL_{run_name}",
        "datasets": base_dir / f"Datasets_save_BeyondL_{run_name}",
        "heatmaps": base_dir / f"Heatmaps_BeyondL_{run_name}",
        "mse": base_dir / f"InferenceMSE_BeyondL_{run_name}",
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

    model_file_path = dir_path / f"{model_base_file_name}_epoch_{current_epoch}.pkl"
    weights_file_path = (
        dir_path / f"{model_base_file_name}_weights_epoch_{current_epoch}.safetensors"
    )
    optimizer_file_path = (
        dir_path / f"{optimizer_base_file_name}_epoch_{current_epoch}.pkl"
    )
    random_state_path = dir_path / f"random_state_epoch_{current_epoch}.pkl"
    config_file_path = dir_path / f"{hyper_base_file_name}_epoch_{current_epoch}.json"

    with model_file_path.open("wb") as fh:
        pickle.dump(model.parameters(), fh)
    model.save_weights(str(weights_file_path))  # MLX wants str

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
    dir_path: str | Path,
    model_base_file_name: str,
    optimizer_base_file_name: str,
    hyper_base_file_name: str,
    checkpoint_epoch: int,
    comparison: bool = True,
) -> tuple[int, dict[str, Any], Any, dict[str, Any], dict[str, Any]]:
    """
    Reload model parameters, optimiser state, RNG state and config
    from a given *checkpoint_epoch* stored under *dir_path*.

    Returns
    -------
    start_epoch          : int
    loaded_optimizer_state : dict
    loaded_random_state    : mx.random state‑object
    loaded_parameters      : dict  (model parameters)
    loaded_config          : dict  (training config at save‑time)
    """

    # ------------------------------------------------------------------ #
    # 1) construct paths with pathlib                                    #
    # ------------------------------------------------------------------ #
    dir_path = Path(dir_path).expanduser()

    model_file = dir_path / f"{model_base_file_name}_epoch_{checkpoint_epoch}.pkl"
    weights_file = (
        dir_path
        / f"{model_base_file_name}_weights_epoch_{checkpoint_epoch}.safetensors"
    )
    optimizer_file = (
        dir_path / f"{optimizer_base_file_name}_epoch_{checkpoint_epoch}.pkl"
    )
    random_state_p = dir_path / f"random_state_epoch_{checkpoint_epoch}.pkl"
    config_file = dir_path / f"{hyper_base_file_name}_epoch_{checkpoint_epoch}.json"

    # ------------------------------------------------------------------ #
    # 2) existence check                                                 #
    # ------------------------------------------------------------------ #
    required = [model_file, weights_file, optimizer_file, random_state_p, config_file]
    if not all(p.exists() for p in required):
        print(
            f"[load_model_and_optimizer] no checkpoint found at epoch {checkpoint_epoch} – starting fresh."
        )
        return 0, {}, mx_random_state, {}, {}

    # ------------------------------------------------------------------ #
    # 3) load artefacts                                                  #
    # ------------------------------------------------------------------ #
    with model_file.open("rb") as fh:
        loaded_parameters = pickle.load(fh)

    with optimizer_file.open("rb") as fh:
        loaded_optimizer_state = pickle.load(fh)

    with random_state_p.open("rb") as fh:
        loaded_random_state = pickle.load(fh)

    with config_file.open("r", encoding="utf‑8") as fh:
        loaded_config = json.load(fh)

    start_epoch = int(loaded_config.get("current_epoch", 0))

    # ------------------------------------------------------------------ #
    # 4) (optional) compare with current in‑memory states                #
    # ------------------------------------------------------------------ #
    if comparison:
        print("[load_model_and_optimizer] comparing checkpoint vs. in‑memory objects …")
        if compare_dict_states(optimizer.state, loaded_optimizer_state, "optimizer"):
            print("   ✓ optimiser state matches")
        else:
            print("   ✗ optimiser state differs")

        if compare_list_states(mx_random_state, loaded_random_state, "random state"):
            print("   ✓ MX random state matches")
        else:
            print("   ✗ MX random state differs")

        if compare_dict_states(model.parameters(), loaded_parameters, "model params"):
            print("   ✓ model parameters match")
        else:
            print("   ✗ model parameters differ")

    print(
        f"[load_model_and_optimizer] checkpoint epoch {checkpoint_epoch} loaded from {dir_path}"
    )

    return (
        start_epoch,
        loaded_optimizer_state,
        loaded_random_state,
        loaded_parameters,
        loaded_config,
    )
