# transformer/Testing_Kourkoutasb.py
"""
Testing_Kourkoutasb.py – refactored for modular CLI use
-------------------------------------------------------

• No top‑level side‑effects except for imports.
• CLI parsing isolated in _parse_cli().
• YAML‑file + CLI flags merged in build_config().
• Former main‑block lives in run_from_config(cfg).
• A thin main() keeps the file executable *and* importable.
"""

from __future__ import annotations

import sys

print(">>> entering Testing_Kourkoutasb_tmp, argv:", sys.argv)

# ── standard lib ───────────────────────────────────────────────────────────
import argparse
import json
import os
import socket
import sys
from pathlib import Path

# ── local project imports (leave unchanged) ───────────────────────────────
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
import yaml

if TYPE_CHECKING:             # only seen by static analyzers
    model:  HeatDiffusionModel
    optimizer: optim.Optimizer
    train_step: object
    evaluate_step: object
    state: object


# -------------------------------------------------------------------------
# Globals populated later in main()/run_from_config()
# -------------------------------------------------------------------------
ARGS: Optional[argparse.Namespace] = None  # CLI flags (kept for legacy use)
config: Optional[Dict[str, Any]] = None  # unified YAML + CLI configuration


# =========================================================================
# 1) YAML helpers
# =========================================================================
def _load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r") as fh:
        return yaml.safe_load(fh)


def _apply_overrides(cfg: Dict[str, Any], kv_pairs: List[str]) -> None:
    """
    Apply `KEY=VALUE` overrides given on the command line.
    Nested keys use dot‑notation, e.g. ``--override model_params.embed_dim=256``.
    """
    for pair in kv_pairs:
        if "=" not in pair:
            raise ValueError(f"--override expects KEY=VAL, got {pair!r}")
        key, val = pair.split("=", 1)
        try:  # interpret numbers / lists / bools
            val = json.loads(val)
        except Exception:
            pass
        sub = cfg
        for k in key.split(".")[:-1]:
            sub = sub.setdefault(k, {})
        sub[key.split(".")[-1]] = val


# =========================================================================
# 2) CLI compatible with the previous 3‑D PINN script
# =========================================================================
def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m transformer.Testing_Kourkoutasb",
        description="2‑D heat‑diffusion Transformer (MX‑GPU)",
    )
    p.add_argument(
        "config", metavar="YAML", type=str, help="Path to YAML configuration file"
    )
    # ----- historical flags copied from the PINN code -------------------
    p.add_argument(
        "--optimizer",
        choices=["adam95", "adam999", "kourkoutas"],
        default=None,
        help="Optimiser to use (overrides YAML)",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides YAML)",
    )
    p.add_argument(
        "--seed", type=int, default=None, help="Random seed (overrides YAML)"
    )
    p.add_argument(
        "--viz", action="store_true", help="Run optional visualisation at the end"
    )
    p.add_argument(
        "--kour_diagnostics",
        action="store_true",
        help="Enable lightweight diagnostics in KourkoutasSoftmaxFlex",
    )
    p.add_argument(
        "--collect_spikes",
        action="store_true",
        help="Track sun‑spike / β₂ distributions during training",
    )
    # ----- generic KEY=VAL overrides ------------------------------------
    p.add_argument(
        "--override",
        nargs="*",
        default=[],
        metavar="KEY=VAL",
        help="Override arbitrary YAML entries; dot‑notation allowed",
    )
    return p.parse_args()


def build_config() -> Dict[str, Any]:
    """
    Merge YAML file, legacy PINN‑style flags and ``--override`` KV pairs
    into a single configuration dictionary.
    """
    global ARGS
    ARGS = _parse_cli()

    cfg = _load_yaml(ARGS.config)

    # ---- simple scalar overrides --------------------------------------
    if ARGS.seed is not None:
        cfg["seed"] = ARGS.seed
    if ARGS.epochs is not None:
        cfg["model_params"]["epochs"] = ARGS.epochs
    if ARGS.optimizer is not None:
        cfg.setdefault("optimizer", {})["name"] = ARGS.optimizer

    # ---- boolean feature‑toggles --------------------------------------
    if ARGS.kour_diagnostics:
        cfg.setdefault("optimizer", {})["kour_diagnostics"] = True
    if ARGS.collect_spikes:
        cfg.setdefault("tracking", {})["collect_spikes"] = True
    if ARGS.viz:
        cfg.setdefault("viz", {})["enabled"] = True

    # ---- arbitrary KEY=VAL overrides ----------------------------------
    _apply_overrides(cfg, ARGS.override)
    return cfg


# =========================================================================
# 3) Import all required modules from transformer
# =========================================================================


from .data import (
    data_loader_2D,
    generate_datasets,
    initialize_geometry_and_bcs,
    load_datasets,
    save_datasets,
)
from .io_utils import (
    load_model_and_optimizer,
    save_model_and_optimizer,
    setup_load_directories,
    setup_save_directories,
)
from .model import loss_fn_2D
from .optim_factory import initialize_model_and_optimizer
from .plot_utils import (
    plot_mse_evolution,
    plot_predictions_2D,
    plot_regressive_predictions_2D,
    save_distribution_density_heatmap,
    save_distribution_violin_plot,
)

# from .train import (train_and_validate, evaluate_model,evaluate_model_block_sequence)
from .train import (
    evaluate_model,
    evaluate_model_block_sequence,
    evaluate_self_regressive_model_BeyondL,  # todo: <----- Need to fix so that it does need matplotlib in train.py
    make_train_and_eval_steps,  # <- we’ll create these two helpers in train.py
    train_and_validate,
)
from .utils import (
    compare_datasets,
    compare_dict_states,
    compare_list_states,
    convert_lists_to_tuples,
    print_config_comparison,
    print_fresh_run_config,
)


# =========================================================================
#  4) “run_from_config” Director
# =========================================================================
def run_from_config(cfg: Dict[str, Any]) -> None:
    global config, ARGS, model, optimizer, state, train_step, evaluate_step
    config = cfg

    seed = config.get("seed", 30)
    np.random.seed(seed)
    mx.random.seed(seed)

    hostname = socket.gethostname()
    print("mx‑Random:", mx.random.state)
    print("Random state:", mx.random.state)
    print(
        f"Configuration: {config['boundary_segment_strategy']}:{config['model_params']['mask_type']}"
    )
    print(f"Hostname: {hostname}: DYLD_LIBRARY_PATH:{os.getenv('DYLD_LIBRARY_PATH')}")

    # Choose the save_interval based on save_checkpoints
    if config["save_checkpoints"]:
        config["save_interval"] = 10  # Set a reasonable interval for saving
    else:
        config["save_interval"] = (
            config["model_params"]["epochs"] + 1
        )  # Disable saving by setting it beyond the number of epochs

    # Set the run name as a parameter using Config parameters for identification of files/folders
    run_name = (
        str(config["run_label"])
        + "_"
        + str(config["boundary_segment_strategy"])
        + "_"
        + str(config["model_params"]["mask_type"])
    )
    # Set the run name based on configuration

    if not config["start_from_scratch"]:
        # Load model and optimizer from a checkpoint directory
        load_dir_path, dataset_load_dir_path = setup_load_directories(
            run_name, config["checkpoint_epoch"]
        )

    # Now set up new directories for saving after restarting
    (
        save_dir_path,
        dataset_save_dir_path,
        frameplots_save_dir_path,
        inference_mse_dir_path,
    ) = setup_save_directories(run_name, config["checkpoint_epoch"])

    # Define the directory path where the model and configuration will be saved
    model_base_file_name = f"heat_diffusion_2D_model_BeyondL_{run_name}"
    hyper_base_file_name = f"config.json_BeyondL_{run_name}"
    optimizer_base_file_name = f"optimizer_state_BeyondL_{run_name}"

    print(f"Random state: {mx.random.state}")
    print(
        f"Configuration: {config['boundary_segment_strategy']}:{config['model_params']['mask_type']}"
    )

    # Print environment variables for debugging
    LIBUSED = os.environ.get("DYLD_LIBRARY_PATH")
    print(f"Hostname: {hostname}: DYLD_LIBRARY_PATH:{LIBUSED}")

    if config[
        "start_from_scratch"
    ]:  # This is a fresh run, create new datasets, save them to file and check
        (
            nx,
            ny,
            training_bcs,
            validation_bcs,
            test_bcs,
            training_alphas,
            validation_alphas,
            test_alphas,
        ) = initialize_geometry_and_bcs(config)
        (
            training_data_mlx,
            training_alphas_mlx,
            training_dts_mlx,
            validation_data_mlx,
            validation_alphas_mlx,
            validation_dts_mlx,
            test_data_mlx,
            test_alphas_mlx,
            test_dts_mlx,
        ) = generate_datasets(
            config,
            training_bcs,
            validation_bcs,
            test_bcs,
            training_alphas,
            validation_alphas,
            test_alphas,
        )

        save_datasets(
            training_data_mlx,
            training_alphas_mlx,
            training_dts_mlx,
            validation_data_mlx,
            validation_alphas_mlx,
            validation_dts_mlx,
            test_data_mlx,
            test_alphas_mlx,
            test_dts_mlx,
            dataset_save_dir_path,
        )

        (
            training_data_mlx_loaded,
            training_alphas_mlx_loaded,
            training_dts_mlx_loaded,
            validation_data_mlx_loaded,
            validation_alphas_mlx_loaded,
            validation_dts_mlx_loaded,
            test_data_mlx_loaded,
            test_alphas_mlx_loaded,
            test_dts_mlx_loaded,
        ) = load_datasets(dataset_save_dir_path)

        # Compare datasets to make sure that they were saved correctly and match when reloaded
        datasets_match = True
        datasets_match &= compare_datasets(
            training_data_mlx, training_data_mlx_loaded, "training_data_mlx"
        )
        datasets_match &= compare_datasets(
            training_alphas_mlx, training_alphas_mlx_loaded, "training_alphas_mlx"
        )
        datasets_match &= compare_datasets(
            training_dts_mlx, training_dts_mlx_loaded, "training_dts_mlx"
        )
        datasets_match &= compare_datasets(
            validation_data_mlx, validation_data_mlx_loaded, "validation_data_mlx"
        )
        datasets_match &= compare_datasets(
            validation_alphas_mlx, validation_alphas_mlx_loaded, "validation_alphas_mlx"
        )
        datasets_match &= compare_datasets(
            validation_dts_mlx, validation_dts_mlx_loaded, "validation_dts_mlx"
        )
        datasets_match &= compare_datasets(
            test_data_mlx, test_data_mlx_loaded, "test_data_mlx"
        )
        datasets_match &= compare_datasets(
            test_alphas_mlx, test_alphas_mlx_loaded, "test_alphas_mlx"
        )
        datasets_match &= compare_datasets(
            test_dts_mlx, test_dts_mlx_loaded, "test_dts_mlx"
        )

        if datasets_match:
            print("All generated and saved datasets match.")
        else:
            print("Some datasets do not match.")
    else:
        # Calculate derived parameters
        (
            training_data_mlx,
            training_alphas_mlx,
            training_dts_mlx,
            validation_data_mlx,
            validation_alphas_mlx,
            validation_dts_mlx,
            test_data_mlx,
            test_alphas_mlx,
            test_dts_mlx,
        ) = load_datasets(dataset_load_dir_path)
        _, _, ny, nx = training_data_mlx.shape

    print(f"training data shape: {training_data_mlx.shape}")
    print(f"validation data shape: {validation_data_mlx.shape}")

    # Initialize model and optimizer
    model, optimizer = initialize_model_and_optimizer(config, nx, ny)
    model.eval()

    mx_random_state = mx.random.state

    if not config["start_from_scratch"]:  # Reload from a previous checkpoint
        checkpoint_epoch = config["checkpoint_epoch"]
        print(load_dir_path)
        (
            start_epoch,
            loaded_optimizer_state,
            loaded_random_state,
            loaded_parameters,
            loaded_config,
        ) = load_model_and_optimizer(
            model,
            optimizer,
            mx_random_state,
            load_dir_path,
            model_base_file_name,
            optimizer_base_file_name,
            hyper_base_file_name,
            checkpoint_epoch,
            comparison=config["compare_current_loaded"],
        )
        if loaded_config:
            # Convert lists to tuples in the loaded config for comparison purposes
            loaded_config = convert_lists_to_tuples(loaded_config)

            # Print side-by-side comparison of current and loaded config
            print_config_comparison(config, loaded_config)

        model.update(parameters=loaded_parameters)

        print(f"Random state before reloading: {mx.random.state}")
        mx.random.state = loaded_random_state
        print(f"Random state after reloading: {mx.random.state}")
        optimizer.init(model.trainable_parameters())
        optimizer.state = loaded_optimizer_state
        print("Current optimizer state after reloading:")
        # print(f'   optimizer.betas: {optimizer.betas}')
        print(f"   optimizer.eps: {optimizer.eps}")
        print(f"   optimizer.step: {optimizer.step}")

        if compare_dict_states(
            optimizer.state, loaded_optimizer_state, "optimizer state"
        ):
            print("After reload: Optimizer state checks")
        else:
            print("After reload: Optimizer state mismatch detected.")

        if compare_list_states(mx.random.state, loaded_random_state, "random state"):
            print("After reload: Random state checks.")
        else:
            print("After reload: Random state mismatch detected.")

        if compare_dict_states(model.parameters(), loaded_parameters, "model state"):
            print("After reload: Model state checks.")
        else:
            print("After reload: Model state mismatch detected.")

    else:  # Start fresh run from scratch
        start_epoch = 0
        if config.get("configuration_dump", False):
            print_fresh_run_config(config)

    eval_cfg = config.setdefault("eval", {})
    n_replace = eval_cfg.get("n_replace", 5)
    n_initial = eval_cfg.get("n_initial_frames", 5)

    train_step, evaluate_step, state = make_train_and_eval_steps(
        model,
        optimizer,
        loss_fn_2D,
        n_initial,
        dx=config["geometry"]["dx"],
        dy=config["geometry"]["dy"],
    )

    print(
        f"************************ Hostname: {hostname} Starting Training *********************** "
    )

    sunspike_dict = {}  # global or outside the loop
    betas2_dict = {}  # global or outside the loop
    # Start or continue training
    train_and_validate(
        model,
        optimizer,
        train_step,
        data_loader_2D,
        evaluate_step,
        config,
        training_data_mlx,
        training_alphas_mlx,
        training_dts_mlx,
        validation_data_mlx,
        validation_alphas_mlx,
        validation_dts_mlx,
        config["model_params"]["batch_size"],
        config["model_params"]["epochs"],
        start_epoch,
        config["save_interval"],
        save_dir_path,  # ← new
        model_base_file_name,  # ← new
        optimizer_base_file_name,  # ← new
        hyper_base_file_name,  # ← new
        dx=config["geometry"]["dx"],
        dy=config["geometry"]["dy"],
    )

    model.eval()

    io_and_plots = config.get("io_and_plots", {})
    plots_cfg = io_and_plots.get("plots", {})

    (
        save_dir_path,
        dataset_save_dir_path,
        frameplots_save_dir_path,
        inference_mse_dir_path,
    ) = setup_save_directories(run_name)

    # save_distribution_violin_plot(sunspike_dict, label="Sunspike", outdir="./sunspike_violin_plots")
    save_distribution_violin_plot(
        sunspike_dict,
        label="Sunspike",
        outdir="./sunspike_violin_plots",
        baseline_value=None,
        sample_every=5,  # keep only 5 & 10
    )
    # save_sunspike_density_heatmap(sunspike_dict, value_range=(0.0, 1.0), outdir="./sunspike_density_plots")
    save_distribution_density_heatmap(
        values_dict=sunspike_dict,
        label="Sunspike",
        num_bins=50,
        value_range=(0.0, 1.0),
        outdir="./sunspike_density_plots",
    )
    save_distribution_violin_plot(
        betas2_dict,
        label="Beta2",
        outdir="./betas2_violin_plots",
        sample_every=5,  # keep only 5 & 10
    )
    # save_distribution_violin_plot(betas2_dict,  label="Beta2", outdir="./betas2_violin_plots")
    # save_sunspike_density_heatmap(betas2_dict, value_range=(0.9, 1.0), outdir="./betas2_density_plots")
    save_distribution_density_heatmap(
        values_dict=betas2_dict,
        label="Beta2",
        num_bins=50,
        value_range=(0.88, 1.0),
        outdir="./betas2_density_plots",
    )

    evaluate_model(
        model,
        evaluate_step,
        data_loader_2D,
        test_data_mlx,
        test_alphas_mlx,
        test_dts_mlx,
        config["geometry"]["dx"],
        config["geometry"]["dy"],
        config["model_params"]["batch_size"],
    )

    if config["model_params"]["mask_type"] == "causal":
        # Call the autoregressive evaluation
        average_mse = evaluate_self_regressive_model_BeyondL(
            model,
            data_loader_2D,
            test_data_mlx,
            test_alphas_mlx,
            test_dts_mlx,
            config["geometry"]["dx"],
            config["geometry"]["dy"],
            config["model_params"]["batch_size"],
            config["model_params"]["time_steps"],
            n_replace,
            n_initial,
            output_dir_regress=inference_mse_dir_path,
        )

        plot_mse_evolution(
            average_mse, output_dir=inference_mse_dir_path, label="causal"
        )

        if plots_cfg.get("movie_frames", False):
            plot_regressive_predictions_2D(
                model,
                data_loader_2D,
                {
                    "data": test_data_mlx,
                    "alphas": test_alphas_mlx,
                    "solution_dts": test_dts_mlx,
                    "batch_size": config["model_params"]["batch_size"],
                    "shuffle": True,
                },
                n_replace=n_replace,
                n_initial=n_initial,
                num_examples=plots_cfg.get("num_examples", 20),
                output_dir=frameplots_save_dir_path,
                t_trained=None,
            )

    elif config["model_params"]["mask_type"] == "block":
        # Call the full sequence evaluation (non-autoregressive)
        average_mse = evaluate_model_block_sequence(
            model,
            data_loader_2D,
            test_data_mlx,
            test_alphas_mlx,
            test_dts_mlx,
            config["geometry"]["dx"],
            config["geometry"]["dy"],
            config["model_params"]["batch_size"],
            config["model_params"]["time_steps"],
            n_initial,
            output_dir_block=inference_mse_dir_path,
        )
        plot_mse_evolution(
            average_mse, output_dir=inference_mse_dir_path, label="block"
        )

        if plots_cfg.get("movie_frames", False):
            plot_predictions_2D(
                model,
                data_loader_2D,
                {
                    "data": test_data_mlx,
                    "alphas": test_alphas_mlx,
                    "solution_dts": test_dts_mlx,
                    "batch_size": config["model_params"]["batch_size"],
                    "shuffle": True,
                },
                num_examples=plots_cfg.get("num_examples", 20),
                output_dir=frameplots_save_dir_path,
                t_trained=None,
            )

    mx_random_state = mx.random.state
    if io_and_plots.get("model_saving", False):
        save_model_and_optimizer(
            model,
            optimizer,
            mx_random_state,
            config,
            config["model_params"]["epochs"],
            save_dir_path,
            model_base_file_name,
            optimizer_base_file_name,
            hyper_base_file_name,
        )


# =========================================================================
#  4) thin executable wrapper
# =========================================================================
def main() -> None:
    """Keeps the file runnable *and* importable as an entry‑point"""
    # build_config() parses the CLI, loads YAML, applies overrides
    cfg = build_config()  # ← uses the helper defined at the top
    run_from_config(cfg)


if __name__ == "__main__":
    main()
