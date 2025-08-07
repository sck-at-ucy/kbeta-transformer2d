"""
demo_heat2d.py – modular CLI entry‑point for the 2‑D heat‑diffusion Transformer
Author : Stavros Kassinos  (Aug 2025)

Key traits
──────────
• Absolutely *no* side‑effects at import time – all work happens in main().
• CLI parsing isolated in _parse_cli().
• YAML + CLI overrides merged by build_config().
• run_from_config(cfg) is the one public “director” routine used by tests.
• Thin main() wrapper keeps the module both importable *and* executable.
"""

# ruff: noqa: E402, F401 -----------------------------------------------------
from __future__ import annotations

import argparse
import json
import os
import socket
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional

import mlx.core as mx
import mlx.optimizers as optim
import numpy as np
import yaml

# ---------------------------------------------------------------------------#
# Optional forward references (mypy / IDEs only)                             #
# ---------------------------------------------------------------------------#
if TYPE_CHECKING:
    from .model import HeatDiffusionModel

    model: HeatDiffusionModel
    optimizer: optim.Optimizer
    train_step: object
    evaluate_step: object
    state: object

# ---------------------------------------------------------------------------#
# Global handles populated by run_from_config()                              #
# ---------------------------------------------------------------------------#
ARGS: Optional[argparse.Namespace] = None
config: Optional[Dict[str, Any]] = None

# ---------------------------------------------------------------------------#
# 1) YAML helpers                                                            #
# ---------------------------------------------------------------------------#
_CONFIG_PKG = "kbeta_transformer2d.configs"


def _resolve_config(arg: str | Path) -> Path:
    """
    Resolve **arg** to an actual YAML file path.

    • If *arg* already points to a file, return it unchanged.
    • Otherwise look for *arg*.yml inside the package’s “configs/” directory.
    """
    p = Path(arg).expanduser()
    if p.is_file():
        return p

    candidate = f"{arg}.yml" if not str(arg).endswith(".yml") else arg
    try:
        with resources.as_file(resources.files(_CONFIG_PKG) / candidate) as fp:
            return fp
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Cannot find config {arg!r}. Either give a real file path or one "
            f"of the built‑in presets in {_CONFIG_PKG}."
        ) from None


def _load_yaml(path: str | Path) -> dict[str, Any]:
    path = _resolve_config(path)
    with path.open("r", encoding="utf‑8") as fh:
        return yaml.safe_load(fh)


# ---------------------------------------------------------------------------#
# 2) CLI                                                                     #
# ---------------------------------------------------------------------------#
def _parse_cli() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python -m kbeta_transformer2d.demo_heat2d",
        description="2‑D heat‑diffusion Transformer (Apple‑MLX)",
    )

    p.add_argument("config", metavar="YAML", help="YAML configuration (file or preset)")

    # convenient shorthands
    p.add_argument("--seed", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--optimizer", choices=["adam95", "adam999", "kourkoutas"])
    p.add_argument("--kour_diagnostics", action="store_true")
    p.add_argument(
        "--collect_spikes",
        action="store_true",
        help="Collect per‑layer *Sun‑spike* /&nbsp;β₂ statistics for "
        "violin / density plots (implies --kour_diagnostics).",
    )

    p.add_argument(
        "--spike_window",
        type=int,
        metavar="N",
        help="Aggregate Sun‑spike / β₂ samples over N epochs "
        "before committing them (maps to tracking.window).",
    )

    p.add_argument(
        "--spike_stride",
        type=int,
        metavar="M",
        help="Show only every M‑th committed window in the violin / "
        "heat‑map plots (maps to tracking.plot_stride).",
    )
    p.add_argument("--viz", action="store_true")

    # free‑form KEY=VAL overrides
    p.add_argument(
        "--override",
        action="extend",
        nargs="+",
        default=[],
        metavar="KEY=VAL",
        help="Arbitrary YAML overrides; dot‑notation allowed",
    )

    args = p.parse_args()

    # ── ❶ fold shorthand flags into args.override ─────────────────────────
    def _js(v):  # JSON‑stringify booleans so “true/false” are parsed correctly
        return json.dumps(v) if isinstance(v, bool) else str(v)

    shorthand = {
        "seed": "seed",
        "epochs": "model_params.epochs",
        "optimizer": "optimizer.name",
        "kour_diagnostics": "optimizer.kour_diagnostics",
        "collect_spikes": "tracking.collect_spikes",
        "spike_window": "tracking.window",
        "spike_stride": "tracking.plot_stride",
        "viz": "io_and_plots.plots.movie_frames",
    }
    for attr, dest in shorthand.items():
        val = getattr(args, attr)
        if val is None or (isinstance(val, bool) and not val):
            continue
        args.override.append(f"{dest}={_js(val)}")

    # ── ❷ normalise common aliases (epochs=…, kour_diagnostics=…) ─────────
    alias = {
        "epochs": "model_params.epochs",
        "kour_diagnostics": "optimizer.kour_diagnostics",
    }
    fixed: list[str] = []
    for pair in args.override:
        key, val = pair.split("=", 1)
        fixed.append(f"{alias.get(key, key)}={val}")
    args.override = fixed

    if args.collect_spikes and not args.kour_diagnostics:
        print("[info] --collect_spikes implies --kour_diagnostics → auto‑enabled")
        args.kour_diagnostics = True
        # make sure the override list reflects the change ⬇︎
        args.override.append("optimizer.kour_diagnostics=true")

    return args


# ---------------------------------------------------------------------------#
# 3) YAML ← CLI merge                                                        #
# ---------------------------------------------------------------------------#
def _apply_overrides(cfg: dict[str, Any], kv_pairs: list[str]) -> None:
    """
    In‑place dot‑notation override helper.
    """
    for pair in kv_pairs:
        if "=" not in pair:
            raise ValueError(f"--override expects KEY=VAL, got {pair!r}")
        key, raw_val = pair.split("=", 1)
        try:
            val: Any = json.loads(raw_val)
        except Exception:  # plain string
            val = raw_val

        tgt = cfg
        *parents, leaf = key.split(".")
        for part in parents:
            tgt = tgt.setdefault(part, {})
        tgt[leaf] = val


def build_config() -> dict[str, Any]:
    """
    CLI ▸ YAML ▸ overrides → final dict (CLI takes precedence).
    """
    args = _parse_cli()
    cfg = _load_yaml(args.config)
    _apply_overrides(cfg, args.override)

    # convenience mirrors
    if "epochs" in cfg:
        cfg.setdefault("model_params", {})["epochs"] = int(cfg.pop("epochs"))
    if "seed" in cfg:
        cfg["seed"] = int(cfg["seed"])
    return cfg


# ---------------------------------------------------------------------------#
# 4) heavy imports (delay until after CLI for faster “‑h” startup)           #
# ---------------------------------------------------------------------------#
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
from .train import (
    evaluate_model,
    evaluate_model_block_sequence,
    evaluate_self_regressive_model_BeyondL,
    make_train_and_eval_steps,
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


# ---------------------------------------------------------------------------#
# 5) top‑level run director                                                  #
# ---------------------------------------------------------------------------#
def run_from_config(cfg: dict[str, Any]) -> None:
    global config, ARGS, model, optimizer, state, train_step, evaluate_step
    config = cfg

    # ----- seeds ----------------------------------------------------------
    seed = cfg.get("seed", 30)
    np.random.seed(seed)
    mx.random.seed(seed)

    print("mx.random state:", mx.random.state)
    print("np.random seed :", np.random.get_state()[1][0])  # type: ignore[index]
    print(
        f"Configuration: {cfg['boundary_segment_strategy']} : "
        f"{cfg['model_params']['mask_type']}"
    )
    print(f"Hostname: {socket.gethostname()}")

    # ----- root output dir ------------------------------------------------
    out_root: str | None = cfg.get("storage", {}).get("outdir")
    # fall back to CWD if unspecified
    base_out = Path(out_root).expanduser().resolve() if out_root else Path.cwd()

    # ----- run label ------------------------------------------------------
    run_name = (
        f"{cfg['run_label']}_{cfg['boundary_segment_strategy']}_"
        f"{cfg['model_params']['mask_type']}"
    )

    # ---------------------------------------------------------------------
    # ① dataset creation / loading
    # ---------------------------------------------------------------------
    if cfg["start_from_scratch"]:
        (
            nx,
            ny,
            train_bcs,
            val_bcs,
            test_bcs,
            train_a,
            val_a,
            test_a,
        ) = initialize_geometry_and_bcs(cfg)

        data = generate_datasets(
            cfg, train_bcs, val_bcs, test_bcs, train_a, val_a, test_a
        )
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
        ) = data

        # save raw datasets next to other artifacts
        _, dataset_dir, *_ = setup_save_directories(
            run_name, base_dir=base_out, restart_epoch=None
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
            dataset_dir,
        )
    else:
        # resume run  → locate previous dataset folder
        load_dir, dataset_dir = setup_load_directories(
            run_name, cfg["checkpoint_epoch"], base_dir=base_out
        )
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
        ) = load_datasets(dataset_dir)
        _, _, ny, nx = training_data_mlx.shape

    print("training data shape   :", training_data_mlx.shape)
    print("validation data shape :", validation_data_mlx.shape)

    # ---------------------------------------------------------------------
    # ② model & optimiser
    # ---------------------------------------------------------------------
    model, optimizer = initialize_model_and_optimizer(cfg, nx, ny)
    model.eval()

    # ---------------------------------------------------------------------
    # ③ storage layout
    # ---------------------------------------------------------------------
    save_dir, dataset_dir, frames_dir, mse_dir = setup_save_directories(
        run_name, base_dir=base_out, restart_epoch=cfg.get("checkpoint_epoch")
    )

    # file stems (epoch suffixes are appended later)
    model_base = f"heat_diffusion_2D_model_BeyondL_{run_name}"
    optim_base = f"optimizer_state_BeyondL_{run_name}"
    hyper_base = f"config.json_BeyondL_{run_name}"

    # ---------------------------------------------------------------------
    # ④ fresh vs resume checkpoint logic
    # ---------------------------------------------------------------------
    if not cfg["start_from_scratch"]:
        chk_epoch = cfg["checkpoint_epoch"]
        (
            start_epoch,
            loaded_opt_state,
            loaded_rng,
            loaded_params,
            loaded_cfg,
        ) = load_model_and_optimizer(
            model,
            optimizer,
            mx.random.state,
            load_dir,
            model_base,
            optim_base,
            hyper_base,
            chk_epoch,
            comparison=cfg["compare_current_loaded"],
        )
        # update & sanity prints
        model.update(parameters=loaded_params)
        mx.random.state = loaded_rng
        optimizer.state = loaded_opt_state
    else:
        start_epoch = 0
        if cfg.get("configuration_dump", False):
            print_fresh_run_config(cfg)

    # ── choose the save interval based on save_checkpoints ──────────────
    if config.get("save_checkpoints", True):
        # honour user‑provided value or fall back to 10
        config["save_interval"] = int(config.get("save_interval", 10))
    else:
        # ‼️ *None* marks “do not checkpoint during training”
        config["save_interval"] = None

    # ---------------------------------------------------------------------
    # ⑤ compile train / eval closures
    # ---------------------------------------------------------------------
    eval_cfg = cfg.setdefault("eval", {})
    n_replace = eval_cfg.get( # noqa: F841
        "n_replace", 5
    )  # Currently unused, reserved for future dev. # noqa: F841
    n_initial = eval_cfg.get("n_initial_frames", 5)

    train_step, eval_step, state = make_train_and_eval_steps(
        model,
        optimizer,
        loss_fn_2D,
        n_initial,
        dx=cfg["geometry"]["dx"],
        dy=cfg["geometry"]["dy"],
    )

    # ---------------------------------------------------------------------
    # ⑥ training
    # ---------------------------------------------------------------------
    print("*** starting training ***")

    model.train()

    sunspike_dict, beta2_dict = train_and_validate(
        model,
        optimizer,
        train_step,
        data_loader_2D,
        eval_step,
        cfg,
        training_data_mlx,
        training_alphas_mlx,
        training_dts_mlx,
        validation_data_mlx,
        validation_alphas_mlx,
        validation_dts_mlx,
        cfg["model_params"]["batch_size"],
        cfg["model_params"]["epochs"],
        start_epoch,
        cfg["save_interval"],
        save_dir,
        model_base,
        optim_base,
        hyper_base,
        dx=cfg["geometry"]["dx"],
        dy=cfg["geometry"]["dy"],
    )

    # ---------------------------------------------------------------------
    # ⑦ evaluation & plots
    # ---------------------------------------------------------------------
    io_and_plots = cfg.get("io_and_plots", {})
    plots_cfg = io_and_plots.get("plots", {})

    # ——————————————————————————————————————————————
    # Diagnostics → violin / heat‑maps (Kour only)
    # ——————————————————————————————————————————————
    track_cfg = cfg.get("tracking", {})
    collect = track_cfg.get("collect_spikes", False)
    is_kour = cfg["optimizer"]["name"].lower().startswith("kour")

    # Have we actually accumulated *any* values?
    has_data = any(len(v) for v in sunspike_dict.values())

    if collect and is_kour and has_data:
        window = int(track_cfg.get("window", 500))
        plot_stride = int(track_cfg.get("plot_stride", 10 * window))

        # ── violin plots ────────────────────────────────────────────────
        save_distribution_violin_plot(
            sunspike_dict,
            sample_every=plot_stride,
            label="Sunspike",
            outdir=frames_dir / "sunspike_violin",
        )
        save_distribution_violin_plot(
            beta2_dict,
            sample_every=plot_stride,
            label="Beta2",
            outdir=frames_dir / "beta2_violin",
        )

        # ── density heat‑maps ───────────────────────────────────────────
        save_distribution_density_heatmap(
            sunspike_dict,
            label="Sunspike",
            num_bins=20,
            value_range=(0.0, 1.0),
            outdir=frames_dir / "sunspike_density",
        )
        save_distribution_density_heatmap(
            beta2_dict,
            label="Beta2",
            num_bins=20,
            value_range=(0.88, 1.0),
            outdir=frames_dir / "beta2_density",
        )
    else:
        print(
            "[plots] Skipped sun‑spike / β₂ plots — "
            f"collect={collect}, is_kour={is_kour}, has_data={has_data}"
        )

    # non‑autoregressive baseline
    average_mse = evaluate_model_block_sequence(
        model,
        data_loader_2D,
        test_data_mlx,
        test_alphas_mlx,
        test_dts_mlx,
        cfg["geometry"]["dx"],
        cfg["geometry"]["dy"],
        cfg["model_params"]["batch_size"],
        cfg["model_params"]["time_steps"],
        n_initial,
        output_dir_block=mse_dir,
    )
    plot_mse_evolution(average_mse, output_dir=mse_dir, label="block")

    if plots_cfg.get("movie_frames", False):
        plot_predictions_2D(
            model,
            data_loader_2D,
            {
                "data": test_data_mlx,
                "alphas": test_alphas_mlx,
                "solution_dts": test_dts_mlx,
                "batch_size": cfg["model_params"]["batch_size"],
                "shuffle": True,
            },
            num_examples=plots_cfg.get("num_examples", 20),
            output_dir=frames_dir,
        )

    # optional full checkpoint at end
    if io_and_plots.get("model_saving", False):
        save_model_and_optimizer(
            model,
            optimizer,
            mx.random.state,
            cfg,
            cfg["model_params"]["epochs"],
            save_dir,
            model_base,
            optim_base,
            hyper_base,
        )

    print(f"[done] Artifacts in {save_dir.parent}")


# ---------------------------------------------------------------------------#
# thin executable wrapper                                                    #
# ---------------------------------------------------------------------------#
def main() -> None:
    cfg = build_config()
    run_from_config(cfg)


if __name__ == "__main__":
    main()
