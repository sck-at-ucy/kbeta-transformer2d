# transformer/optim_factory.py
"""
Return (model, optimizer) according to YAML/CLI.
"""

from __future__ import annotations

# ‑‑ stdlib -------------------------------------------------
from typing import Any, Callable

# ‑‑ third‑party -------------------------------------------
import mlx.core as mx
import mlx.optimizers as optim
from kbeta.optim import KourkoutasBeta

# ‑‑ local --------------------------------------------------
from .model import HeatDiffusionModel

Schedule = Callable[[mx.array], mx.array]


# ---------------------------------------------------------------------
# 1) tiny utility ------------------------------------------------------
# ---------------------------------------------------------------------
def _cosine_then_const(init: float, target: float, ramp_steps: int) -> Schedule:
    """
    Cosine‑decay from ``init`` → ``target`` for ``ramp_steps``,
    then keep the LR flat at ``target``.
    """
    cosine_part = optim.cosine_decay(init, decay_steps=ramp_steps, end=target)

    def _constant(_: int) -> float:
        return target

    return optim.join_schedules([cosine_part, _constant], [ramp_steps])


# ---------------------------------------------------------------------
# 2) main public helper ------------------------------------------------
# ---------------------------------------------------------------------
def initialize_model_and_optimizer(
    cfg: dict[str, Any], nx: int, ny: int
) -> tuple[HeatDiffusionModel, optim.Adam | KourkoutasSoftmaxFlex]:

    p = cfg["model_params"]

    # ---------------- make the model ---------------------------------
    model = HeatDiffusionModel(
        ny,
        nx,
        p["time_steps"],
        p["num_heads"],
        p["num_encoder_layers"],
        p["mlp_dim"],
        p["embed_dim"],
        p["start_predicting_from"],
        p["mask_type"],
    )
    mx.eval(model.parameters())  # materialise weights on device

    # ---------------- learning‑rate schedule -------------------------
    opt_cfg = cfg.setdefault("optimizer", {})
    init_lr = opt_cfg.get("init_lr", 1e-3)
    target_lr = opt_cfg.get("target_lr", 1e-5)
    ramp_ep = opt_cfg.get("ramp_steps", 60_000)

    lr_schedule = _cosine_then_const(init_lr, target_lr, ramp_ep)

    # ---------------- pick optimiser --------------------------------
    name: str = opt_cfg.get("name", "adam999").lower()
    optimizer: optim.Adam | KourkoutasSoftmaxFlex

    if name == "adam95":
        optimizer = optim.Adam(
            learning_rate=lr_schedule,
            betas=[0.90, 0.95],
            eps=1e-8,
            bias_correction=True,
        )

    elif name == "adam999":
        optimizer = optim.Adam(
            learning_rate=lr_schedule,
            betas=[0.90, 0.999],
            eps=1e-8,
            bias_correction=True,
        )
        # _pretty_print_adam("ADAM999", optimizer)

    elif name == "kourkoutas":
        # ------ optional layer‑key fn (stable path) ------------------
        from mlx.utils import tree_flatten

        param_to_path = {
            param: ".".join(map(str, path))
            for path, param in tree_flatten(model.parameters())
        }

        def layer_key_fn(param) -> str:
            """Stable bucket for per‑layer statistics in Kourkoutas‑β."""
            return param_to_path.get(param, "unknown")

        optimizer = KourkoutasBeta(
            learning_rate=1e-3,
            beta1=opt_cfg.get("beta1", 0.90),
            beta2_max=opt_cfg.get("beta2_max", 0.999),
            beta2_min=opt_cfg.get("beta2_min", 0.88),
            alpha=opt_cfg.get("alpha", 0.93),
            eps=opt_cfg.get("eps", 1e-8),
            tiny_spike=opt_cfg.get("tiny_spike", 1e-8),
            tiny_denom=opt_cfg.get("tiny_denom", 1e-8),
            decay=opt_cfg.get("decay", None),
            adaptive_tiny=opt_cfg.get("adaptive_tiny", False),
            max_ratio=opt_cfg.get("max_ratio", None),
            warmup_steps=opt_cfg.get("warmup_steps", 350),
            bias_correction=opt_cfg.get("bias_correction", "beta2max"),
            layer_key_fn=layer_key_fn,
            diagnostics=opt_cfg.get("kour_diagnostics", False),
        )

    else:
        raise ValueError(
            f"Unknown optimiser '{name}'. Valid: adam95 | adam999 | kourkoutas"
        )

    if isinstance(optimizer, KourkoutasBeta):
        _pretty_print_kour(optimizer)  # dedicated formatter
    else:  # plain Adam
        _pretty_print_adam(name.upper(), optimizer)  # reuse helper

    optimizer.init(model.parameters())
    return model, optimizer


# ---------------------------------------------------------------------
# 3) pretty‑printers (keep console tidy) ------------------------------
# ---------------------------------------------------------------------
def _pretty_print_adam(label: str, opt: optim.Adam) -> None:
    print(
        f"{label}  β1,β2={opt.betas} | eps={opt.eps:.2e} | "
        f"bias_correction={opt.bias_correction}"
    )


def _pretty_print_kour(opt: KourkoutasSoftmaxFlex) -> None:
    print(
        "KOUR   "
        f"β1={opt.beta1} | β2_max={opt.beta2_max} | β2_min={opt.beta2_min} | "
        f"α={opt.alpha} | tiny={opt.tiny_spike:.1e}/{opt.tiny_denom:.1e} | "
        f"adaptTiny={opt.adaptive_tiny} | decay={opt.decay or 'off'} | "
        f"maxR={opt.max_ratio or 'off'} | eps={opt.eps:.2e} | "
        f"diagnostics={opt._diag}"
    )
