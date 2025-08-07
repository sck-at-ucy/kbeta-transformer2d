# transformer/train.py
"""file start: train.py
Training / validation / evaluation loops"""

from __future__ import annotations

import os
import time
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

__all__ = [
    "train_and_validate",
    "evaluate_model",
    "evaluate_model_block_sequence",
    "evaluate_self_regressive_model_BeyondL",
    "make_train_and_eval_steps",
]

from functools import partial

import matplotlib.pyplot as plt

from .io_utils import save_model_and_optimizer  # now explicit


def make_train_and_eval_steps(model, optimizer, loss_fn, n_initial, *, dx, dy):
    state = [model.state, optimizer.state, mx.random.state]

    @partial(mx.compile, inputs=state, outputs=state)
    def _train_step(src, target, src_alphas, src_dts, dx, dy):
        loss, grads = nn.value_and_grad(model, loss_fn)(
            model, src, target, n_initial, src_alphas, src_dts, dx, dy
        )
        optimizer.update(model, grads)
        return loss

    @partial(mx.compile, inputs=state, outputs=())
    def _eval_step(src, target, src_alphas, src_dts, dx, dy):
        return loss_fn(model, src, target, n_initial, src_alphas, src_dts, dx, dy)

    return _train_step, _eval_step, state


# Training and validation function with periodic saving
def train_and_validate(
    model,
    optimizer,
    train_step,
    data_loader_2D,
    evaluate_step,
    cfg: dict[str, Any],
    train_data,
    train_alphas,
    train_dts,
    validation_data,
    validation_alphas,
    validation_dts,
    batch_size,
    epochs,
    start_epoch,
    save_interval: int | None,
    save_dir_path,  # ← new
    model_base_file_name,  # ← new
    optimizer_base_file_name,  # ← new
    hyper_base_file_name,  # ← new
    dx,
    dy,
):
    """
    Trains and validates a 2D heat diffusion model over a specified number of epochs.

    This function performs the training and validation of a 2D heat diffusion model. It iterates over
    the training and validation datasets in mini-batches, calculates the loss, and updates the model
    parameters using the optimizer. The model and optimizer states are periodically saved, and
    the training and validation losses are printed after each epoch. If training is resumed from
    a checkpoint, the model and optimizer are reloaded, and the learning rate is adjusted accordingly.

    Parameters
    ----------
    train_data : numpy.ndarray or mlx.core.array
        The training dataset representing the temperature distribution over time. The shape is
        (num_samples, time_steps, ny, nx).
    train_alphas : numpy.ndarray or mlx.core.array
        The thermal diffusivity values for each sample in the training dataset, with shape (num_samples,).
    train_dts : numpy.ndarray or mlx.core.array
        The time step sizes for each sample in the training dataset, with shape (num_samples,).
    validation_data : numpy.ndarray or mlx.core.array
        The validation dataset representing the temperature distribution over time. The shape is
        (num_samples, time_steps, ny, nx).
    validation_alphas : numpy.ndarray or mlx.core.array
        The thermal diffusivity values for each sample in the validation dataset, with shape (num_samples,).
    validation_dts : numpy.ndarray or mlx.core.array
        The time step sizes for each sample in the validation dataset, with shape (num_samples,).
    batch_size : int
        The number of samples per batch during training and validation.
    epochs : int
        The total number of epochs to train the model.
    start_epoch : int
        The epoch from which to resume training. Typically 0 for a fresh start or the epoch from
        which training is resumed after a checkpoint.
    save_interval : int
        The number of epochs between saving the model and optimizer states.
    dx : float
        The spatial step size in the x direction.
    dy : float
        The spatial step size in the y direction.

    Returns
    -------
    None
        The function performs training and validation and does not return any value. It prints
        training and validation losses and saves the model and optimizer states at regular intervals.
    """
    # ------------------------------------------------------------
    #  light‑weight per‑epoch statistics (Sun‑spike & β₂)
    # ------------------------------------------------------------
    tr_cfg = cfg.get("tracking", {})
    track_diag = tr_cfg.get("collect_spikes", False)
    WINDOW = int(tr_cfg.get("window", 1))  # epochs per violin window

    sunspike_log: dict[int, list[float]] = {}
    beta2_log: dict[int, list[float]] = {}
    _buf_spikes: list[float] = []  # rolling window
    _buf_betas2: list[float] = []

    tic = time.perf_counter()
    print(
        f"start_epoch: {start_epoch} epochs: {epochs} learning_rate: {optimizer.learning_rate}"
    )

    for epoch in range(start_epoch, epochs):
        if epoch in cfg.get("learning_rate_schedule", {}):
            optimizer.learning_rate = cfg["learning_rate_schedule"][epoch]
        if epoch in cfg.get("scale_schedule", {}):
            optimizer.scale = cfg["scale_schedule"][epoch]

        total_train_loss = 0
        num_train_batches = 0

        for src, target, src_alphas, src_dts in data_loader_2D(
            train_data, train_alphas, train_dts, batch_size, shuffle=True
        ):
            loss = train_step(src, target, src_alphas, src_dts, dx, dy)

            # ── Sun‑spike & β₂ per‑batch sample ─────────────────────────
            if (
                track_diag
                and getattr(optimizer, "_diag", False)  # diagnostics on
                and hasattr(optimizer, "snapshot_sunspike_history")
            ):
                spikes, betas = optimizer.snapshot_sunspike_history()
                _buf_spikes.extend(spikes)
                _buf_betas2.extend(betas)

            total_train_loss += loss.item()
            num_train_batches += 1
            mx.eval(model, optimizer.state)

        total_val_loss = 0
        num_val_batches = 0
        model.eval()

        for src, target, src_alphas, src_dts in data_loader_2D(
            validation_data,
            validation_alphas,
            validation_dts,
            batch_size,
            shuffle=False,
        ):
            val_loss = evaluate_step(src, target, src_alphas, src_dts, dx, dy)
            total_val_loss += val_loss.item()
            num_val_batches += 1

        print(
            f"Epoch {epoch + 1}, lr: {optimizer.learning_rate}, "
            f"Training Loss: {total_train_loss / num_train_batches}, "
            f"Validation Loss: {total_val_loss / num_val_batches}, "
            f"Number of Train batches: {num_train_batches}, "
        )

        # --------------------------------------------
        #  pretty print epoch summary *and* diagnostics
        # --------------------------------------------
        if hasattr(optimizer, "snapshot_diagnostics") and getattr(
            optimizer, "_diag", False
        ):
            # Kourkoutas – rich info available
            diags = optimizer.snapshot_diagnostics()
            print(
                "   ↳ "
                f"denom_min={diags['diag_denom_min']:.2e} | "
                f"upd/ρ_max={diags['diag_max_ratio']:.1f} | "
                f"upd_norm_max={diags['diag_upd_norm_max']:.1e} | "
                f"v̂_max={diags['diag_vhat_max']:.1e}"
            )
        # ── commit & reset rolling buffers every *WINDOW* epochs ─────────
        if track_diag and (epoch + 1) % WINDOW == 0:
            sunspike_log[epoch + 1] = _buf_spikes[:]
            beta2_log[epoch + 1] = _buf_betas2[:]
            _buf_spikes.clear()
            _buf_betas2.clear()

        if save_interval and (epoch + 1) % save_interval == 0:
            mx.eval(model.parameters(), optimizer.state)
            model.eval()
            save_model_and_optimizer(
                model,
                optimizer,
                mx.random.state,
                cfg,
                epoch + 1,
                save_dir_path,
                model_base_file_name,
                optimizer_base_file_name,
                hyper_base_file_name,
            )
        model.train()
    # ── flush final (partial) window, if any ───────────────────────────
    if track_diag and (_buf_spikes or _buf_betas2):
        sunspike_log[epochs] = _buf_spikes
        beta2_log[epochs] = _buf_betas2

    toc = time.perf_counter()
    tpi = (toc - tic) / 60 / (epochs + 1 - start_epoch)
    print(f"Time per epoch {tpi:.3f} (min)")
    # ----------------------------------------------------------------
    #  return diagnostics for downstream plots (run_from_config needs)
    # ----------------------------------------------------------------
    return sunspike_log, beta2_log


def evaluate_model(
    model,
    evaluate_step,
    data_loader_func,
    test_data,
    test_alphas,
    test_dts,
    dx,
    dy,
    batch_size,
):
    """
    Evaluates the performance of the model on the test dataset and measures inference time.

    Parameters
    ----------
    (As previously defined)
    """
    total_loss = 0
    num_batches = 0
    data_loader = data_loader_func(
        test_data, test_alphas, test_dts, batch_size, shuffle=False
    )

    # Start timing the inference
    start_time = time.perf_counter()

    for src, target, src_alphas, src_dts in data_loader:
        loss = evaluate_step(src, target, src_alphas, src_dts, dx, dy)
        total_loss += loss.item()
        num_batches += 1

    # End timing the inference
    end_time = time.perf_counter()

    # Calculate the average inference time
    inference_time = end_time - start_time
    average_inference_time_per_batch = (
        inference_time / num_batches if num_batches > 0 else float("inf")
    )

    # Print results
    average_loss = total_loss / num_batches if num_batches > 0 else float("inf")
    print(f"Average Test Loss: {average_loss}")
    print(f"Total Inference Time: {inference_time:.4f} seconds")
    print(
        f"Average Inference Time per Batch: {average_inference_time_per_batch:.4f} seconds"
    )

    return average_loss, average_inference_time_per_batch


def evaluate_self_regressive_model_BeyondL(
    model,
    data_loader_func,
    test_data,
    test_alphas,
    test_dts,
    dx,
    dy,
    batch_size,
    seq_len,
    n_replace,
    n_initial,
    output_dir_regress="./MSE_step_regress",
):
    """
    Evaluates the model in a self-regressive manner and tracks MSE for each predicted time step.

    This function evaluates a 2D heat diffusion model by performing self-regressive prediction,
    tracking the Mean Squared Error (MSE) for each time step. In self-regressive evaluation, the model
    predicts a sequence of temperature distributions, where each predicted frame is used as input to predict
    future frames. The function accumulates the MSE loss for each time step across all batches, and
    produces a plot of MSE evolution over time steps.

    Parameters
    ----------
    model : object
        The trained 2D heat diffusion model to be evaluated.
    data_loader_func : function
        The function that generates mini-batches from the dataset. It should accept the test data,
        alphas, time steps, batch size, and shuffle option.
    test_data : numpy.ndarray or mlx.core.array
        The test dataset representing the temperature distribution over time. The shape is
        (num_samples, time_steps, ny, nx).
    test_alphas : numpy.ndarray or mlx.core.array
        The thermal diffusivity values for each sample in the test dataset, with shape (num_samples,).
    test_dts : numpy.ndarray or mlx.core.array
        The time step sizes for each sample in the test dataset, with shape (num_samples,).
    dx : float
        The spatial step size in the x direction.
    dy : float
        The spatial step size in the y direction.
    batch_size : int
        The number of samples per batch during evaluation.
    seq_len : int
        The total number of time steps (sequence length) for the evaluation.
    output_dir_regress : str, optional
        The directory where the MSE plots will be saved. Defaults to "./MSE_step_regress".

    Returns
    -------
    None

    This function tracks the MSE for each time step and generates a plot showing the evolution
    of MSE across time steps. The plot is saved as an image in the specified output directory.
    """
    np.set_printoptions(threshold=np.inf)

    if not os.path.exists(output_dir_regress):
        os.makedirs(output_dir_regress)

    data_loader = data_loader_func(
        test_data, test_alphas, test_dts, batch_size, shuffle=False
    )

    # Initialize MSE tracker for all time steps
    cumulative_mse = np.zeros(seq_len - n_initial)
    num_batches = 0

    for src, target, src_alphas, src_dts in data_loader:
        # Initialize per-batch MSE tracker for all time steps
        time_step_mse = np.zeros(seq_len - n_initial)

        # Loop through time steps autoregressively, replacing predictions
        for t in range(n_initial, seq_len, n_replace):
            prediction = model(src, src_alphas)
            end_idx = min(t + n_replace, seq_len)
            src[:, t:end_idx, :, :] = prediction[:, t:end_idx, :, :]

        # Track and accumulate MSE for each time step (starting from t=5)
        for t in range(n_initial, seq_len):
            mse_loss = nn.losses.mse_loss(
                prediction[:, t, :, :], target[:, t, :, :], reduction="mean"
            )
            time_step_mse[t - n_initial] = (
                mse_loss.item()
            )  # Store MSE for the current batch
            cumulative_mse[
                t - n_initial
            ] += mse_loss.item()  # Accumulate MSE for this time step

        num_batches += 1
        print(f"finished batch: {num_batches}")

    # After looping through all batches:
    # Average the MSE by the number of batches
    average_mse = cumulative_mse / num_batches
    print(f"Average MSE over auto-regressive sequence {average_mse}")

    return average_mse

    # Plot MSE evolution over time steps
    plt.figure(figsize=(10, 6))
    plt.plot(
        range(n_initial, seq_len), average_mse, marker="o", linestyle="-", color="b"
    )
    plt.xlabel("Time Step")
    plt.ylabel("MSE Loss")
    plt.title("Average MSE Evolution Over Autoregressive Time Steps")
    plt.grid(True)
    mse_plot_filename = os.path.join(output_dir_regress, "mse_evolution.png")
    plt.savefig(mse_plot_filename)
    plt.close()

    print(f"MSE for each time step saved in {mse_plot_filename}")


def evaluate_model_block_sequence(
    model,
    data_loader_func,
    test_data,
    test_alphas,
    test_dts,
    dx,
    dy,
    batch_size,
    seq_len,
    n_initial,
    output_dir_block="./MSE_step_block",
):
    """
    Evaluates the model by predicting the entire sequence at once and tracks MSE for each predicted time step.
    This version assumes the model predicts the entire sequence at once (not autoregressively).
    """
    batch_size = 4
    np.set_printoptions(threshold=np.inf)

    if not os.path.exists(output_dir_block):
        os.makedirs(output_dir_block)

    data_loader = data_loader_func(
        test_data, test_alphas, test_dts, batch_size, shuffle=False
    )

    # Initialize MSE tracker for all time steps
    cumulative_mse = np.zeros(seq_len - n_initial)
    num_batches = 0

    for src, target, src_alphas, src_dts in data_loader:
        # Initialize per-batch MSE tracker for all time steps
        time_step_mse = np.zeros(seq_len - n_initial)

        # Get full sequence predictions
        prediction = model(src, src_alphas)

        # Track and accumulate MSE for each time step (starting from t=5)
        for t in range(n_initial, seq_len):
            mse_loss = nn.losses.mse_loss(
                prediction[:, t, :, :], target[:, t, :, :], reduction="mean"
            )
            time_step_mse[t - n_initial] = (
                mse_loss.item()
            )  # Store MSE for the current batch
            cumulative_mse[
                t - n_initial
            ] += mse_loss.item()  # Accumulate MSE for this time step

        num_batches += 1

    # After looping through all batches:
    # Average the MSE by the number of batches
    average_mse = cumulative_mse / num_batches
    print(f"Average MSE over block sequence {average_mse}")

    return average_mse
