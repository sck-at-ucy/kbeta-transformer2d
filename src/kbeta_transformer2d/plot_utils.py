# transformer/plot_utils.py
# ─────────────────────────────────────────────────────────────────────────────
# High‑level plotting utilities for the transformer2d demo
# ─────────────────────────────────────────────────────────────────────────────

import os

import matplotlib.pyplot as plt
import mlx.core as mx
import numpy as np


def plot_sunspike_histogram(sunspike_list, epoch_idx, bins=50):
    """
    Convenience function to plot a histogram of the sunspike values.
    """
    import matplotlib.pyplot as plt

    # Convert to float if needed:
    sun_vals = [x.item() for x in sunspike_list]

    plt.hist(sun_vals, bins=bins)
    plt.title(f"Sunspike Distribution at epoch {epoch_idx}")
    plt.xlabel("Sunspike")
    plt.ylabel("Frequency")
    plt.show()


def save_sunspike_histogram(sunspike_list, epoch_idx, bins=50, outdir="./plots"):
    """
    Convenience function to create and SAVE a histogram of the sunspike values
    at a given epoch. This avoids memory overhead from displaying on-screen.
    """

    os.makedirs(outdir, exist_ok=True)

    sun_vals = [x for x in sunspike_list]

    plt.figure()
    plt.hist(sun_vals, bins=bins, range=(0.0, 1.00))
    plt.title(f"Sunspike Distribution at epoch {epoch_idx}")
    plt.xlabel("Sunspike")
    plt.ylabel("Frequency")
    plt.ylim(0, 40)

    # Save the figure
    outfile = os.path.join(outdir, f"sunspike_epoch{epoch_idx}.png")
    plt.savefig(outfile)
    plt.close()  # close the figure to free memory


def plot_beta2_histogram(beta2_list, epoch_idx, bins=50):
    """
    Convenience function to plot a histogram of the sunspike values.
    """

    # Convert to float if needed:
    beta2_vals = [x.item() for x in beta2_list]

    plt.hist(beta2_vals, bins=bins)
    plt.title(f"Beta2 Distribution at epoch {epoch_idx}")
    plt.xlabel("Beta2 value")
    plt.ylabel("Frequency")
    plt.show()


def save_beta2_histogram(beta2_list, epoch_idx, bins=50, outdir="./plots"):
    """
    Convenience function to create and SAVE a histogram of the sunspike values
    at a given epoch. This avoids memory overhead from displaying on-screen.
    """

    os.makedirs(outdir, exist_ok=True)

    beta2_vals = [x for x in beta2_list]

    plt.figure()
    plt.hist(beta2_vals, bins=bins, range=(0.9, 1.00))
    plt.title(f"Beta2 Distribution at epoch {epoch_idx}")
    plt.xlabel("Beta2 value")
    plt.ylabel("Frequency")
    plt.ylim(0, 40)

    # Save the figure
    outfile = os.path.join(outdir, f"beta2_epoch{epoch_idx}.png")
    plt.savefig(outfile)
    plt.close()  # close the figure to free memory


def save_distribution_violin_plot_old(
    values_dict, label="Sunspike", outdir="./violin_plots"
):
    """
    Creates and saves a violin plot showing the distribution of either 'sunspike'
    or 'beta2' (or anything else) values for each epoch.

    Parameters
    ----------
    values_dict : dict
        A dictionary mapping epoch -> list of numeric values (e.g. sunspike or beta2).
        Example: {1: [0.01, 0.02, ...], 2: [0.015, 0.03, ...], ...}
    label : str
        The label for your data ("Sunspike", "Beta2", etc.), used in plot title/axes.
    outdir : str
        Directory to save the output plot.

    Returns
    -------
    None
    """
    import os

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    os.makedirs(outdir, exist_ok=True)

    # Build a DataFrame with columns ["epoch", "value"]
    rows = []
    for epoch, vals in values_dict.items():
        # Only collect data if epoch is a multiple of 10
        if epoch % 5 == 0:
            for v in vals:
                rows.append((epoch, float(v)))

    df = pd.DataFrame(rows, columns=["epoch", "value"])

    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df, x="epoch", y="value", color="skyblue", linewidth=1.0)
    sns.swarmplot(data=df, x="epoch", y="value", color="k", alpha=0.3, size=2)

    plt.title(f"{label} Distribution per Epoch (Violin Plot)")
    plt.xlabel("Epoch")
    plt.ylabel(label)

    plt.xticks(rotation=0)
    plt.tight_layout()

    # Save
    outfile = os.path.join(outdir, f"{label.lower()}_violin_plot.png")
    plt.savefig(outfile, dpi=300)
    plt.close()

    print(f"Violin plot saved to {outfile}")


def save_distribution_violin_plot(
    values_dict,
    *,
    label="Beta2",
    outdir="./violin_plots",
    sample_every=5,
    baseline_value=0.999,
    baseline_label="Adam β₂ = 0.999",
    show_medians=True,
):
    """
    Violin plot of per‑epoch distributions with baseline & median overlay.
    """
    import os

    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns

    os.makedirs(outdir, exist_ok=True)

    # ── build tidy dataframe ────────────────────────────────
    rows = [
        (epoch, float(v))
        for epoch, vals in values_dict.items()
        if epoch % sample_every == 0
        for v in vals
    ]
    if not rows:
        print("Nothing to plot – check sample_every / values_dict.")
        return

    df = pd.DataFrame(rows, columns=["epoch", "value"])
    df["epoch"] = df["epoch"].astype(str)  # treat as discrete categories

    # compute category order once
    order = sorted(df["epoch"].unique(), key=int)

    # ── figure & violin ────────────────────────────────────
    plt.figure(figsize=(12, 6))
    ax = sns.violinplot(
        data=df,
        x="epoch",
        y="value",
        color="#8ecae6",
        order=order,
        inner=None,
        linewidth=1.2,
        cut=0,
    )
    sns.swarmplot(
        data=df,
        x="epoch",
        y="value",
        color="k",
        alpha=0.3,
        size=2,
        ax=ax,
    )

    # ── baseline line ──────────────────────────────────────
    if baseline_value is not None:
        ax.axhline(
            baseline_value,
            ls="--",
            lw=1.0,
            color="red",
            label=baseline_label,
            zorder=3,
        )
        ax.legend(
            title="",
            frameon=False,
            handlelength=1.2,
            loc="upper right",
            borderpad=0.2,
        )

    # ── median overlay ─────────────────────────────────────
    if show_medians:
        med = df.groupby("epoch")["value"].median().reindex(order)
        ax.scatter(order, med.values, s=30, color="white", edgecolor="black", zorder=4)
        ax.plot(order, med.values, color="black", lw=1, alpha=0.6, zorder=4)

    # ── axes cosmetics ─────────────────────────────────────
    ymin, ymax = df["value"].min(), df["value"].max()
    if baseline_value is not None:
        ymax = max(ymax, baseline_value)
        ymin = min(ymin, baseline_value)
    pad = 0.02 * (ymax - ymin if ymax > ymin else 1.0)
    ax.set_ylim(ymin - pad, ymax + pad)

    ax.set_title(f"{label} Distribution per Epoch (Violin Plot)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(label)
    ax.legend(frameon=False, loc="upper right")

    plt.tight_layout()
    outfile = os.path.join(outdir, f"{label.lower()}_violin_plot.png")
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"Violin plot saved to {outfile}")


def save_distribution_density_heatmap(
    values_dict,
    label="Sunspike",
    num_bins=50,
    value_range=(0.0, 1.0),
    outdir="./density_heatmap",
):
    """
    Creates and saves a 2D heatmap, where the y-axis is epoch and the x-axis is
    bins of the distribution (e.g. sunspike or beta2). The color indicates how
    many values fell into each bin at that epoch.

    Parameters
    ----------
    values_dict : dict
        A dictionary mapping epoch -> list of values (e.g. sunspike or beta2)
        observed in that epoch. For example:
            { 1: [0.01, 0.02, ...],
              2: [0.015, 0.03, ...],
              ... }
    label : str
        A short name/label for the distribution, e.g. "Sunspike" or "Beta2".
    num_bins : int
        Number of bins to use for the distribution axis.
    value_range : tuple
        The (min, max) range for the distribution values, e.g. (0.0, 1.0).
    outdir : str
        Directory to save the output heatmap image.

    Returns
    -------
    None
    """

    os.makedirs(outdir, exist_ok=True)

    # Sort epochs so we iterate in ascending order
    all_epochs = sorted(values_dict.keys())
    if not all_epochs:
        print("No data to plot in values_dict.")
        return

    # Prepare a 2D array for histogram counts:
    # rows = number of epochs, columns = num_bins
    epoch_hist = np.zeros((len(all_epochs), num_bins), dtype=np.float32)

    bin_edges = np.linspace(value_range[0], value_range[1], num_bins + 1)

    for i, epoch in enumerate(all_epochs):
        vals = np.array(values_dict[epoch], dtype=np.float32)
        counts, _ = np.histogram(vals, bins=bin_edges)
        epoch_hist[i, :] = counts

    plt.figure(figsize=(10, 6))

    # 'origin=lower' => smallest epoch at the bottom
    plt.imshow(
        epoch_hist,
        extent=[value_range[0], value_range[1], all_epochs[0], all_epochs[-1]],
        aspect="auto",
        origin="lower",
        cmap="plasma",  #'viridis' #'magma'  # or 'viridis', 'plasma', etc.
    )
    plt.colorbar(label="Count")

    plt.xlabel(label)
    plt.ylabel("Epoch")
    plt.title(f"Density Heatmap of {label} by Epoch")

    # Optionally, invert the y-axis if you want epoch=1 on top:
    # plt.gca().invert_yaxis()

    outfile = os.path.join(outdir, f"{label.lower()}_density_heatmap.png")
    plt.tight_layout()
    plt.savefig(outfile, dpi=300)
    plt.close()
    print(f"Density heatmap saved to {outfile}")


def plot_predictions_2D(
    model,
    data_loader_func,
    data_loader_args,
    num_examples=5,
    output_dir="./frames2D",
    t_trained=1200,
):
    """
    Generates and saves model predictions vs. actual data for qualitative evaluation.

    This function plots and saves the model's predictions against the actual temperature distribution data
    for a specified number of examples. For each time step, the actual temperature and the predicted
    temperature are plotted side by side, allowing for a qualitative comparison. The generated images
    are saved in the specified output directory.

    Parameters
    ----------
    model : object
        The trained 2D heat diffusion model to generate predictions.
    data_loader_func : function
        The function that loads the test data in mini-batches. It should accept the necessary
        arguments through `data_loader_args`.
    data_loader_args : dict
        A dictionary containing the arguments to be passed to the `data_loader_func`, including the
        test data, alphas, time steps, batch size, and shuffle option.
    ny : int
        The number of grid points along the y-axis (height of the 2D grid).
    nx : int
        The number of grid points along the x-axis (width of the 2D grid).
    num_examples : int, optional
        The number of examples to plot. Defaults to 5.
    output_dir : str, optional
        The directory where the generated images will be saved. Defaults to "./frames2D".
    t_trained : int, optional
        The time step until which the model was trained. Time steps beyond this value will be
        marked as extrapolated in the plot title. Defaults to 1200.

    Returns
    -------
    None
        The function generates and saves comparison plots of actual vs. predicted temperature distributions.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(output_dir)

    for i, (src, target, test_alphas, test_dts) in enumerate(
        data_loader_func(**data_loader_args)
    ):
        if i >= num_examples:
            break

        prediction = model(src, test_alphas)  # .reshape(-1, src.shape[1], ny, nx)

        for t in range(target.shape[1]):
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            actual_temp = src[0, t, :, :]
            im0 = axs[0].imshow(
                actual_temp, cmap="coolwarm", interpolation="nearest", vmin=0, vmax=1
            )
            axs[0].set_title(f"Actual Temp, Time Step {t}")
            fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

            predicted_temp = prediction[0, t, :, :]
            im1 = axs[1].imshow(
                predicted_temp, cmap="coolwarm", interpolation="nearest", vmin=0, vmax=1
            )
            axs[1].set_title(f"Predicted Temp, Time Step {t}")
            fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

            plt.suptitle(
                f"Example {i + 1}: alpha {test_alphas[0].item():.5f}"
                + (", Extrapolated" if t_trained is not None and t > t_trained else "")
            )
            plt.tight_layout()

            frame_filename = os.path.join(output_dir, f"example_{i + 1}_step_{t}.png")
            plt.savefig(frame_filename)
            # plt.show()

            plt.close(fig)
        print(f"Created heatmaps for Test Example: {i + 1}")


def plot_regressive_predictions_2D(
    model,
    data_loader_func,
    data_loader_args,
    n_replace,
    n_initial,
    num_examples=5,
    output_dir="./frames2D_regress",
    t_trained=1200,
):
    """
    Evaluate the model in a self-regressive manner and plot predictions.
    """
    num_batches = 0
    num_plots = 0
    t_trained = None
    np.set_printoptions(threshold=np.inf)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # data_loader = data_loader_func(test_data, test_alphas, test_dts, batch_size, shuffle=False)

    for i, (src, target, test_alphas, test_dts) in enumerate(
        data_loader_func(**data_loader_args)
    ):
        if i >= num_examples:
            break
        for t in range(n_initial, target.shape[1], n_replace):
            prediction = model(src, test_alphas)
            end_idx = min(t + n_replace, target.shape[1])
            src[:, t:end_idx, :, :] = prediction[:, t:end_idx, :, :]

        num_batches += 1

        for t in range(target.shape[1]):
            fig, axs = plt.subplots(1, 2, figsize=(12, 6))
            actual_temp = target[0, t, :, :]
            im0 = axs[0].imshow(
                actual_temp, cmap="coolwarm", interpolation="nearest", vmin=0, vmax=1
            )
            axs[0].set_title(f"Actual Temp, Time Step {t + 1}")
            fig.colorbar(im0, ax=axs[0], fraction=0.046, pad=0.04)

            predicted_temp = prediction[0, t, :, :]
            im1 = axs[1].imshow(
                predicted_temp, cmap="coolwarm", interpolation="nearest", vmin=0, vmax=1
            )
            axs[1].set_title(f"Predicted Temp, Time Step {t + 1}")
            fig.colorbar(im1, ax=axs[1], fraction=0.046, pad=0.04)

            plt.suptitle(
                f"Example {num_plots + 1}: alpha {test_alphas[0].item():.5f}"
                + (
                    ", !Forward Extrapolation!"
                    if t_trained is not None and t > t_trained - 1
                    else ""
                )
            )
            plt.tight_layout()

            frame_filename = os.path.join(
                output_dir, f"example_{num_plots + 1}_step_{t + 1}.png"
            )
            plt.savefig(frame_filename)
            plt.close(fig)
        num_plots += 1
        print(f"Created heatmaps for Test Example: {num_plots}")


def plot_positional_encodings(model, seq_len):
    """
    Plots the positional encodings for the spatial (y, x) and temporal dimensions from the model.

    This function visualizes the positional encodings learned by the model for the y-axis, x-axis,
    and time dimension (temporal). The positional encodings are summed across the embedding dimension
    and plotted as heatmaps to show how the encodings change across the respective dimensions.

    Parameters
    ----------
    model : object
        The trained 2D heat diffusion model, which contains positional encodings for the spatial
        (y and x axes) and temporal (time steps) dimensions.
    seq_len : int
        The total number of time steps (sequence length) for the temporal positional encoding.

    Returns
    -------
    None
        The function generates and displays plots of the positional encodings for the y-axis, x-axis,
        and temporal dimensions.
    """
    ny, nx = model.ny, model.nx
    # Generate positional indices for the full range
    pos_indices_y = mx.arange(ny).reshape(1, -1)  # Shape: (1, ny)
    pos_indices_x = mx.arange(nx).reshape(-1, 1)  # Shape: (nx, 1)
    pos_indices_t = mx.arange(seq_len).reshape(1, -1)  # Shape: (1, seq_len)

    # Get positional encodings from the model
    pos_enc_y = np.array(
        model.positional_encoding_y(pos_indices_y)
    )  # Should be (1, ny, embed_dim)
    pos_enc_x = np.array(
        model.positional_encoding_x(pos_indices_x)
    )  # Should be (nx, 1, embed_dim)
    pos_enc_t = np.array(
        model.positional_encoding_t(pos_indices_t)
    )  # Should be (1, seq_len, embed_dim)

    # Sum across embedding dimension to reduce to 2D
    pos_enc_y_sum = pos_enc_y.sum(axis=2)  # Sum over embedding dimension
    pos_enc_x_sum = pos_enc_x.sum(axis=2)
    pos_enc_t_sum = pos_enc_t.sum(axis=2)  # Sum over embedding dimension

    import matplotlib.pyplot as plt

    plt.figure(figsize=(21, 7))
    plt.subplot(1, 3, 1)
    plt.title("Positional Encoding Y")
    plt.imshow(pos_enc_y_sum, aspect="auto", cmap="viridis")
    plt.colorbar()

    plt.subplot(1, 3, 2)
    plt.title("Positional Encoding X")
    plt.imshow(
        pos_enc_x_sum.T, aspect="auto", cmap="viridis"
    )  # Transposed to align dimensions
    plt.colorbar()

    plt.subplot(1, 3, 3)
    plt.title("Temporal Positional Encoding")
    plt.imshow(pos_enc_t_sum, aspect="auto", cmap="viridis")
    plt.colorbar()

    plt.show()


def plot_model_weights(model, epoch):
    """
    Plots statistics of the model's weights and saves the plots.

    This function extracts, processes, and visualizes the model's weight statistics. It filters the model's weights,
    applies transformations (e.g., mean, max, count of weights above a threshold), and plots the results as heatmaps.
    The plots are saved as PNG files with the epoch number included in the filename.

    Parameters
    ----------
    model : object
        The trained model whose weights will be extracted and visualized. The model should contain an
        'output_projection' layer from which the weights are visualized.
    epoch : int
        The epoch number used to label the saved plots.

    Returns
    -------
    None
        The function generates and saves heatmaps of the weight statistics (mean, max, and counts above thresholds)
        but does not return any value.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Set the model to evaluation mode
    model.eval()

    # Helper function to check if a value is an MLX array (adjust according to actual MLX type)
    def is_mlx_array(value):
        # Replace 'mx.array' with the actual array type in the MLX framework
        return isinstance(value, mx.array)

    # Function to filter the model's weights based on a condition
    def filter_fn(module, key, value):
        # Check if the key contains 'weight' and is a valid MLX array
        if isinstance(value, dict) and "weight" in value:
            weight = value["weight"]
            if is_mlx_array(weight):
                mean_abs = weight.abs().mean()
                return mean_abs > 0.003  # Filter out small weights
        elif is_mlx_array(value):
            mean_abs = value.abs().mean()
            return mean_abs > 0.003
        return False

    # Function to modify the weights during filtering
    def map_fn(value):
        if is_mlx_array(value):
            return value * 10  # Example scaling
        return value

    # Filter and modify the model's weights
    filtered_weights = model.filter_and_map(filter_fn, map_fn)

    # Retrieve weights from the 'output_projection' layer
    weights_dict = filtered_weights.get("output_projection", None)
    if weights_dict is not None:
        weights = weights_dict.get("weight", None)
        if weights is not None:
            # Reshape weights if needed
            if weights.ndim == 1:
                weights = weights.reshape(1, -1)

            # Calculate absolute values of weights
            weights_abs = mx.abs(weights)
            weights_abs_reshaped = weights_abs.reshape(26, 26, 512)

            # Calculate mean, and max across the embedding dimension
            mean_data = mx.mean(weights_abs_reshaped, axis=2)
            max_data = mx.max(weights_abs_reshaped, axis=2)

            # Threshold for significant weights
            threshold = 0.75 * max_data
            count_above_threshold = mx.sum(
                weights_abs_reshaped > threshold[:, :, None], axis=2
            )
            count_above_mean = mx.sum(
                weights_abs_reshaped > mean_data[:, :, None], axis=2
            )

            # Plot the results as heatmaps
            fig, axes = plt.subplots(2, 2, figsize=(20, 20))
            xticks = np.arange(0, 26, 2)
            yticks = np.arange(0, 26, 2)

            # Heatmap for the mean of absolute weights
            sns.heatmap(
                np.array(mean_data),
                ax=axes[0, 0],
                cmap="viridis",
                square=True,
                cbar_ax=fig.add_axes([0.48, 0.53, 0.02, 0.35]),
            )
            axes[0, 0].set_title("Mean of Abs Weights")
            axes[0, 0].set_xticks(xticks)
            axes[0, 0].set_yticks(yticks)

            # Heatmap for the max of absolute weights
            sns.heatmap(
                np.array(max_data),
                ax=axes[0, 1],
                cmap="viridis",
                square=True,
                cbar_ax=fig.add_axes([0.903, 0.53, 0.02, 0.35]),
            )
            axes[0, 1].set_title("Max of Abs Weights")
            axes[0, 1].set_xticks(xticks)
            axes[0, 1].set_yticks(yticks)

            # Heatmap for count above the mean
            sns.heatmap(
                np.array(count_above_mean),
                ax=axes[1, 0],
                cmap="viridis",
                square=True,
                cbar_ax=fig.add_axes([0.48, 0.11, 0.015, 0.35]),
            )
            axes[1, 0].set_title("Count Above Mean Value")
            axes[1, 0].set_xticks(xticks)
            axes[1, 0].set_yticks(yticks)

            # Heatmap for count above 0.75 * max value
            sns.heatmap(
                np.array(count_above_threshold),
                ax=axes[1, 1],
                cmap="viridis",
                square=True,
                cbar_ax=fig.add_axes([0.903, 0.11, 0.015, 0.35]),
            )
            axes[1, 1].set_title("Count Above 0.75 * Max Value")
            axes[1, 1].set_xticks(xticks)
            axes[1, 1].set_yticks(yticks)

            # Save the plots to file
            frame_dir = os.path.join(
                os.path.dirname(__file__), "Base_Block_MPI_noGradAve_weights"
            )
            os.makedirs(frame_dir, exist_ok=True)

            # Correct the frame filename
            frame_filename = os.path.join(frame_dir, f"epoch_{epoch}.png")
            plt.savefig(frame_filename)
            plt.close(fig)
        else:
            print("No weights found for output_projection")
    else:
        print("No output_projection key found in filtered weights")


def plot_mse_evolution(mse: np.ndarray, output_dir: str, label="block"):
    os.makedirs(output_dir, exist_ok=True)
    steps = np.arange(5, 5 + len(mse))
    plt.figure(figsize=(10, 6))
    plt.plot(steps, mse, marker="o")
    plt.xlabel("Time step")
    plt.ylabel("MSE")
    plt.title("Average MSE evolution")
    path = os.path.join(output_dir, f"mse_evolution_{label}.png")
    plt.grid(True)
    plt.savefig(path)
    plt.close()
    print("Saved plot ->", path)
