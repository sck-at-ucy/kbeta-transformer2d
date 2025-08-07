# transformer/data.py
"""file start: data.py
Dataset creation and loaders (moved from Testing_Kourkoutasb.py)"""

from __future__ import annotations

import os

import mlx.core as mx
import numpy as np

# ── all the functions that were purely data‑related ───────────────────────
__all__ = [
    "initialize_geometry_and_bcs",
    "generate_datasets",
    "data_loader_2D",
    "save_datasets",
    "load_datasets",
]


def initialize_geometry_and_bcs(config):
    """
    Initializes the geometry and boundary conditions for the 2D temperature distribution simulation.

    This function calculates the number of grid points in the x and y directions (nx, ny) based on the
    specified rod length, width, and spatial steps. It also generates boundary conditions and thermal diffusivity
    values for training, validation, and testing by splitting them according to the provided limits in the configuration.

    Parameters
    ----------
    config : dict
        Dictionary containing configuration for the geometry, boundary conditions, and thermal diffusivity.
        Example structure:
        config = {
            "geometry": {
                "rod_length": float,  # Length of the rod
                "rod_width": float,   # Width of the rod
                "dx": float,          # Spatial step in the x direction
                "dy": float           # Spatial step in the y direction
            },
            "boundary_conditions": {
                "left_limits": tuple,  # Boundary conditions on the left side
                "right_limits": tuple, # Boundary conditions on the right side
                "top_limits": tuple,   # Boundary conditions on the top side
                "bottom_limits": tuple # Boundary conditions on the bottom side
            },
            "thermal_diffusivity": {
                "alpha_limits": tuple  # Limits for the thermal diffusivity values
            },
            "training_samples": int  # Number of samples to generate
        }

    Returns
    -------
    tuple
        A tuple containing:
        - nx : int
            Number of grid points along the x-axis.
        - ny : int
            Number of grid points along the y-axis.
        - training_bcs : list
            Boundary conditions for the training dataset.
        - validation_bcs : list
            Boundary conditions for the validation dataset.
        - test_bcs : list
            Boundary conditions for the test dataset.
        - training_alphas : list or numpy.ndarray
            Thermal diffusivity values for the training dataset.
        - validation_alphas : list or numpy.ndarray
            Thermal diffusivity values for the validation dataset.
        - test_alphas : list or numpy.ndarray
            Thermal diffusivity values for the test dataset.
    """
    geom = config["geometry"]
    bcs = config["boundary_conditions"]
    alphas = config["thermal_diffusivity"]

    # Calculate derived parameters
    nx = int(geom["rod_length"] / geom["dx"]) + 1
    ny = int(geom["rod_width"] / geom["dy"]) + 1

    # Generate boundary conditions
    (
        training_bcs,
        validation_bcs,
        test_bcs,
        training_alphas,
        validation_alphas,
        test_alphas,
    ) = generate_bcs_and_split_2D(
        config["training_samples"],
        bcs["left_limits"],
        bcs["right_limits"],
        bcs["top_limits"],
        bcs["bottom_limits"],
        alphas["alpha_limits"],
    )

    return (
        nx,
        ny,
        training_bcs,
        validation_bcs,
        test_bcs,
        training_alphas,
        validation_alphas,
        test_alphas,
    )


# Refactored MLX version
@mx.compile
def generate_bcs_and_split_2D(
    num_samples,
    left_limits,
    right_limits,
    top_limits,
    bottom_limits,
    alpha_limits,
    synchronized_shuffling=False,
):
    left_bcs = mx.linspace(left_limits[0], left_limits[1], num_samples)
    right_bcs = mx.linspace(right_limits[0], right_limits[1], num_samples)
    top_bcs = mx.linspace(top_limits[0], top_limits[1], num_samples)
    bottom_bcs = mx.linspace(bottom_limits[0], bottom_limits[1], num_samples)
    alphas = mx.linspace(alpha_limits[0], alpha_limits[1], num_samples)

    """
    if synchronized_shuffling:
        # Generate random floating-point numbers uniformly distributed between 0 and 1
        random_floats = mx.random.uniform(low=0, high=1, shape=(left_bcs.shape[0],))

        # Use argsort to get a random permutation of indices based on the random floats
        indices = mx.argsort(random_floats)

        # Apply this random permutation to all the arrays in sync
        left_bcs, right_bcs, top_bcs, bottom_bcs, alphas = [array[indices] for array in
                                                            [left_bcs, right_bcs, top_bcs, bottom_bcs, alphas]]
    else:
        # Independent random shuffles (with different random permutations for each array)
        indicesL = mx.argsort(mx.random.uniform(low=0, high=1, shape=(left_bcs.shape[0],)))
        indicesR = mx.argsort(mx.random.uniform(low=0, high=1, shape=(right_bcs.shape[0],)))
        indicesT = mx.argsort(mx.random.uniform(low=0, high=1, shape=(top_bcs.shape[0],)))
        indicesB = mx.argsort(mx.random.uniform(low=0, high=1, shape=(bottom_bcs.shape[0],)))
        indicesA = mx.argsort(mx.random.uniform(low=0, high=1, shape=(alphas.shape[0],)))

        # Apply the independent random permutations to each array
        left_bcs = left_bcs[indicesL]
        right_bcs = right_bcs[indicesR]
        top_bcs = top_bcs[indicesT]
        bottom_bcs = bottom_bcs[indicesB]
        alphas = alphas[indicesA]
    """

    arrs = [left_bcs, right_bcs, top_bcs, bottom_bcs, alphas]
    if synchronized_shuffling:
        indices = mx.random.permutation(left_bcs.shape[0])
        arrs = [a[indices] for a in arrs]
    else:
        arrs = map(mx.random.permutation, arrs)
        # Alternative form: arrs_comprehension = [mx.random.permutation(a) for a in arrs]

    left_bcs, right_bcs, top_bcs, bottom_bcs, alphas = arrs

    # Splitting data
    training_left = left_bcs[: int(0.7 * num_samples)]
    validation_left = left_bcs[int(0.7 * num_samples) : int(0.9 * num_samples)]
    test_left = left_bcs[int(0.9 * num_samples) :]

    training_right = right_bcs[: int(0.7 * num_samples)]
    validation_right = right_bcs[int(0.7 * num_samples) : int(0.9 * num_samples)]
    test_right = right_bcs[int(0.9 * num_samples) :]

    training_top = top_bcs[: int(0.7 * num_samples)]
    validation_top = top_bcs[int(0.7 * num_samples) : int(0.9 * num_samples)]
    test_top = top_bcs[int(0.9 * num_samples) :]

    training_bottom = bottom_bcs[: int(0.7 * num_samples)]
    validation_bottom = bottom_bcs[int(0.7 * num_samples) : int(0.9 * num_samples)]
    test_bottom = bottom_bcs[int(0.9 * num_samples) :]

    training_alphas = alphas[: int(0.7 * num_samples)]
    validation_alphas = alphas[int(0.7 * num_samples) : int(0.9 * num_samples)]
    test_alphas = alphas[int(0.9 * num_samples) :]

    return (
        (training_left, training_right, training_top, training_bottom),
        (validation_left, validation_right, validation_top, validation_bottom),
        (test_left, test_right, test_top, test_bottom),
        training_alphas,
        validation_alphas,
        test_alphas,
    )


def generate_heat_data_2D(
    rod_length,
    rod_width,
    dx,
    dy,
    time_steps,
    left_bcs,
    right_bcs,
    top_bcs,
    bottom_bcs,
    alphas,
    boundary_segment_strategy="base_case",
):
    num_samples = left_bcs.shape[0]
    nx = int(rod_length / dx) + 1
    ny = int(rod_width / dy) + 1

    # Initialize solutions using MLX arrays
    solutions = mx.zeros((num_samples, time_steps, ny, nx))
    dts = dx**2 / (10 * alphas)  # Vectorized computation of time steps for each sample

    # Initialize temperature array with random values for each sample
    random_values = mx.random.uniform(low=0, high=1, shape=(num_samples,))
    random_values_reshaped = random_values[
        :, None, None
    ]  # Shape becomes (num_samples, 1, 1)
    T = mx.full((num_samples, ny, nx), 1.0) * random_values_reshaped

    # Apply boundary conditions
    T[:, :, -1] = right_bcs[:, None]  # Right boundary
    T[:, :, 0] = left_bcs[:, None]  # Left boundary
    T[:, -1, :] = bottom_bcs[:, None]  # Bottom boundary
    T[:, 0, :] = top_bcs[:, None]  # Top boundary

    # Handle boundary segment strategy
    if boundary_segment_strategy == "challenge_1":
        side1, side2 = 0, 1
        pos1, pos2 = 8, 8  # Set positions for segments
        apply_fixed_segments(T, side1, side2, pos1, pos2)
    elif boundary_segment_strategy == "challenge_2":
        # Handle random segments for each sample
        random_floats = mx.random.uniform(low=0, high=1, shape=(num_samples, 4))
        side_indices = mx.argsort(random_floats, axis=1)
        side1 = side_indices[:, 0]
        side2 = side_indices[:, 1]
        # Prepare arrays for vectorized indexing
        start_positions_x = mx.random.randint(
            low=0, high=(nx - 4), shape=(num_samples,)
        )
        start_positions_y = mx.random.randint(
            low=0, high=(ny - 4), shape=(num_samples,)
        )
        place_segments_vectorized(T, side1, start_positions_x, 1, ny, nx)
        place_segments_vectorized(T, side2, start_positions_y, 0, ny, nx)

    # Store initial condition
    solutions[:, 0, :, :] = T

    # Time-stepping loop, vectorized
    for t in range(1, time_steps):
        d2T_dx2, d2T_dy2 = calculate_spatial_derivative_2D_initial(T, dx, dy)

        # Update only the inner region (1:-1) of the temperature array
        T_new = mx.array(T)
        T_new[:, 1:-1, 1:-1] = T[:, 1:-1, 1:-1] + alphas[:, None, None] * dts[
            :, None, None
        ] * (d2T_dx2 + d2T_dy2)

        # Reapply boundary conditions
        T_new[:, :, -1] = right_bcs[:, None]  # Right boundary
        T_new[:, :, 0] = left_bcs[:, None]  # Left boundary
        T_new[:, -1, :] = bottom_bcs[:, None]  # Bottom boundary
        T_new[:, 0, :] = top_bcs[:, None]  # Top boundary

        # Reapply the boundary segments based on the strategy (if in Challenge configuration)
        if boundary_segment_strategy == "challenge_1":
            apply_fixed_segments(T_new, side1, side2, pos1, pos2)
        elif boundary_segment_strategy == "challenge_2":
            place_segments_vectorized(T_new, side1, start_positions_x, 1)
            place_segments_vectorized(T_new, side2, start_positions_y, 0)

        # Store updated temperature for the current time step
        solutions[:, t, :, :] = T_new
        T = T_new  # Move to the next time step

    return solutions, alphas, dts


def place_segments_vectorized(T, side, start_positions, value, ny, nx):
    arange_segment = np.arange(4)

    # Convert start_positions to MLX array
    start_positions = mx.array(start_positions)

    # Use NumPy for boolean mask indexing
    side_np = np.array(side)  # Convert MLX array to NumPy array
    mask_top_indices = np.where(side_np == 0)[0]
    mask_bottom_indices = np.where(side_np == 1)[0]
    mask_left_indices = np.where(side_np == 2)[0]
    mask_right_indices = np.where(side_np == 3)[0]

    # Convert the NumPy indices back to MLX arrays
    mask_top_indices = mx.array(mask_top_indices)
    mask_bottom_indices = mx.array(mask_bottom_indices)
    mask_left_indices = mx.array(mask_left_indices)
    mask_right_indices = mx.array(mask_right_indices)

    # Top side
    if mask_top_indices.shape[0] > 0:
        cols_top = (
            start_positions[mask_top_indices][:, None] + arange_segment
        )  # (5968, 4)
        for i in range(cols_top.shape[1]):  # Loop over segment length
            T[mask_top_indices, 0, cols_top[:, i]] = value  # Update each segment column

    # Bottom side
    if mask_bottom_indices.shape[0] > 0:
        cols_bottom = (
            start_positions[mask_bottom_indices][:, None] + arange_segment
        )  # (5968, 4)
        for i in range(cols_bottom.shape[1]):  # Loop over segment length
            T[mask_bottom_indices, ny - 1, cols_bottom[:, i]] = (
                value  # Update each segment column
            )

    # Left side
    if mask_left_indices.shape[0] > 0:
        rows_left = (
            start_positions[mask_left_indices][:, None] + arange_segment
        )  # (5968, 4)
        for i in range(rows_left.shape[1]):  # Loop over segment length
            T[mask_left_indices, rows_left[:, i], 0] = value  # Update each segment row

    # Right side
    if mask_right_indices.shape[0] > 0:
        rows_right = (
            start_positions[mask_right_indices][:, None] + arange_segment
        )  # (5968, 4)
        for i in range(rows_right.shape[1]):  # Loop over segment length
            T[mask_right_indices, rows_right[:, i], nx - 1] = (
                value  # Update each segment row
            )

    return T


def apply_fixed_segments(T, side1, side2, pos1, pos2):
    """
    Apply fixed boundary segments using MLX array operations for Challenge 1.
    """
    if side1 == 0:
        T[:, pos1 : pos1 + 4, 0] = 1.0  # Apply left boundary
    if side2 == 1:
        T[:, pos2 : pos2 + 4, -1] = 0.0  # Apply right boundary
    return T


# Dataset generation for MLX
def generate_datasets(
    config,
    training_bcs,
    validation_bcs,
    test_bcs,
    training_alphas,
    validation_alphas,
    test_alphas,
):
    geom = config["geometry"]
    model_params = config["model_params"]
    strategy = config["boundary_segment_strategy"]  # <‑ one name everywhere

    # ---- training set ---------------------------------------------------
    train_data, train_alphas_out, train_dts = generate_heat_data_2D(
        geom["rod_length"],
        geom["rod_width"],
        geom["dx"],
        geom["dy"],
        model_params["time_steps"],
        *training_bcs,  # expands to left / right / top / bottom
        training_alphas,  # diffusivities for this split
        strategy,
    )

    # ---- validation set --------------------------------------------------
    val_data, val_alphas_out, val_dts = generate_heat_data_2D(
        geom["rod_length"],
        geom["rod_width"],
        geom["dx"],
        geom["dy"],
        model_params["time_steps"],
        *validation_bcs,
        validation_alphas,
        strategy,
    )

    # ---- test set --------------------------------------------------------
    test_data, test_alphas_out, test_dts = generate_heat_data_2D(
        geom["rod_length"],
        geom["rod_width"],
        geom["dx"],
        geom["dy"],
        model_params["time_steps"],
        *test_bcs,
        test_alphas,
        strategy,
    )

    return (
        train_data,
        train_alphas_out,
        train_dts,
        val_data,
        val_alphas_out,
        val_dts,
        test_data,
        test_alphas_out,
        test_dts,
    )


def data_loader_2D(data, alphas, solution_dts, batch_size, shuffle=True):
    """
    Data loader function to create mini-batches from the generated temperature distribution data for training.

    This function takes the generated temperature distribution data and splits it into mini-batches.
    It yields the source data (inputs), target data (outputs), thermal diffusivities (alphas),
    and time steps (dts) for each batch. The data can optionally be shuffled to ensure randomness in
    the mini-batches during training.

    Parameters
    ----------
    data : numpy.ndarray or mlx.core.array
        The 4D array representing the generated temperature distribution data, with shape
        (num_samples, time_steps, ny, nx).
    alphas : numpy.ndarray or mlx.core.array
        1D array of thermal diffusivity values (alphas) for each sample.
    solution_dts : numpy.ndarray or mlx.core.array
        1D array of time step sizes (dts) for each sample.
    batch_size : int
        The number of samples per batch.
    shuffle : bool, optional
        If True, the data will be shuffled before creating mini-batches. Defaults to True.

    Yields
    ------
    tuple
        A tuple containing:
        - src_tensor : mlx.core.array
            The source input tensor for the batch.
        - target_tensor : mlx.core.array
            The target output tensor for the batch.
        - batch_alphas_tensor : mlx.core.array
            The thermal diffusivities (alphas) for the batch.
        - batch_dts_tensor : mlx.core.array
            The time step sizes (dts) for the batch.
    """
    num_samples = data.shape[0]

    # Create indices as an MLX array instead of using NumPy
    indices = mx.arange(num_samples)

    if shuffle:
        # Shuffle using MLX's permutation function for a random permutation of indices
        indices = mx.random.permutation(indices)

    for start_idx in range(0, num_samples, batch_size):
        end_idx = min(start_idx + batch_size, num_samples)
        batch_indices = indices[start_idx:end_idx]

        # Index the data and convert to MLX tensors
        batch_data = data[batch_indices, :, :, :]
        batch_alphas = alphas[batch_indices]
        batch_dts = solution_dts[batch_indices]

        src_tensor = batch_data
        target_tensor = batch_data  # Assuming target is the same as input in this case
        batch_alphas_tensor = batch_alphas
        batch_dts_tensor = batch_dts

        yield src_tensor, target_tensor, batch_alphas_tensor, batch_dts_tensor


@mx.compile
def calculate_spatial_derivative_2D_initial(T, dx, dy):
    """
    Vectorized calculation of second-order spatial derivatives using MLX's array operations.
    This avoids explicit loops.
    """
    d2T_dx2 = (T[:, 1:-1, 2:] - 2 * T[:, 1:-1, 1:-1] + T[:, 1:-1, :-2]) / dx**2
    d2T_dy2 = (T[:, 2:, 1:-1] - 2 * T[:, 1:-1, 1:-1] + T[:, :-2, 1:-1]) / dy**2
    return d2T_dx2, d2T_dy2


def save_datasets(
    train_data,
    train_alphas,
    train_dts,
    val_data,
    val_alphas,
    val_dts,
    test_data,
    test_alphas,
    test_dts,
    dir_path,
):
    """
    Saves the training, validation, and test datasets to the specified directory.

    This function stores the training, validation, and test datasets, along with their respective thermal
    diffusivities and time steps, into the provided directory path. The data is saved in `.npy` format using
    the MLX `mx.save` utility.

    Parameters
    ----------
    train_data : mlx.core.array
        Training dataset representing the temperature distribution field over time.
    train_alphas : mlx.core.array
        Thermal diffusivity values for the training dataset.
    train_dts : mlx.core.array
        Time steps for the training dataset.
    val_data : mlx.core.array
        Validation dataset representing the temperature distribution field over time.
    val_alphas : mlx.core.array
        Thermal diffusivity values for the validation dataset.
    val_dts : mlx.core.array
        Time steps for the validation dataset.
    test_data : mlx.core.array
        Test dataset representing the temperature distribution field over time.
    test_alphas : mlx.core.array
        Thermal diffusivity values for the test dataset.
    test_dts : mlx.core.array
        Time steps for the test dataset.
    dir_path : str
        Directory path where the datasets will be saved.

    Returns
    -------
    None
        The datasets are saved to the specified directory.
    """
    os.makedirs(dir_path, exist_ok=True)
    mx.save(os.path.join(dir_path, "train_data"), train_data)
    mx.save(os.path.join(dir_path, "train_alphas"), train_alphas)
    mx.save(os.path.join(dir_path, "train_dts"), train_dts)
    mx.save(os.path.join(dir_path, "val_data"), val_data)
    mx.save(os.path.join(dir_path, "val_alphas"), val_alphas)
    mx.save(os.path.join(dir_path, "val_dts"), val_dts)
    mx.save(os.path.join(dir_path, "test_data"), test_data)
    mx.save(os.path.join(dir_path, "test_alphas"), test_alphas)
    mx.save(os.path.join(dir_path, "test_dts"), test_dts)


def load_datasets(dir_path):
    """
    Loads the training, validation, and test datasets from the specified directory.

    This function retrieves previously saved training, validation, and test datasets, along with their thermal
    diffusivities and time steps, from `.npy` files. The data is loaded using the MLX `mx.load` utility.

    Parameters
    ----------
    dir_path : str
        Directory path where the datasets are stored.

    Returns
    -------
    tuple of (mlx.core.array, mlx.core.array, mlx.core.array, mlx.core.array, mlx.core.array, mlx.core.array, mlx.core.array, mlx.core.array, mlx.core.array)
        - train_data : mlx.core.array
            Training dataset representing the temperature distribution field over time.
        - train_alphas : mlx.core.array
            Thermal diffusivity values for the training dataset.
        - train_dts : mlx.core.array
            Time steps for the training dataset.
        - val_data : mlx.core.array
            Validation dataset representing the temperature distribution field over time.
        - val_alphas : mlx.core.array
            Thermal diffusivity values for the validation dataset.
        - val_dts : mlx.core.array
            Time steps for the validation dataset.
        - test_data : mlx.core.array
            Test dataset representing the temperature distribution field over time.
        - test_alphas : mlx.core.array
            Thermal diffusivity values for the test dataset.
        - test_dts : mlx.core.array
            Time steps for the test dataset.
    """
    train_data = mx.load(os.path.join(dir_path, "train_data.npy"))
    train_alphas = mx.load(os.path.join(dir_path, "train_alphas.npy"))
    train_dts = mx.load(os.path.join(dir_path, "train_dts.npy"))
    val_data = mx.load(os.path.join(dir_path, "val_data.npy"))
    val_alphas = mx.load(os.path.join(dir_path, "val_alphas.npy"))
    val_dts = mx.load(os.path.join(dir_path, "val_dts.npy"))
    test_data = mx.load(os.path.join(dir_path, "test_data.npy"))
    test_alphas = mx.load(os.path.join(dir_path, "test_alphas.npy"))
    test_dts = mx.load(os.path.join(dir_path, "test_dts.npy"))
    return (
        train_data,
        train_alphas,
        train_dts,
        val_data,
        val_alphas,
        val_dts,
        test_data,
        test_alphas,
        test_dts,
    )
