# transformer/model.py"
"""HeatDiffusionModel & losses"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

__all__ = [
    "HeatDiffusionModel",
    "loss_fn_2D",
    "physics_informed_loss_2D",
    "compute_boundary_loss_2D",
    "compute_initial_loss_2D",
]


class HeatDiffusionModel(nn.Module):
    def __init__(
        self,
        ny,
        nx,
        seq_len,
        num_heads,
        num_encoder_layers,
        mlp_dim,
        embed_dim,
        start_predicting_from,
        mask_type,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.output_seq_len = seq_len
        self.ny = ny
        self.nx = nx
        self.input_dim = ny * nx
        self.embed_dim = embed_dim
        self.spatial_features = embed_dim // 2
        self._start_predicting_from = start_predicting_from
        self._mask_type = mask_type

        # QAT-compatible spatial encoding layers
        self.projection_spatial_enc = nn.QuantizedLinear(
            ny * nx * self.spatial_features, self.embed_dim
        )

        self.positional_encoding_y = nn.QuantizedEmbedding(ny, self.spatial_features)
        self.positional_encoding_x = nn.QuantizedEmbedding(nx, self.spatial_features)
        self.positional_encoding_t = nn.QuantizedEmbedding(seq_len, self.embed_dim)

        # Transformer encoder with quantized layers
        self.transformer_encoder = nn.TransformerEncoder(
            num_layers=num_encoder_layers,
            dims=embed_dim,
            num_heads=num_heads,
            mlp_dims=mlp_dim,
            checkpoint=False,
        )

        self.output_projection = nn.Linear(embed_dim, ny * nx)
        self.diffusivity_embedding = nn.Linear(1, embed_dim)
        self.layer_normalizer = nn.LayerNorm(dims=embed_dim)

        if self._mask_type == "causal":
            self.mask = self.create_src_causal_mask(self.seq_len)
        elif self._mask_type == "block":
            self.mask = self.create_src_block_mask(self.seq_len)
        else:
            raise ValueError("Unsupported mask type")

    def create_src_block_mask(self, seq_len):
        mask = mx.full((seq_len, seq_len), -mx.inf, dtype=mx.float32)
        mask[:, : self._start_predicting_from] = 0
        return mask

    def create_src_causal_mask(self, seq_len):
        mask = mx.triu(-mx.inf * mx.ones((seq_len, seq_len)), k=0)
        mask[:, : self._start_predicting_from] = 0
        return mask

    def spatial_positional_encoding(self):
        # Generate positional encodings for y and x dimensions
        ny_encoding = self.positional_encoding_y(mx.arange(self.ny))
        nx_encoding = self.positional_encoding_x(mx.arange(self.nx))

        # Apply expand_dims correctly with mx.expand_dims
        ny_encoding = mx.expand_dims(
            mx.expand_dims(ny_encoding, axis=0), axis=2
        )  # Shape: [1, ny, 1, spatial_features]
        nx_encoding = mx.expand_dims(
            mx.expand_dims(nx_encoding, axis=0), axis=1
        )  # Shape: [1, 1, nx, spatial_features]

        # Apply RoPE with the required parameters on expanded dimensions
        ny_encoding = mx.fast.rope(
            ny_encoding,
            dims=self.spatial_features,
            traditional=True,
            base=100,
            scale=1,  # / np.sqrt(self.spatial_features / 2),
            offset=0,
        )
        nx_encoding = mx.fast.rope(
            nx_encoding,
            dims=self.spatial_features,
            traditional=True,
            base=100,
            scale=1,  # / np.sqrt(self.spatial_features / 2),
            offset=0,
        )

        return ny_encoding, nx_encoding

    def temporal_positional_encoding(self, batch_size):
        # Generate temporal encoding and ensure it has 3 dimensions before RoPE
        temporal_encoding = self.positional_encoding_t(mx.arange(self.seq_len))

        # Expand dimensions using mx.expand_dims to match RoPE's requirements
        temporal_encoding = mx.expand_dims(
            temporal_encoding, axis=0
        )  # Shape: [1, seq_len, embed_dim]

        # Apply RoPE with the required parameters on the expanded temporal encoding
        temporal_encoding = mx.fast.rope(
            temporal_encoding,
            dims=self.embed_dim,
            traditional=True,
            base=100,
            scale=1,  # / np.sqrt(self.embed_dim / 2),
            offset=0,
        )

        return temporal_encoding

    def __call__(self, src, alpha):
        batch_size, seq_len, _, _ = src.shape
        src_unflattened = src[:, :, :]
        src_expanded = mx.expand_dims(src_unflattened, -1)
        pos_enc_ny, pos_enc_nx = self.spatial_positional_encoding()
        src_pos_enc_y = src_expanded + pos_enc_ny
        src_pos_enc = src_pos_enc_y + pos_enc_nx
        src_pos_enc_flattened = src_pos_enc[:, :, :, :].reshape(
            -1, seq_len, self.ny * self.nx * self.spatial_features
        )
        src_projected = self.projection_spatial_enc(src_pos_enc_flattened)

        temporal_enc = self.temporal_positional_encoding(batch_size)
        src_encoded = src_projected + temporal_enc

        alpha_reshaped = alpha.reshape(-1, 1)
        alpha_embed = self.diffusivity_embedding(alpha_reshaped)
        alpha_embed_expanded = mx.expand_dims(alpha_embed, axis=1)
        alpha_embed_expanded = mx.broadcast_to(
            alpha_embed_expanded, (batch_size, seq_len, self.embed_dim)
        )
        src_encoded += alpha_embed_expanded

        encoded = self.transformer_encoder(src_encoded, mask=self.mask)
        normalized = self.layer_normalizer(encoded)
        output = self.output_projection(normalized)

        return output.reshape(batch_size, self.output_seq_len, self.ny, self.nx)


def calculate_spatial_derivative_2D(T, dx, dy):
    """
    Calculates the second-order spatial derivatives of the temperature field in the x and y directions.

    This function computes the second-order central difference approximation of the spatial derivatives
    for the temperature distribution `T` over a 2D grid. The derivatives are calculated using finite
    differences with respect to the spatial steps `dx` (x-axis) and `dy` (y-axis).

    Parameters
    ----------
    T : numpy.ndarray or mlx.core.array
        The 4D array representing the temperature distribution, with shape (batch_size, time_steps, ny, nx), where:
        - batch_size is the number of samples,
        - time_steps is the number of time steps in the simulation,
        - ny is the number of grid points along the y-axis,
        - nx is the number of grid points along the x-axis.
    dx : float
        The spatial step size in the x direction.
    dy : float
        The spatial step size in the y direction.

    Returns
    -------
    tuple of (numpy.ndarray or mlx.core.array, numpy.ndarray or mlx.core.array)
        A tuple containing:
        - d2T_dx2 : numpy.ndarray or mlx.core.array
            The second derivative of `T` with respect to the x direction, with shape (batch_size, time_steps, ny-2, nx-2).
        - d2T_dy2 : numpy.ndarray or mlx.core.array
            The second derivative of `T` with respect to the y direction, with shape (batch_size, time_steps, ny-2, nx-2).
    """
    d2T_dx2 = (T[:, :, 1:-1, 2:] - 2 * T[:, :, 1:-1, 1:-1] + T[:, :, 1:-1, :-2]) / dx**2
    d2T_dy2 = (T[:, :, 2:, 1:-1] - 2 * T[:, :, 1:-1, 1:-1] + T[:, :, :-2, 1:-1]) / dy**2
    return d2T_dx2, d2T_dy2


def calculate_temporal_derivative_2D(T, dt):
    """
    Calculates the temporal derivative of the temperature field over time.

    This function computes the first-order difference approximation of the temporal #derivative
    for the temperature distribution `T` over a 2D grid. The derivative is #calculated using
    finite differences with respect to the time steps `dt`. It assumes that the #first dimension
    of `T` corresponds to time.

    Parameters
    ----------
    T : numpy.ndarray or mlx.core.array
        The 4D array representing the temperature distribution, with shape #(batch_size, time_steps, ny, nx), where:
        - batch_size is the number of samples,
        - time_steps is the number of time steps in the simulation,
        - ny is the number of grid points along the y-axis,
        - nx is the number of grid points along the x-axis.
    dt : numpy.ndarray or mlx.core.array
        1D array of time step sizes for each sample, with shape (batch_size,). Each #element represents
        the time step size for the corresponding sample.

    Returns
    -------
    numpy.ndarray or mlx.core.array
        The temporal derivative of the temperature field `T` with respect to time, #with shape
        (batch_size, time_steps - 1, ny, nx).
    """
    dt_reshaped = dt.reshape(-1, 1, 1, 1)
    dT_dt = (T[:, 1:, :, :] - T[:, :-1, :, :]) / dt_reshaped
    return dT_dt


def physics_informed_loss_2D(model_output, src_alphas, src_dts, dx, dy):
    """
    Computes the physics-informed loss for a 2D heat diffusion model.

    This function calculates the physics-informed loss by enforcing the heat equation on the model's
    output. It aligns the model output to the grid, computes the second-order spatial derivatives
    and the first-order temporal derivative, and then calculates the residuals between the temporal
    derivative and the spatial derivatives scaled by the thermal diffusivity. The loss is the mean
    squared error of these residuals.

    Parameters
    ----------
    model_output : numpy.ndarray or mlx.core.array
        The output from the model, representing the predicted temperature distribution field.
        It has the shape (batch_size, time_steps, ny, nx).
    src_alphas : numpy.ndarray or mlx.core.array
        The thermal diffusivity values (alphas) for each sample, with shape (batch_size,).
    src_dts : numpy.ndarray or mlx.core.array
        The time step sizes (dts) for each sample, with shape (batch_size,).
    dx : float
        The spatial step size in the x direction.
    dy : float
        The spatial step size in the y direction.

    Returns
    -------
    mx.core.array
        The physics-informed loss value, calculated as the mean squared error of the residuals between
        the temporal derivative and the scaled spatial derivatives.
    """
    model_output_pre_aligned = model_output[:, :, 1:-1, 1:-1]

    d2T_dx2, d2T_dy2 = calculate_spatial_derivative_2D(model_output, dx, dy)
    dT_dt = calculate_temporal_derivative_2D(model_output_pre_aligned, src_dts)

    alphas_reshaped = src_alphas.reshape(-1, 1, 1, 1)
    residuals = dT_dt - alphas_reshaped * (
        d2T_dx2[:, :-1, :, :] + d2T_dy2[:, :-1, :, :]
    )

    # Idea: if you want to z‑score the residuals, re‑enable the next line:
    # residual_std = mx.sqrt(mx.var(residuals) + 1e-8)
    normalized_residuals = residuals

    pi_loss = nn.losses.mse_loss(
        normalized_residuals, mx.zeros_like(normalized_residuals), reduction="mean"
    )

    return pi_loss


def compute_boundary_loss_2D(model_output, target):
    """
    Computes the boundary loss for a 2D heat diffusion model.

    This function calculates the mean squared error (MSE) between the predicted boundary conditions
    (left, right, top, and bottom) and the expected boundary conditions (from the target). The boundary
    conditions are compared for each time step and sample in the batch. The total boundary loss is the sum
    of the individual MSE losses for the four boundaries.

    Parameters
    ----------
    model_output : numpy.ndarray or mlx.core.array
        The predicted temperature distribution field from the model, with shape
        (batch_size, time_steps, ny, nx). The last two dimensions represent the spatial grid.
    target : numpy.ndarray or mlx.core.array
        The expected temperature distribution field, with the same shape as `model_output`.

    Returns
    -------
    mx.core.array
        The boundary loss, computed as the sum of the MSE losses for the left, right, top, and bottom boundaries.
    """
    expected_left_boundary = target[:, :, :, 0]
    expected_right_boundary = target[:, :, :, -1]
    expected_top_boundary = target[:, :, 0, :]
    expected_bottom_boundary = target[:, :, -1, :]

    left_boundary_pred = model_output[:, :, :, 0]
    right_boundary_pred = model_output[:, :, :, -1]
    top_boundary_pred = model_output[:, :, 0, :]
    bottom_boundary_pred = model_output[:, :, -1, :]

    left_boundary_loss = nn.losses.mse_loss(
        left_boundary_pred, expected_left_boundary, reduction="mean"
    )
    right_boundary_loss = nn.losses.mse_loss(
        right_boundary_pred, expected_right_boundary, reduction="mean"
    )
    top_boundary_loss = nn.losses.mse_loss(
        top_boundary_pred, expected_top_boundary, reduction="mean"
    )
    bottom_boundary_loss = nn.losses.mse_loss(
        bottom_boundary_pred, expected_bottom_boundary, reduction="mean"
    )

    boundary_loss = (
        left_boundary_loss
        + right_boundary_loss
        + top_boundary_loss
        + bottom_boundary_loss
    )
    return boundary_loss


def compute_initial_loss_2D(model_output, target, n_initial):
    """
    Computes the initial frame loss for a 2D heat diffusion model.

    This function calculates the mean squared error (MSE) between the predicted and expected initial
    frames (the first few time steps) of the temperature distribution field. The initial frames are
    compared for each sample in the batch.

    Parameters
    ----------
    model_output : numpy.ndarray or mlx.core.array
        The predicted temperature distribution field from the model, with shape
        (batch_size, time_steps, ny, nx). The last two dimensions represent the spatial grid.
    target : numpy.ndarray or mlx.core.array
        The expected temperature distribution field, with the same shape as `model_output`.

    Returns
    -------
    mx.core.array
        The initial frame loss, computed as the mean squared error between the predicted and expected initial frames.
    """
    expected_initial_frames = target[:, 0:n_initial, :, :]
    initial_frames_predicted = model_output[:, 0:n_initial, :, :]

    initial_frames_loss = nn.losses.mse_loss(
        initial_frames_predicted, expected_initial_frames, reduction="mean"
    )

    return initial_frames_loss


def loss_fn_2D(model, src, target, n_initial, src_alphas, src_dts, dx, dy):
    """
    Calculates the total loss for a 2D heat diffusion model.

    This function computes the total loss as a weighted sum of several loss components:
    - Mean Squared Error (MSE) between the predicted and target temperature fields.
    - Physics-informed loss, which enforces the heat equation on the model output.
    - Boundary condition loss, which measures the deviation from expected boundary conditions.
    - Initial condition loss, which compares the first few time steps of the predicted and target fields.

    The total loss is weighted by predefined factors for each component (boundary, physics, and initial losses).

    Parameters
    ----------
    model : object
        The 2D heat diffusion model that predicts the temperature distribution.
    src : numpy.ndarray or mlx.core.array
        The input data representing the initial temperature distribution, with shape
        (batch_size, time_steps, ny, nx).
    target : numpy.ndarray or mlx.core.array
        The expected output temperature distribution, with the same shape as `src`.
    src_alphas : numpy.ndarray or mlx.core.array
        The thermal diffusivity values for each sample, with shape (batch_size,).
    src_dts : numpy.ndarray or mlx.core.array
        The time step sizes for each sample, with shape (batch_size,).
    dx : float
        The spatial step size in the x direction.
    dy : float
        The spatial step size in the y direction.

    Returns
    -------
    mx.core.array
        The total loss, which is a weighted sum of the MSE loss, physics-informed loss, boundary condition loss,
        and initial condition loss.
    """
    boundary_loss_weight = 0.1
    physics_loss_weight = 0.001
    initial_loss_weight = 0.1

    model_output = model(src, src_alphas)

    mse_loss = nn.losses.mse_loss(model_output, target, reduction="mean")
    pi_loss = physics_informed_loss_2D(model_output, src_alphas, src_dts, dx, dy)
    boundary_loss = compute_boundary_loss_2D(model_output, target)
    initial_loss = compute_initial_loss_2D(model_output, target, n_initial)

    total_loss = (
        mse_loss
        + boundary_loss_weight * boundary_loss
        + physics_loss_weight * pi_loss
        + initial_loss_weight * initial_loss
    )

    return total_loss
