# transformer/io_utils.py
"""Generic path helpers – no ML logic."""

from __future__ import annotations
import os
import mlx as mx
import pickle, json
from pathlib import Path
from typing import Tuple

__all__ = ["setup_save_directories", "setup_load_directories"]


# ── copy‑paste exactly your two directory helpers ─────────────────────────
def setup_save_directories(
    run_name: str, restart_epoch: int | None = None
) -> Tuple[str, str, str, str]:
    import os

    script_dir = os.path.dirname(__file__)
    output_dir = os.path.join(script_dir, "OUTPUTS")
    os.makedirs(output_dir, exist_ok=True)

    if restart_epoch is not None:
        run_name = f"{run_name}_restart_epoch_{restart_epoch}"

    save_dir_path = os.path.join(output_dir, f"Transformer_save_BeyondL_{run_name}")
    dataset_save_path = os.path.join(output_dir, f"Datasets_save_BeyondL_{run_name}")
    frameplots_path = os.path.join(output_dir, f"Heatmaps_BeyondL_{run_name}")
    inference_mse_path = os.path.join(output_dir, f"InferenceMSE_BeyondL_{run_name}")

    for p in (save_dir_path, dataset_save_path, frameplots_path, inference_mse_path):
        os.makedirs(p, exist_ok=True)

    return save_dir_path, dataset_save_path, frameplots_path, inference_mse_path


def setup_load_directories(run_name: str, checkpoint_epoch: int) -> Tuple[str, str]:
    import os

    script_dir = os.path.dirname(__file__)
    output_dir = os.path.join(script_dir, "OUTPUTS")

    load_dir_path = os.path.join(output_dir, f"Transformer_save_BeyondL_{run_name}")
    dataset_load_path = os.path.join(output_dir, f"Datasets_save_BeyondL_{run_name}")

    if not os.path.exists(load_dir_path):
        raise FileNotFoundError(f"Checkpoint directory {load_dir_path} does not exist!")
    if not os.path.exists(dataset_load_path):
        raise FileNotFoundError(
            f"Dataset directory {dataset_load_path} does not exist!"
        )

    return load_dir_path, dataset_load_path


# Function to save model and optimizer state periodically
def save_model_and_optimizer(
    model,
    optimizer,
    mx_random_state,
    config,
    current_epoch,
    dir_path,
    model_base_file_name,
    optimizer_base_file_name,
    hyper_base_file_name,
):
    """
    Saves the model's state, weights, optimizer state, random state, and configuration at the specified epoch.

    This function saves the current state of the model, including its weights, the optimizer's state,
    the random state, and the training configuration to files. These files can be used to resume
    training or perform model evaluation at a later point.

    Parameters
    ----------
    config : dict
        The training configuration, including parameters like batch size, learning rate, and model architecture.
    current_epoch : int
        The current epoch number, used to save the model and optimizer states with the corresponding epoch suffix.
    dir_path : str
        The directory where the model, optimizer, random state, and configuration files will be saved.
    model_base_file_name : str
        The base name for the model file (the epoch number will be appended).
    optimizer_base_file_name : str
        The base name for the optimizer file (the epoch number will be appended).
    hyper_base_file_name : str
        The base name for the configuration (hyperparameters) file (the epoch number will be appended).

    Returns
    -------
    None
        The function saves the model's state, optimizer state, random state, and configuration to files in the specified directory.
    """
    # Ensure the directory exists, create it if not
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    model_file_name = f"{model_base_file_name}_epoch_{current_epoch}.pkl"
    weights_file_name = (
        f"{model_base_file_name}_weights_epoch_{current_epoch}.safetensors"
    )
    # optimizer_file_name = f"{optimizer_base_file_name}_epoch_{current_epoch}.safetensors"
    optimizer_file_name = f"{optimizer_base_file_name}_epoch_{current_epoch}.pkl"
    random_state_file_name = f"random_state_epoch_{current_epoch}.pkl"
    config_file_name = f"{hyper_base_file_name}_epoch_{current_epoch}.json"

    # Save model state (parameters only)
    model_file_path = os.path.join(dir_path, model_file_name)
    with open(model_file_path, "wb") as f:
        pickle.dump(model.parameters(), f)
        # pickle.dump(model.state, f)

    # Save model weights
    weights_file_path = os.path.join(dir_path, weights_file_name)
    model.save_weights(weights_file_path)

    # Save optimizer state
    optimizer_file_path = os.path.join(dir_path, optimizer_file_name)
    with open(optimizer_file_path, "wb") as f:
        pickle.dump(optimizer.state, f)

    # Save random state
    random_state_path = os.path.join(dir_path, random_state_file_name)
    with open(random_state_path, "wb") as f:
        pickle.dump(mx_random_state, f)

    # Save training configuration
    config["current_epoch"] = current_epoch
    hyper_file_path = os.path.join(dir_path, config_file_name)
    with open(hyper_file_path, "w") as json_file:
        json.dump(config, json_file, indent=4)

    print(
        f"Model, optimizer, random state, and configuration saved at epoch {current_epoch}."
    )


def load_model_and_optimizer(
    model,
    optimizer,
    mx_random_state,
    dir_path,
    model_base_file_name,
    optimizer_base_file_name,
    hyper_base_file_name,
    checkpoint_epoch,
    comparison=True,
):
    """
    Loads the model state, optimizer state, random state, and configuration from a specific checkpoint.

    This function loads the saved model parameters, optimizer state, random state, and training configuration
    from a specified epoch. It checks for the existence of the saved files and compares the current states with
    the loaded ones to ensure consistency. If no checkpoint is found, the training starts from scratch.

    Parameters
    ----------
    model : object
        The model instance whose state (parameters and weights) will be loaded.
    optimizer : object
        The optimizer instance whose state will be loaded.
    dir_path : str
        The directory where the model, optimizer, random state, and configuration files are saved.
    model_base_file_name : str
        The base file name used for saving the model (with epoch appended).
    optimizer_base_file_name : str
        The base file name used for saving the optimizer (with epoch appended).
    hyper_base_file_name : str
        The base file name used for saving the configuration (with epoch appended).
    checkpoint_epoch : int
        The epoch number from which to load the model, optimizer, random state, and configuration.

    Returns
    -------
    tuple
        A tuple containing:
        - start_epoch : int
            The epoch from which to resume training. If no checkpoint is found, this is set to 0.
        - loaded_optimizer_state : dict
            The loaded optimizer state.
        - loaded_random_state : object
            The loaded random state for reproducibility.
        - loaded_parameters : dict
            The loaded model parameters.
        - loaded_config: dict
            The loaded configration
    """
    # Construct the file paths for model, optimizer, random state, and configuration
    model_file_name = f"{model_base_file_name}_epoch_{checkpoint_epoch}.pkl"
    weights_file_name = (
        f"{model_base_file_name}_weights_epoch_{checkpoint_epoch}.safetensors"
    )
    optimizer_file_name = f"{optimizer_base_file_name}_epoch_{checkpoint_epoch}.pkl"
    random_state_file_name = f"random_state_epoch_{checkpoint_epoch}.pkl"
    config_file_name = f"{hyper_base_file_name}_epoch_{checkpoint_epoch}.json"

    model_file_path = os.path.join(dir_path, model_file_name)
    weights_file_path = os.path.join(dir_path, weights_file_name)
    optimizer_file_path = os.path.join(dir_path, optimizer_file_name)
    random_state_file_path = os.path.join(dir_path, random_state_file_name)
    config_file_path = os.path.join(dir_path, config_file_name)

    # Check if all necessary files exist for loading
    if (
        os.path.exists(model_file_path)
        and os.path.exists(optimizer_file_path)
        and os.path.exists(random_state_file_path)
        and os.path.exists(config_file_path)
        and os.path.exists(weights_file_path)
    ):
        # Load model state (parameters only)
        with open(model_file_path, "rb") as f:
            loaded_parameters = pickle.load(f)

        # Load optimizer state
        with open(optimizer_file_path, "rb") as f:
            loaded_optimizer_state = pickle.load(f)

        # Load random state
        with open(random_state_file_path, "rb") as f:
            loaded_random_state = pickle.load(f)

        # Load training configuration
        with open(config_file_path, "r") as json_file:
            loaded_config = json.load(json_file)
            # config = loaded_config

        # Get the start epoch from the configuration
        start_epoch = loaded_config.get("current_epoch", 0)

        # Load current states to compare with the loaded states
        current_optimizer_state = optimizer.state
        current_random_state = mx_random_state
        current_model_parameters = model.parameters()

        # Compare the states for consistency (useful when live saving-and_reloading during training for debugging)
        if comparison:
            if compare_dict_states(
                current_optimizer_state, loaded_optimizer_state, "optimizer state"
            ):
                print("Optimizer state matches.")
            else:
                print("Optimizer state mismatch detected.")

            if compare_list_states(
                current_random_state, loaded_random_state, "random state"
            ):
                print("Random state matches.")
            else:
                print("Random state mismatch detected.")

            if compare_dict_states(
                current_model_parameters, loaded_parameters, "model state"
            ):
                print("Model state matches.")
            else:
                print("Model state mismatch detected.")

            print(
                f"Model, optimizer, random state, and configuration loaded from {dir_path} at epoch {checkpoint_epoch}."
            )
    else:
        # If no checkpoint is found, start from scratch
        start_epoch = 0
        print(
            f"No saved model found at epoch {checkpoint_epoch}. Starting training from scratch."
        )

    return (
        start_epoch,
        loaded_optimizer_state,
        loaded_random_state,
        loaded_parameters,
        loaded_config,
    )
