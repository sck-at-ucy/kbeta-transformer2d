# transformer/utils.py
from __future__ import annotations

# ── std‑lib (optional, but handy for type hints) --------------------------
import mlx.core as mx  # needed by compute_average_var

# ── third‑party -----------------------------------------------------------
import numpy as np  # needed by compare_* helpers


def get_learning_rate_for_epoch(epoch, schedule):
    """
    Function get_learning_rate_for_epoch.
    Args:
        epoch: Description of epoch.
        schedule: Description of schedule.
    Returns:
        Loaded data or configuration.
    """
    sorted_epochs = sorted(schedule.keys())
    for i in range(len(sorted_epochs) - 1):
        if sorted_epochs[i] <= epoch - 1 < sorted_epochs[i + 1]:
            return schedule[sorted_epochs[i]]
    return schedule[sorted_epochs[-1]] if epoch >= sorted_epochs[-1] else 0.0


def compute_average_var(optimizer):
    """
    Computes the average posterior_var across all parameters stored
    in optimizer.state, ignoring sub-entries that don't have 'posterior_var.'
    """
    all_vars = []

    # Iterate over (param or name, param_state) pairs. Because MLX can nest
    # dictionaries, you'll see entries like ('transformer_encoder', {...}),
    # ('positional_encoding_x', {...}), etc. Each could contain further nesting.

    for key, value in optimizer.state.items():
        # 'value' might be a nested dict with multiple sub-layers or sub-params
        if isinstance(value, dict):
            # Explore deeper sub-dicts
            for subkey, subval in value.items():
                if isinstance(subval, dict):
                    # E.g. subval might be "layers": [...]
                    # If that is a list of layers, each layer might have a dict of sub-params
                    if isinstance(subval, list):
                        # subval is e.g. the list of layer definitions
                        for layer in subval:
                            # layer is also a dict
                            for layer_key, layer_val in layer.items():
                                # e.g. 'attention': { 'query_proj': {...}, ... }
                                if isinstance(layer_val, dict):
                                    # We'll search for actual param dicts
                                    for param_name, param_data in layer_val.items():
                                        if (
                                            isinstance(param_data, dict)
                                            and "posterior_var" in param_data
                                        ):
                                            arr = param_data["posterior_var"]
                                            all_vars.append(mx.mean(arr))
                    else:
                        # Possibly a direct param dict
                        # E.g. subval might have "posterior_var"
                        if "posterior_var" in subval:
                            arr = subval["posterior_var"]
                            all_vars.append(mx.mean(arr))
                # Another possibility is subval itself has 'posterior_var'
                elif "posterior_var" in subval:
                    arr = subval["posterior_var"]
                    all_vars.append(mx.mean(arr))
        # If 'value' itself directly has a 'posterior_var'
        elif "posterior_var" in value:
            arr = value["posterior_var"]
            all_vars.append(mx.mean(arr))

    # Now compute final average
    if not all_vars:
        return None

    # Stack them and average
    return float(mx.mean(mx.stack(all_vars)))


def compare_datasets(saved, loaded, dataset_name):
    """
    Compares two datasets and logs mismatches.

    This function compares two datasets (`saved` and `loaded`) to check for mismatches. It handles
    both MLX-specific arrays (if `mx.array` is used) and generic arrays, using `np.array_equal` to
    verify if the datasets match. If a mismatch is found, it logs an error message and returns `False`.
    If the datasets match, it returns `True`.

    Parameters
    ----------
    saved : mx.array or numpy.ndarray
        The saved dataset that serves as the reference.
    loaded : mx.array or numpy.ndarray
        The loaded dataset to be compared against the saved dataset.
    dataset_name : str
        A string representing the name of the dataset being compared, used for logging purposes.

    Returns
    -------
    bool
        `True` if the datasets match, `False` otherwise.
    """
    # Compare MLX-specific arrays by converting them to numpy arrays
    if isinstance(saved, mx.array) and isinstance(loaded, mx.array):
        if not np.array_equal(np.array(saved), np.array(loaded)):
            print(f"Dataset {dataset_name} does not match.")
            return False
    else:
        # Compare generic numpy arrays or other data structures
        if not np.array_equal(saved, loaded):
            print(f"Dataset {dataset_name} does not match.")
            return False
    return True


def print_fresh_run_config(current_config):
    """
    Prints the current configuration for fresh runs to check the setup before starting.
    Each "Config Key" is printed on its own line, followed by the value for "Current Config".
    This is useful to inspect the configuration before a fresh run begins.

    Parameters
    ----------
    current_config : dict
        The configuration to print, including model parameters, training settings, etc.
    """
    print(" ===== Current Configuration for Fresh Run ====")
    print(" ----  Ensure everything is correct before starting ----")
    print(f"{'Config Key':<25}")
    print("=" * 50)

    # Print each key-value pair in the current configuration
    for key in sorted(current_config.keys()):
        current_value = current_config.get(key, "N/A")

        # Print the key and its value
        print(f"{key:<25}")
        print(f"  Current Config: {str(current_value)}")
        print("-" * 50)


def print_config_comparison(current_config, loaded_config):
    """
    Prints the current configuration alongside the reloaded configuration (from the checkpoint)
    for easy comparison. Each "Config Key" is printed on its own line, followed by the values for
    "Current Config" and "Loaded Config" on two separate, indented lines.
    """
    print(" ===== Comparison of Current Config and Loaded Config ====")
    print(
        " ----  Current config is NOT overwritten. Loaded Config is only used to check consistency ----"
    )
    print(f"{'Config Key':<25}")
    print("=" * 50)

    # Get all unique keys from both configs
    all_keys = set(current_config.keys()).union(set(loaded_config.keys()))

    for key in sorted(all_keys):
        current_value = current_config.get(
            key, "N/A"
        )  # Get value from current config or "N/A" if not present
        loaded_value = loaded_config.get(
            key, "N/A"
        )  # Get value from loaded config or "N/A" if not present

        # Print the key and values on separate lines
        print(f"{key:<25}")
        print(f"  Current Config: {str(current_value)}")
        print(f"  Loaded Config:  {str(loaded_value)}")
        print("-" * 50)


# Convert lists to tuples for comparison purposes
def convert_lists_to_tuples(config):
    """
    Recursively converts lists to tuples in the configuration for consistency.
    This is particularly useful for configurations that expect tuples, and it ensures
    that comparison between current and loaded configurations is accurate.
    """
    if isinstance(config, dict):
        # Iterate over all keys in the dict
        for key, value in config.items():
            if isinstance(value, list):
                config[key] = tuple(value)
            elif isinstance(value, dict):
                convert_lists_to_tuples(value)  # Recurse for nested dicts
    return config


def compare_list_states(original, loaded, state_name):
    """
    Compares two list states and logs differences.

    This function compares two lists (`original` and `loaded`) element by element, logging any
    mismatches. It supports lists that may contain dictionaries or nested lists. The function
    also handles cases where elements are `None` and skips logging errors for such cases. If
    the lists have different lengths, it logs an error and returns `False`. The function returns
    `True` if the lists match in both length and content, and `False` otherwise.

    Parameters
    ----------
    original : list
        The original list state to be compared.
    loaded : list
        The loaded list state to be compared against the original.
    state_name : str
        A string representing the name of the state being compared, used for logging purposes.

    Returns
    -------
    bool
        `True` if the two lists match in length and content, `False` otherwise.
    """
    match = True
    if len(original) != len(loaded):
        print(f"Error comparing {state_name} - Length mismatch.")
        return False
    for i in range(len(original)):
        if isinstance(original[i], dict) and isinstance(loaded[i], dict):
            if not compare_dict_states(original[i], loaded[i], f"{state_name}[{i}]"):
                match = False
        elif isinstance(original[i], list) and isinstance(loaded[i], list):
            if not compare_list_states(original[i], loaded[i], f"{state_name}[{i}]"):
                match = False
        else:
            if original[i] is None and loaded[i] is None:
                continue
            if not np.array_equal(original[i], loaded[i]):
                print(f"Error comparing {state_name} at index: {i}")
                match = False
    return match


def compare_dict_states(original, loaded, state_name):
    """
    Compares two dictionary states and logs differences.

    This function compares the values in two dictionaries (`original` and `loaded`) to check for
    mismatches. It supports nested dictionaries and lists within the dictionaries. The function
    also handles cases where keys or values are `None`, empty lists, or empty dictionaries, and
    skips logging errors for such cases. The function returns `True` if all keys and values match
    between the two dictionaries, and `False` otherwise.

    Parameters
    ----------
    original : dict
        The original dictionary state to be compared.
    loaded : dict
        The loaded dictionary state to be compared against the original.
    state_name : str
        A string representing the name of the state being compared, used for logging purposes.

    Returns
    -------
    bool
        `True` if the two dictionaries match in all keys and values, `False` otherwise.
    """
    match = True
    for key in original:
        if key not in loaded:
            if original[key] is None or original[key] == [] or original[key] == {}:
                continue  # Do not treat as an error if the original value is None or empty
            print(
                f"Error comparing {state_name} at key: {key} - Key not found in loaded state."
            )
            match = False
            continue
        if isinstance(original[key], dict) and isinstance(loaded[key], dict):
            if not compare_dict_states(
                original[key], loaded[key], f"{state_name}.{key}"
            ):
                match = False
        elif isinstance(original[key], list) and isinstance(loaded[key], list):
            if not compare_list_states(
                original[key], loaded[key], f"{state_name}.{key}"
            ):
                match = False
        else:
            if original[key] is None and loaded[key] is None:
                continue
            if not np.array_equal(original[key], loaded[key]):
                print(f"Error comparing {state_name} at key: {key}")
                match = False
    for key in loaded:
        if key not in original:
            print(
                f"Error comparing {state_name} at key: {key} - Key not found in original state."
            )
            match = False
    return match
