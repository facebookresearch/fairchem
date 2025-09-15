"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Input validation utilities for the FastCSP workflow.

This module provides comprehensive validation for FastCSP inputs,
ensuring all required parameters are present and have valid values before
workflow execution begins.
"""

from __future__ import annotations

from typing import Any


def validate_config(config: dict[str, Any], stages: list[str]) -> None:
    """
    Validate that the configuration contains all required keys for the specified stages.

    This function performs validation of the FastCSP configuration,
    including checking for required keys, validating nested configurations,
    type checking critical parameters, and ensuring value constraints are met.

    Args:
        config: Configuration dictionary to validate
        stages: List of workflow stages that will be executed
    """
    required_base_keys = ["root"]

    # Stage-specific required keys
    stage_requirements = {
        "generate": {
            "keys": ["molecules", "genarris"],
            "nested": {"genarris": ["python_cmd", "genarris_script", "base_config"]},
        },
        "process_generated": {
            "keys": ["pre_relaxation_filter"],
            "nested": {"pre_relaxation_filter": ["ltol", "stol", "angle_tol"]},
        },
        "relax": {
            "keys": ["relax"],
            "nested": {
                "relax": [
                    "calculator",
                    "optimizer",
                    "fmax",
                    "max-steps",
                    "fix-symmetry",
                    "relax-cell",
                ]
            },
        },
        "filter": {
            "keys": ["post_relaxation_filter"],
            "nested": {
                "post_relaxation_filter": [
                    "energy_cutoff",
                    "density_cutoff",
                    "ltol",
                    "stol",
                    "angle_tol",
                ]
            },
        },
        "evaluate": {
            "keys": ["evaluate"],
            "nested": {"evaluate": ["exp_structures", "method"]},
        },
    }

    # Check base required keys
    missing_base = [key for key in required_base_keys if key not in config]
    if missing_base:
        raise KeyError(f"Missing required base configuration keys: {missing_base}")

    # Check stage-specific requirements
    for stage in stages:
        if stage in stage_requirements:
            stage_req = stage_requirements[stage]

            if "keys" in stage_req:
                missing_keys = [key for key in stage_req["keys"] if key not in config]
                if missing_keys:
                    raise KeyError(
                        f"Missing configuration keys required for stage '{stage}': {missing_keys}"
                    )

            if "nested" in stage_req:
                for parent_key, nested_keys in stage_req["nested"].items():
                    if parent_key not in config:
                        continue

                    missing_nested = [
                        key for key in nested_keys if key not in config[parent_key]
                    ]
                    if missing_nested:
                        raise KeyError(
                            f"Missing nested keys in '{parent_key}' for stage '{stage}': {missing_nested}"
                        )

    # Type validation for critical parameters
    _validate_relax_config_types(config)

    # Value validation
    _validate_config_values(config)


def _validate_relax_config_types(config: dict[str, Any]) -> None:
    """Validate types for relaxation configuration parameters."""
    if "relax" in config:
        relax_config = config["relax"]
        type_validations = {
            "fmax": (float, int),
            "max-steps": int,
            "fix-symmetry": bool,
            "relax-cell": bool,
        }

        for key, expected_types in type_validations.items():
            if key in relax_config and not isinstance(
                relax_config[key], expected_types
            ):
                raise TypeError(
                    f"Configuration '{key}' must be of type {expected_types}, got {type(relax_config[key])}"
                )


def _validate_config_values(config: dict[str, Any]) -> None:
    """Validate value constraints for configuration parameters."""
    # Energy cutoff validation
    if "energy_cutoff" in config and (
        not isinstance(config["energy_cutoff"], (int, float))
        or config["energy_cutoff"] < 0
    ):
        raise ValueError("'energy_cutoff' is problematic")

    # Density cutoff validation
    if "density_cutoff" in config and (
        not isinstance(config["density_cutoff"], (int, float))
        or config["density_cutoff"] < 0
    ):
        raise ValueError("'density_cutoff' is problematic")

    # Tolerance parameter validation
    for param_set in ["pre_relaxation_filter", "post_relaxation_filter"]:
        if param_set in config:
            _validate_tolerance_params(config[param_set], param_set)


def _validate_tolerance_params(params: dict[str, Any], param_set_name: str) -> None:
    """Validate tolerance parameters are positive numbers."""
    tolerance_params = ["ltol", "stol", "angle_tol"]
    for param in tolerance_params:
        if (param in params and not isinstance(params[param], (int, float))) or params[
            param
        ] <= 0:
            raise ValueError(
                f"'{param}' in '{param_set_name}' must be a positive number"
            )


def reorder_stages_by_dependencies(stages: list[str]) -> list[str]:
    """
    Reorder stages to follow the correct workflow dependency order.

    Args:
        stages: List of stages to execute (possibly in wrong order)

    Returns:
        List of stages reordered according to dependency requirements

    Example:
        >>> stages = ["evaluate", "relax", "generate"]
        >>> reorder_stages_by_dependencies(stages)
        ["generate", "relax", "evaluate"]
        # Note: Only reorders, doesn't add in-between dependencies
    """
    # Define the canonical order of all possible stages
    canonical_order = [
        "generate",
        "process_generated",
        "relax",
        "filter",
        "evaluate",
        "free_energy",
        "create_vasp_inputs_relaxed",
        "create_vasp_inputs_unrelaxed",
        "submit_vasp",
        "read_vasp_outputs",
    ]

    requested_stages = set(stages)

    reordered = [stage for stage in canonical_order if stage in requested_stages]

    missing_stages = requested_stages - set(reordered)
    if missing_stages:
        print(f"Warning: Unknown stages found: {missing_stages}")

    if reordered != stages:
        print(f"Reordered stages from: {stages}")
        print(f"                   to: {reordered}")

    return reordered
