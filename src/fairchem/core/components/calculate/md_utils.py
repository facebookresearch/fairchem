"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import importlib

from omegaconf import OmegaConf


def get_const(module_path: str, const_name: str) -> float:
    """Get a constant from a module by dotted path.

    This resolver allows fetching constants like ase.units.fs in Hydra configs.

    Args:
        module_path: Dotted path to module (e.g., 'ase.units')
        const_name: Name of the constant to fetch (e.g., 'fs')

    Returns:
        The constant value

    Example in config:
        timestep:
          _target_: operator.mul
          _args_:
            - 0.5
            - ${get_const:ase.units,fs}
    """
    module = importlib.import_module(module_path)
    return getattr(module, const_name)


def register_md_resolvers():
    """Register custom OmegaConf resolvers for MD configs.

    Call this function before loading MD configs to enable custom resolvers.
    """
    OmegaConf.register_new_resolver("get_const", get_const, replace=True)
