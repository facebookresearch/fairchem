"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os
import shutil

CACHE_DIR = os.environ.get(
    "FAIRCHEM_CACHE_DIR", os.path.join(os.path.expanduser("~"), ".cache/fairchem")
)
os.makedirs(CACHE_DIR, exist_ok=True)


def config_dir() -> str:
    """
    The user configuration directory, read from the environment on each call.

    ``FAIRCHEM_CONFIG_DIR`` with ``~`` expanded if set, else ``~/.fairchem``.
    Unlike ``CACHE_DIR`` it is not created here, and ``clear_cache`` leaves it
    alone.

    Returns:
        The directory path, which may not exist.
    """
    return os.path.expanduser(os.environ.get("FAIRCHEM_CONFIG_DIR") or "~/.fairchem")


def clear_cache():
    try:
        shutil.rmtree(CACHE_DIR)
    except FileNotFoundError:
        print(f"No FAIRChem cache directory found at {CACHE_DIR}")
