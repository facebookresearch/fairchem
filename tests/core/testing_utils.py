"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import contextlib
import gc
import sys

import hydra

import fairchem.core.common.gp_utils as gp_utils
from fairchem.core._cli import main
from fairchem.core.common import distutils


def _cleanup_predict_units() -> None:
    """Clean up any MLIPPredictUnit instances that might have background processes."""
    # Force garbage collection to ensure __del__ methods are called
    gc.collect()

    # Look for any ParallelMLIPPredictUnit instances and clean them up explicitly
    with contextlib.suppress(ImportError):
        from fairchem.core.units.mlip_unit.predict import ParallelMLIPPredictUnit

        # Find all objects in the garbage collector that are ParallelMLIPPredictUnit instances
        for obj in gc.get_objects():
            if isinstance(obj, ParallelMLIPPredictUnit):
                with contextlib.suppress(Exception):
                    obj.cleanup()

    # Force another garbage collection to clean up any remaining objects
    gc.collect()


def launch_main(sys_args: list) -> None:
    if gp_utils.initialized():
        gp_utils.cleanup_gp()
    distutils.cleanup()
    hydra.core.global_hydra.GlobalHydra.instance().clear()

    try:
        sys.argv[1:] = sys_args
        main()
    finally:
        # Always ensure cleanup happens, even if main() raises an exception
        if gp_utils.initialized():
            gp_utils.cleanup_gp()
        distutils.cleanup()
        _cleanup_predict_units()
