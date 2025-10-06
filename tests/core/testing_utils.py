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
    import multiprocessing as mp
    import threading
    import time

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

    # Clean up any remaining multiprocessing resources
    with contextlib.suppress(Exception):
        # Get all active child processes and terminate them
        active_children = mp.active_children()
        for child in active_children:
            with contextlib.suppress(Exception):
                child.terminate()
                child.join(timeout=5)
                if child.is_alive():
                    child.kill()

    # Clean up PyTorch threads and CUDA contexts
    with contextlib.suppress(Exception):
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # Try to reset CUDA context
            with contextlib.suppress(Exception):
                torch.cuda.reset_peak_memory_stats()

    # Clean up any hanging threads
    with contextlib.suppress(Exception):
        main_thread = threading.current_thread()
        for thread in threading.enumerate():
            if thread != main_thread and thread.is_alive():
                # Give threads a moment to finish naturally
                thread.join(timeout=2)

    # Force another garbage collection to clean up any remaining objects
    gc.collect()

    # Give the system a moment to clean up
    time.sleep(0.1)


def launch_main(sys_args: list) -> None:
    import os
    import signal
    import time

    # Set up a signal handler to force cleanup on timeout
    def timeout_handler(signum, frame):
        print("Test timeout detected, forcing cleanup...")
        _cleanup_predict_units()
        os._exit(1)

    # Detect if we're running in CI environment
    is_ci = any(
        ci_var in os.environ
        for ci_var in ["CI", "GITHUB_ACTIONS", "TRAVIS", "CIRCLECI", "JENKINS_URL"]
    )

    # Set a timeout only in CI environments (5 minutes)
    old_handler = None
    if is_ci:
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(300)  # 5 minutes timeout for CI

    try:
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

            # In CI, add extra cleanup steps and wait
            if is_ci:
                # Force additional cleanup in CI environments
                _force_ci_cleanup()
                # Give extra time for cleanup in CI
                time.sleep(1)
    finally:
        # Reset signal handler and cancel alarm
        if is_ci and old_handler is not None:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)


def _force_ci_cleanup() -> None:
    """Additional cleanup steps specifically for CI environments."""
    import os
    import threading
    import time

    with contextlib.suppress(Exception):
        # Force terminate any remaining torch processes
        import torch

        if hasattr(torch, "_C") and hasattr(torch._C, "_cleanup"):
            torch._C._cleanup()

    with contextlib.suppress(Exception):
        # Clean up OpenMP threads that might be hanging
        os.environ.pop("OMP_NUM_THREADS", None)

    with contextlib.suppress(Exception):
        # Aggressively terminate any non-main threads
        main_thread = threading.current_thread()
        for thread in threading.enumerate():
            if thread != main_thread and thread.is_alive():
                # Force daemon status and try to terminate
                thread.daemon = True

    # Multiple garbage collection passes
    for _ in range(3):
        gc.collect()
        time.sleep(0.1)
