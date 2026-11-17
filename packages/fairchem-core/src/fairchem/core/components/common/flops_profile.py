"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging


def get_flops_profile(model, input_data, verbose: bool = False):
    try:
        from flops_profiler.profiler import FlopsProfiler
    except Exception as e:
        logging.error(
            "To use this feature you need to install the flops profiler, pip install pip install flops-profiler"
        )
        raise e
    prof = FlopsProfiler(model)
    prof.start_profile()
    model(input_data)
    prof.stop_profile()
    flops = prof.get_total_flops()
    if verbose:
        prof.print_model_profile(profile_step=1)
    return flops
