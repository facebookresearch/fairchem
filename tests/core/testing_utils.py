"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import sys

import hydra

import fairchem.core.common.gp_utils as gp_utils
from fairchem.core._cli import main
from fairchem.core.common import distutils


def launch_main(sys_args: list) -> None:
    if gp_utils.initialized():
        gp_utils.cleanup_gp()
    distutils.cleanup()
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    sys.argv[1:] = sys_args
    main()
    if gp_utils.initialized():
        gp_utils.cleanup_gp()
    distutils.cleanup()
