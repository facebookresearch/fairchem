"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import textwrap

import hydra

from fairchem.core.scripts.sweep_inference_benchmark import load_config


def test_load_config_hydra_composition_and_defaults(tmp_path):
    """
    Test that benchmark config composition works and initializes JobConfig metadata.
    """
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    config_path = tmp_path / "benchmark_config.yaml"
    config_path.write_text(
        textwrap.dedent(
            """
            benchmark:
              base_natoms: 32
            """
        )
    )

    cfg = load_config(
        str(config_path),
        overrides=["+job.scheduler.ranks_per_node=2"],
    )

    assert cfg.benchmark.base_natoms == 32
    assert cfg.job.scheduler.ranks_per_node == 2
    assert cfg.job.metadata is not None
    assert cfg.job.run_name is not None
