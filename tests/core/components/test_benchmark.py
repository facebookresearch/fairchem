"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os.path

import pandas as pd
import pytest
from ase.db import connect

from tests.core.testing_utils import launch_main


@pytest.mark.skip(reason="Pending OCPCalculator fix")
def test_elastic_benchmark_launch(dummy_binary_dataset_path):
    # create a fake target data DF
    target_data = []
    with connect(dummy_binary_dataset_path) as db:
        target_data = [
            {"sid": row.data["sid"], "shear_modulus_vrh": 0, "bulk_modulus_vrh": 0}
            for row in db.select()
        ]

    dirname = os.path.dirname(dummy_binary_dataset_path)
    target_data_path = os.path.join(dirname, "elastic_benchmark_target.json")
    pd.DataFrame(target_data).to_json(target_data_path)

    sys_args = [
        "--config",
        "tests/components/configs/test_elastic_benchmark.yaml",
        f"test_data_path={dummy_binary_dataset_path}",
        f"target_data_path={target_data_path}",
    ]
    launch_main(sys_args)
