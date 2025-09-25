"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os
from random import choice

import numpy as np
import pandas as pd
import pytest

from fairchem.core import FAIRChemCalculator, pretrained_mlip
from fairchem.core.components.calculate import ElasticityRunner


@pytest.fixture(scope="module")
def calculator() -> FAIRChemCalculator:
    uma_sm_models = [
        model for model in pretrained_mlip.available_models if "_sm" in model
    ]
    return FAIRChemCalculator.from_model_checkpoint(
        choice(uma_sm_models), task_name="omat"
    )


def test_elasticity_runner(calculator, dummy_binary_dataset, tmp_path):
    elastic_runner = ElasticityRunner(calculator, input_data=dummy_binary_dataset)

    # check running a calculation of all the dataset
    results = elastic_runner.calculate()
    assert len(results) == len(dummy_binary_dataset)
    assert "sid" in results[0]

    for result in results:
        assert "sid" in result
        assert "errors" in result
        assert "traceback" in result
        if result["elastic_tensor"] is not np.nan:
            etensor = np.array(result["elastic_tensor"])
            assert np.allclose(etensor, etensor.transpose())
        if result["shear_modulus_vrh"] is not np.nan:
            assert result["shear_modulus_vrh"] > 0
        if result["bulk_modulus_vrh"] is not np.nan:
            assert result["bulk_modulus_vrh"] > 0

    # check results written to file
    results_df = pd.DataFrame(results).set_index("sid").sort_index()
    elastic_runner.write_results(results, tmp_path)
    results_path = os.path.join(tmp_path, "elasticity_1-0.json.gz")
    assert os.path.exists(results_path)
    results_df_from_file = pd.read_json(results_path).set_index("sid").sort_index()
    assert results_df.equals(results_df_from_file)

    # check running only part of the dataset
    results = elastic_runner.calculate(job_num=0, num_jobs=2)
    assert len(results) == len(dummy_binary_dataset) // 2
