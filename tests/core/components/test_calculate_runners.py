"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from fairchem.core.components.calculate import (
    DeformationRunner, ElasticityRunner
)


@pytest.mark.skip(reason="Pending OCPCalculator fix")
def test_calculate_runner(dummy_binary_dataset, tmp_path):
    # TODO for now we will use an old OCPCalculator, update this with a new checkpoint
    calc = None
    # OCPCalculator(
    #     model_name="eSCN-L4-M2-Lay12-S2EF-OC20-2M", local_cache=tmp_path
    # )
    elastic_runner = ElasticityRunner(calc, input_data=dummy_binary_dataset)

    # check running a calculation of all the dataset
    results = elastic_runner.calculate()
    assert len(results) == len(dummy_binary_dataset)
    assert "sid" in results[0]

    for result in results:
        assert "sid" in result
        assert "errors" in result
        assert "traceback" in result
        if result["elastic_tensor"] is not np.nan:
            assert np.array(result["elastic_tensor"])
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


@pytest.mark.gpu()
def test_deformation_runner(calculator, dummy_deformation_dataset, tmp_path):

    # test defaults
    deformation_runner = DeformationRunner(
        calculator,
        input_data=dummy_deformation_dataset,
        calculate_properties=["energy"],
    )

    results = deformation_runner.calculate()
    assert len(results) == len(dummy_deformation_dataset)

    first = results[0]

    # basic required fields
    assert "sid" in first
    assert "natoms" in first
    assert "errors" in first
    assert "traceback" in first

    # check energies exist
    assert "E_ads" in first
    assert "E_int" in first
    assert "E_mof_deform" in first

    # check DFT reference energies
    assert "E_ads_dft" in first
    assert "E_int_dft" in first
    assert "E_mof_deform_dft" in first

    # check atoms saved
    assert "atoms_initial" in first
    assert "atoms" in first

    # test with custom params
    deformation_custom = DeformationRunner(
        calculator,
        input_data=dummy_deformation_dataset,
        calculate_properties=["energy", "forces"],
        save_relaxed_atoms=False,
        normalize_properties_by={"energy": "natoms"},
        save_target_properties=["E_ads_dft"],
        steps=500,
        fmax=0.1,
        maxstep=0.05,
    )

    custom_results = deformation_custom.calculate()
    c0 = custom_results[0]

    assert "atoms" not in c0
    assert "atoms_initial" not in c0

    assert "energy" in c0
    assert "forces" in c0

    assert "E_ads_dft_target" in c0

    # test write results
    deformation_runner.write_results(results, tmp_path)
    results_path = os.path.join(tmp_path, "relaxation_1-0.json.gz")
    assert os.path.exists(results_path)

    chunked = deformation_runner.calculate(job_num=0, num_jobs=2)
    assert len(chunked) == len(dummy_deformation_dataset) // 2

    # test save state
    assert deformation_runner.save_state("dummy_checkpoint") is True
