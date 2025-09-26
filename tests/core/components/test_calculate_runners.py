
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
from fairchem.core.datasets.atoms_sequence import AtomsDatasetSequence
from fairchem.core.components.calculate import ElasticityRunner
from fairchem.core.components.calculate import NVEMDRunner
from fairchem.core.components.calculate import RelaxationRunner
from fairchem.core.components.calculate import SinglePointRunner


@pytest.fixture(autouse=True)
def run_around_tests():
    # disable this locally for now as it is causing issues with empty_cache causeing CUDA device side assert
    # If debugging GPU memory issues, uncomment this print statement
    # to get full GPU memory allocations before each test runs
    # #print(torch.cuda.memory_summary())
    # yield
    # torch.cuda.empty_cache()
    pass


@pytest.fixture(scope="module")
def calculator() -> FAIRChemCalculator:
    uma_sm_models = [
        model for model in pretrained_mlip.available_models if "uma-s" in model
    ]
    return FAIRChemCalculator.from_model_checkpoint(
        choice(uma_sm_models), task_name="omat"
    )


def test_elasticity_runner(calculator, dummy_binary_dataset, tmp_path):
    elastic_runner = ElasticityRunner(calculator, input_data=AtomsDatasetSequence(dummy_binary_dataset))

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


@pytest.mark.disable_run_around_tests
def test_singlepoint_runner(calculator, dummy_binary_dataset, tmp_path):
    # Test basic instantiation
    singlepoint_runner = SinglePointRunner(calculator, input_data=AtomsDatasetSequence(dummy_binary_dataset))
    # Test with default parameters
    results = singlepoint_runner.calculate()
    assert len(results) == len(dummy_binary_dataset)
    assert "sid" in results[0]
    assert "natoms" in results[0]
    assert "energy" in results[0]
    assert "errors" in results[0]
    assert "traceback" in results[0]
    
    # # Test with custom properties
    # singlepoint_runner_custom = SinglePointRunner(
    #     calculator, 
    #     input_data=AtomsDatasetSequence(dummy_binary_dataset),
    #     calculate_properties=["energy", "forces"],
    #     normalize_properties_by={"energy": "natoms"},
    #     save_target_properties=["energy"]
    # )
    # results_custom = singlepoint_runner_custom.calculate()
    # assert len(results_custom) == len(dummy_binary_dataset)
    # assert "energy" in results_custom[0]
    # assert "forces" in results_custom[0]
    
    # # Test write_results method
    # singlepoint_runner.write_results(results, tmp_path)
    # results_path = os.path.join(tmp_path, "singlepoint_1-0.json.gz")
    # assert os.path.exists(results_path)
    
    # # Test chunked calculation
    # results_chunked = singlepoint_runner.calculate(job_num=0, num_jobs=2)
    # assert len(results_chunked) == len(dummy_binary_dataset) // 2
    
    # # Test save_state method
    # assert singlepoint_runner.save_state("dummy_checkpoint") is True


def test_relaxation_runner(calculator, dummy_binary_dataset, tmp_path):
    # Test basic instantiation
    relaxation_runner = RelaxationRunner(calculator, input_data=AtomsDatasetSequence(dummy_binary_dataset))
    
    # Test with default parameters
    results = relaxation_runner.calculate()
    assert len(results) == len(dummy_binary_dataset)
    assert "sid" in results[0]
    assert "natoms" in results[0]
    assert "energy" in results[0]
    assert "errors" in results[0]
    assert "traceback" in results[0]
    assert "opt_nsteps" in results[0]
    assert "opt_converged" in results[0]
    assert "atoms_initial" in results[0]  # default save_relaxed_atoms=True
    assert "atoms" in results[0]  # relaxed atoms
    
    # Test with custom parameters
    relaxation_runner_custom = RelaxationRunner(
        calculator,
        input_data=AtomsDatasetSequence(dummy_binary_dataset),
        calculate_properties=["energy", "forces"],
        save_relaxed_atoms=False,
        normalize_properties_by={"energy": "natoms"},
        save_target_properties=["energy"],
        fmax=0.1,  # relax_kwargs
        steps=50   # relax_kwargs
    )
    results_custom = relaxation_runner_custom.calculate()
    assert len(results_custom) == len(dummy_binary_dataset)
    assert "energy" in results_custom[0]
    assert "forces" in results_custom[0]
    assert "atoms_initial" not in results_custom[0]  # save_relaxed_atoms=False
    assert "atoms" not in results_custom[0]  # save_relaxed_atoms=False
    
    # Test write_results method
    relaxation_runner.write_results(results, tmp_path)
    results_path = os.path.join(tmp_path, "relaxation_1-0.json.gz")
    assert os.path.exists(results_path)
    
    # Test chunked calculation
    results_chunked = relaxation_runner.calculate(job_num=0, num_jobs=2)
    assert len(results_chunked) == len(dummy_binary_dataset) // 2
    
    # Test save_state method
    assert relaxation_runner.save_state("dummy_checkpoint") is True
