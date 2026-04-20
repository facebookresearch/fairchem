"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from ase.build import bulk
from ase.calculators.emt import EMT

from fairchem.core.components.calculate import (
    ElasticityRunner,
    RelaxationRunner,
    SinglePointRunner,
)
from fairchem.core.datasets.atoms_sequence import AtomsDatasetSequence

# ---------------------------------------------------------------------------
# Minimal mock job config for CPU-only checkpointing tests
# ---------------------------------------------------------------------------


@dataclass
class _MockMetadata:
    results_dir: str
    checkpoint_dir: str = ""
    preemption_checkpoint_dir: str = ""
    array_job_num: int = 0


@dataclass
class _MockScheduler:
    num_array_jobs: int = 1


@dataclass
class _MockJobConfig:
    metadata: _MockMetadata
    scheduler: _MockScheduler = field(default_factory=_MockScheduler)


def _make_job_config(tmp_path: Path, *, array_job_num: int = 0) -> _MockJobConfig:
    checkpoint_dir = str(tmp_path / "checkpoint")
    return _MockJobConfig(
        metadata=_MockMetadata(
            results_dir=str(tmp_path / "results"),
            checkpoint_dir=checkpoint_dir,
            preemption_checkpoint_dir=checkpoint_dir,
            array_job_num=array_job_num,
        ),
    )


def _make_atoms_list(n: int):
    """Return a plain list of Cu bulk atoms with sid info set."""
    atoms_list = []
    for i in range(n):
        a = bulk("Cu")
        a.info["sid"] = i
        atoms_list.append(a)
    return atoms_list


@pytest.mark.gpu()
def test_elasticity_runner(calculator, dummy_binary_dataset, tmp_path):
    elastic_runner = ElasticityRunner(
        calculator, input_data=AtomsDatasetSequence(dummy_binary_dataset)
    )

    # check running a calculation of all the dataset
    results = elastic_runner.calculate()
    assert len(results) == len(dummy_binary_dataset)
    assert "sid" in results[0]

    for result in results:
        assert "sid" in result
        assert "errors" in result
        assert "traceback" in result
        # TODO this passes locally but not on CI - investigate
        # if result["elastic_tensor"] is not np.nan:
        #     etensor = np.array(result["elastic_tensor"])
        #     npt.assert_allclose(etensor, etensor.transpose())
        # if result["shear_modulus_vrh"] is not np.nan:
        #     assert result["shear_modulus_vrh"] > 0
        # if result["bulk_modulus_vrh"] is not np.nan:
        #     assert result["bulk_modulus_vrh"] > 0

    # check results written to file
    # results_df = pd.DataFrame(results).set_index("sid").sort_index()
    elastic_runner.write_results(results, tmp_path)
    results_path = os.path.join(tmp_path, "elasticity_1-0.json.gz")
    assert os.path.exists(results_path)

    # TODO this passes locally but not on CI - investigate
    # results_df_from_file = pd.read_json(results_path).set_index("sid").sort_index()
    # assert results_df.equals(results_df_from_file)

    # check running only part of the dataset
    results = elastic_runner.calculate(job_num=0, num_jobs=2)
    assert len(results) == len(dummy_binary_dataset) // 2


@pytest.mark.gpu()
def test_singlepoint_runner(calculator, dummy_binary_dataset, tmp_path):
    # Test basic instantiation
    singlepoint_runner = SinglePointRunner(
        calculator, input_data=AtomsDatasetSequence(dummy_binary_dataset)
    )
    # Test with default parameters
    results = singlepoint_runner.calculate()
    assert len(results) == len(dummy_binary_dataset)
    assert "sid" in results[0]
    assert "natoms" in results[0]
    assert "energy" in results[0]
    assert "errors" in results[0]
    assert "traceback" in results[0]

    # # Test with custom properties
    singlepoint_runner_custom = SinglePointRunner(
        calculator,
        input_data=AtomsDatasetSequence(dummy_binary_dataset),
        calculate_properties=["energy", "forces"],
        normalize_properties_by={"energy": "natoms"},
    )
    results_custom = singlepoint_runner_custom.calculate()
    assert len(results_custom) == len(dummy_binary_dataset)
    assert "energy" in results_custom[0]
    assert "forces" in results_custom[0]

    # Test write_results method
    singlepoint_runner.write_results(results, tmp_path)
    results_path = os.path.join(tmp_path, "singlepoint_1-0.json.gz")
    assert os.path.exists(results_path)

    # Test chunked calculation
    results_chunked = singlepoint_runner.calculate(job_num=0, num_jobs=2)
    assert len(results_chunked) == len(dummy_binary_dataset) // 2

    # Test save_state method
    assert singlepoint_runner.save_state("dummy_checkpoint") is True


@pytest.mark.gpu()
def test_relaxation_runner(calculator, dummy_binary_dataset, tmp_path):
    # Test basic instantiation
    relaxation_runner = RelaxationRunner(
        calculator, input_data=AtomsDatasetSequence(dummy_binary_dataset)
    )
    job_config = _make_job_config(tmp_path)
    relaxation_runner._job_config = job_config

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
        fmax=0.1,  # relax_kwargs
        steps=5,  # relax_kwargs
    )
    relaxation_runner_custom._job_config = job_config

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


# ---------------------------------------------------------------------------
# CPU-only checkpointing tests (use EMT, no GPU needed)
# ---------------------------------------------------------------------------


class TestRelaxationRunnerCheckpointing:
    def test_checkpoint_resume_skips_completed_structures(self, tmp_path):
        """Run all structures, save checkpoint, reload, verify all are skipped on resume."""
        atoms_list = _make_atoms_list(4)
        job_config = _make_job_config(tmp_path)

        runner1 = RelaxationRunner(
            EMT(),
            input_data=atoms_list,
            save_relaxed_atoms=False,
            steps=3,
            fmax=1.0,
        )
        runner1._job_config = job_config
        results1 = runner1.calculate(job_num=0, num_jobs=1)
        assert len(results1) == 4

        # Checkpoint should exist and contain all 4 completed results
        checkpoint_dir = Path(job_config.metadata.preemption_checkpoint_dir)
        progress_file = checkpoint_dir / "relaxation_checkpoint_1_0.json.gz"
        assert progress_file.exists()

        # Resume: new runner loads checkpoint, skips all completed structures
        runner2 = RelaxationRunner(
            EMT(),
            input_data=atoms_list,
            save_relaxed_atoms=False,
            steps=3,
            fmax=1.0,
        )
        runner2._job_config = job_config
        runner2.load_state(str(checkpoint_dir))

        assert len(runner2._completed_results) == 4
        assert runner2._completed_sids == {0, 1, 2, 3}

        results2 = runner2.calculate(job_num=0, num_jobs=1)
        # All structures were skipped; returned results come from the loaded checkpoint
        assert len(results2) == 4
        assert {r["sid"] for r in results2} == {0, 1, 2, 3}

    def test_stopfair_graceful_stop(self, tmp_path):
        """STOPFAIR file triggers graceful stop: partial checkpoint saved, results not written."""
        atoms_list = _make_atoms_list(4)
        job_config = _make_job_config(tmp_path)
        results_dir = Path(job_config.metadata.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        runner = RelaxationRunner(
            EMT(),
            input_data=atoms_list,
            save_relaxed_atoms=False,
            steps=50,
            fmax=1e-6,  # very tight so optimizer runs many steps
            heartbeat_interval=1,
        )
        runner._job_config = job_config

        # Place STOPFAIR file before running so the observer fires on the first structure
        stopfair_path = Path(job_config.metadata.checkpoint_dir).parent / "STOPFAIR"
        stopfair_path.write_text("")

        results = runner.calculate(job_num=0, num_jobs=1)

        assert runner._stopped_by_stopfair is True
        # The in-progress structure is discarded; 0 structures were completed before STOPFAIR
        assert len(results) == 0
        # STOPFAIR sentinel is deleted after detection
        assert not stopfair_path.exists()
        # Checkpoint was written (even with 0 results)
        checkpoint_dir = Path(job_config.metadata.preemption_checkpoint_dir)
        assert (checkpoint_dir / "relaxation_checkpoint_1_0.json.gz").exists()
        # write_results is a no-op when stopped by STOPFAIR
        runner.write_results(results, str(results_dir), job_num=0, num_jobs=1)
        assert not (results_dir / "relaxation_1-0.json.gz").exists()

    def test_checkpoint_filename_does_not_match_result_glob(self, tmp_path):
        """Progress file uses underscores so it won't be mistaken for a completed result file."""
        import fnmatch

        atoms_list = _make_atoms_list(2)
        job_config = _make_job_config(tmp_path)

        runner = RelaxationRunner(
            EMT(),
            input_data=atoms_list,
            save_relaxed_atoms=False,
            steps=2,
            fmax=1.0,
        )
        runner._job_config = job_config
        runner.calculate(job_num=0, num_jobs=1)

        checkpoint_dir = Path(job_config.metadata.preemption_checkpoint_dir)
        progress_file = checkpoint_dir / "relaxation_checkpoint_1_0.json.gz"
        assert progress_file.exists()

        # The progress filename must NOT match the result glob pattern ("relaxation_*-*.json.gz")
        assert not fnmatch.fnmatch(
            progress_file.name, RelaxationRunner.result_glob_pattern
        )
