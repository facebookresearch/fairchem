"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from fairchem.core.scripts import sweep_inference_benchmark as sweep


def test_job_results_keep_successful_and_failed_indices():
    jobs = [
        SimpleNamespace(job_id="ok", results=Mock(return_value=[])),
        SimpleNamespace(job_id="bad", results=Mock(side_effect=RuntimeError("failed"))),
    ]
    assert sweep.wait_for_jobs(jobs) == ({0}, {1})
    for job in jobs:
        job.results.assert_called_once()
    for count, expected in [(1, (1, 1)), (4, (1, 4)), (8, (1, 8)), (16, (2, 8))]:
        assert sweep.calculate_node_gpu_distribution(count) == expected
    for count in (0, -1, 9):
        with pytest.raises(ValueError):
            sweep.calculate_node_gpu_distribution(count)


def test_aggregate_plot_and_summarize_results(tmp_path, capsys):
    metadata = []
    for gpus, points in [
        (1, [[10, 8.64], [20, "OOM"]]),
        (2, [[10, 17.28], [20, 17.28]]),
    ]:
        directory = tmp_path / str(gpus)
        directory.mkdir()
        (directory / "benchmark_results.json").write_text(
            json.dumps({"model_to_qps_data": {"test/model": points}})
        )
        metadata.append(
            {
                "num_nodes": 1,
                "num_gpus": gpus,
                "run_name": str(gpus),
                "results_dir": str(directory),
            }
        )
    metadata.append({**metadata[0], "results_dir": str(tmp_path / "missing")})
    result = sweep.collect_and_aggregate_results(
        metadata, "test", [1, 2], str(tmp_path), natoms_list=[10, 20]
    )
    assert len(result["configurations"]) == 2
    assert json.loads((tmp_path / "aggregated_results_test.json").read_text()) == result
    sweep.print_summary_table(result)
    output = capsys.readouterr().out
    assert "100.00 QPS" in output
    assert "20 atoms: OOM" in output
    sweep.generate_plots(result, str(tmp_path), run_name="test run")
    for plot in ("single_gpu_speed", "weak_scaling", "strong_scaling"):
        assert (tmp_path / f"test_model_{plot}.png").stat().st_size > 0
    sweep.generate_plots({"configurations": []}, str(tmp_path))
    sweep.generate_plots(
        {"configurations": [{"num_gpus": 1, "benchmark_data": {}}]}, str(tmp_path)
    )


@pytest.mark.parametrize("per_gpu", [False, True])
def test_sweep_builds_resource_and_atom_overrides_without_submitting_jobs(
    tmp_path, monkeypatch, per_gpu
):
    calls = []

    def load_config(path, overrides):
        config = dict(item.split("=", 1) for item in overrides)
        calls.append(config)
        run_dir = Path(config["job.run_dir"])
        return OmegaConf.create(
            {
                "job": {
                    "timestamp_id": "test",
                    "metadata": {
                        "log_dir": str(run_dir / "logs"),
                        "results_dir": str(run_dir / "results"),
                    },
                }
            }
        )

    monkeypatch.setattr(sweep, "load_config", load_config)
    launch = Mock(return_value=[])
    monkeypatch.setattr(sweep, "slurm_launch", launch)
    arguments = {"natoms_per_gpu_list" if per_gpu else "natoms_list": [10]}
    result = sweep.sweep_nodes_and_atoms(
        "config.yaml",
        [1, 16],
        str(tmp_path),
        base_overrides=["+label=test"],
        **arguments,
    )
    assert launch.call_count == 2
    assert result["configurations"] == []
    assert calls[0]["runner.natoms_list"] == "[10]"
    assert calls[1]["runner.natoms_list"] == ("[160]" if per_gpu else "[10]")
    assert calls[1]["job.scheduler.num_nodes"] == "2"
    assert calls[1]["job.scheduler.ranks_per_node"] == "8"
    assert calls[1]["job.graph_parallel_group_size"] == "16"
    assert calls[1]["+label"] == "test"
    with pytest.raises(ValueError, match="Exactly one"):
        sweep.sweep_nodes_and_atoms("config.yaml", [1], str(tmp_path))
    with pytest.raises(ValueError, match="Exactly one"):
        sweep.sweep_nodes_and_atoms(
            "config.yaml",
            [1],
            str(tmp_path),
            natoms_list=[10],
            natoms_per_gpu_list=[10],
        )
