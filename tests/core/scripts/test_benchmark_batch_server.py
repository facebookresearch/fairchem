"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

from ase.calculators.emt import EMT

from fairchem.core.scripts import benchmark_batch_server as benchmark


def test_workload_and_concurrent_collection_preserve_count_and_inputs(monkeypatch):
    workload = benchmark.build_workload(2, 4)
    assert [len(atoms) for atoms in workload] == [4, 4]
    assert [atoms.symbols[0] for atoms in workload] == ["Cu", "Ni"]
    assert benchmark.build_workload(0, 4) == []
    monkeypatch.setattr(benchmark, "FAIRChemCalculator", lambda *a, **kw: EMT())
    serial = benchmark.run_serial(workload, "omat", None)
    with ThreadPoolExecutor(max_workers=2) as executor:
        batcher = SimpleNamespace(executor=executor, batch_predict_unit=None)
        batched = benchmark.run_batched(workload, "omat", batcher)
    for result in (serial, batched):
        assert result["count"] == 2
        assert result["qps"] > 0
        assert result["mean_latency_s"] > 0
    assert all(atoms.calc is None for atoms in workload)
    summary = benchmark._summarize([1.0, 2.0, 3.0], 5.0, 3)
    assert summary["qps"] == 0.6
    assert summary["median_latency_s"] == summary["mean_latency_s"] == 2.0
    assert benchmark._summarize([2.0], 0, 1)["p95_latency_s"] == 2.0


def test_batch_sweep_reports_speedup_and_saves_probe_configuration(
    tmp_path, monkeypatch
):
    options = []

    class Batcher:
        def __init__(self, **kwargs):
            options.append(kwargs)
            self.predict_unit = kwargs["predict_unit"]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def auto_configure_batching(self, probe, config):
            assert probe[0].natoms.item() == 4
            assert config.min_batch_size == 2
            return SimpleNamespace(
                max_batch_size=8, batch_wait_timeout_s=0.01, median_latency_s=0.02
            )

    monkeypatch.setattr(benchmark, "InferenceBatcher", Batcher)
    monkeypatch.setattr(
        benchmark.pretrained_mlip, "get_predict_unit", Mock(return_value="unit")
    )
    monkeypatch.setattr(
        benchmark,
        "run_serial",
        lambda workload, *a: benchmark._summarize(
            [0.1] * len(workload), 0.2, len(workload)
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "run_batched",
        lambda workload, *a: benchmark._summarize(
            [0.02] * len(workload), 0.1, len(workload)
        ),
    )
    result = benchmark.sweep_batch_server(
        "model",
        "omat",
        [4],
        [1, 2],
        str(tmp_path),
        serial_samples=2,
        warmup_requests=1,
        min_batch_size=2,
        max_batch_size_cap=8,
    )
    configuration = result["results"][0]
    assert configuration["autobatch"]["max_batch_size"] == 8
    assert configuration["natoms_actual"] == 4
    assert [point["speedup"] for point in configuration["concurrency"]] == [1, 2]
    assert options[0]["concurrency_backend_options"] == {"max_workers": 2}
    assert options[0]["split_oom_batch"]
    path = tmp_path / f"aggregated_results_{result['timestamp']}.json"
    assert json.loads(path.read_text()) == result
    benchmark.print_summary_table(result)
    benchmark.generate_plots({"results": []}, str(tmp_path))
