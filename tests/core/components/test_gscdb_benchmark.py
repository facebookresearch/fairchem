"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import csv
import io
import pickle
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pandas as pd
import pytest
from ase import Atoms, units
from ase.calculators.calculator import Calculator

from fairchem.core.components.benchmark import GSCDBReducer
from fairchem.core.components.calculate import GSCDBRunner

if TYPE_CHECKING:
    from pathlib import Path


class _InfoCalculator(Calculator):
    implemented_properties: ClassVar = ["energy"]

    def calculate(self, atoms=None, properties=None, system_changes=None):
        super().calculate(atoms, properties, system_changes)
        assert isinstance(atoms.info["charge"], int)
        assert isinstance(atoms.info["spin"], int)
        self.results["energy"] = float(len(atoms))


def _csv_text(fieldnames: list[str], rows: list[dict[str, object]]) -> str:
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def _write_input_data(
    path: Path,
    reactions: list[dict[str, object]],
    structures: dict[str, Atoms] | None = None,
    dataset_datatypes: dict[str, str] | None = None,
    fd_step_sizes: dict[str, float] | None = None,
) -> None:
    info_files = {
        "DatasetEval.csv": _csv_text(
            ["Reaction", "Dataset", "Reference", "Stoichiometry"], reactions
        ),
        "Datasets.csv": _csv_text(
            ["Name", "Datatype"],
            [
                {"Name": name, "Datatype": datatype}
                for name, datatype in (dataset_datatypes or {}).items()
            ],
        ),
        "FD_stepsize.csv": _csv_text(
            ["Reaction", "fd step"],
            [
                {"Reaction": reaction, "fd step": step_size}
                for reaction, step_size in (fd_step_sizes or {}).items()
            ],
        ),
    }
    with open(path, "wb") as output_file:
        pickle.dump(
            {
                "schema_version": 1,
                "structures": structures or {},
                "info_files": info_files,
            },
            output_file,
        )


def test_gscdb_runner_loads_required_species_and_sets_omol_metadata(tmp_path):
    structures = {}
    for name in ["h", "h2", "excluded"]:
        atoms = Atoms("H" if name == "h" else "H2")
        atoms.info.update(charge="0", multiplicity="1")
        structures[name] = atoms

    input_path = tmp_path / "gscdb_inputs.pkl"
    _write_input_data(
        input_path,
        [
            {
                "Reaction": "r1",
                "Dataset": "TEST",
                "Reference": "0.0",
                "Stoichiometry": "-2,h,1,h2",
            },
            {
                "Reaction": "r2",
                "Dataset": "EXCLUDED",
                "Reference": "0.0",
                "Stoichiometry": "1,excluded",
            },
        ],
        structures=structures,
    )

    runner = GSCDBRunner(
        _InfoCalculator(),
        str(input_path),
        excluded_datasets=["EXCLUDED"],
    )
    assert runner.input_keys == ["h", "h2"]
    assert runner.calculate() == {"h": {"energy": 1.0}, "h2": {"energy": 2.0}}


def test_gscdb_reducer_reports_untrimmed_category_metrics(tmp_path):
    input_path = tmp_path / "gscdb_inputs.pkl"
    _write_input_data(
        input_path,
        [
            {
                "Reaction": "r1",
                "Dataset": "D1",
                "Reference": "0.0",
                "Stoichiometry": "1,a",
            },
            {
                "Reaction": "r2",
                "Dataset": "D1",
                "Reference": "0.0",
                "Stoichiometry": "1,b",
            },
            {
                "Reaction": "r3",
                "Dataset": "D1",
                "Reference": "0.0",
                "Stoichiometry": "1,c",
            },
            {
                "Reaction": "missing",
                "Dataset": "D1",
                "Reference": "0.0",
                "Stoichiometry": "1,missing",
            },
        ],
        dataset_datatypes={"D1": "Thermochemistry"},
    )

    reducer = GSCDBReducer(str(input_path))
    results = {
        "a": {"energy": units.kcal / units.mol},
        "b": {"energy": 3 * units.kcal / units.mol},
        "c": {"energy": 100 * units.kcal / units.mol},
    }
    metrics = reducer.compute_metrics(results, "test-model").iloc[0]

    errors = np.array([1.0, 3.0, 100.0])
    assert metrics["evaluated_count"] == 3
    assert metrics["total_count"] == 4
    assert metrics["coverage"] == pytest.approx(0.75)
    assert metrics["mae"] == pytest.approx(np.mean(np.abs(errors)))
    assert metrics["rmse"] == pytest.approx(np.sqrt(np.mean(errors**2)))
    assert metrics["mean_error"] == pytest.approx(np.mean(errors))
    assert metrics["std_error"] == pytest.approx(np.std(errors))
    assert metrics["p95_absolute_error"] == pytest.approx(
        np.percentile(np.abs(errors), 95)
    )


def test_gscdb_reducer_converts_v30_to_frequencies(tmp_path):
    input_path = tmp_path / "gscdb_inputs.pkl"
    _write_input_data(
        input_path,
        [
            {
                "Reaction": "V30_1",
                "Dataset": "V30",
                "Reference": "0.0001",
                "Stoichiometry": "1,a",
            }
        ],
        dataset_datatypes={"V30": "Frequency"},
        fd_step_sizes={"V30_1": 0.01},
    )

    reducer = GSCDBReducer(str(input_path))
    metrics = reducer.compute_metrics(
        {"a": {"energy": 0.0001 * units.Hartree}}, "test-model"
    ).iloc[0]

    assert metrics["unit"] == "cm^-1"
    assert metrics["mae"] == pytest.approx(0.0)


def test_gscdb_reducer_retains_rows_excluded_from_metrics(tmp_path):
    input_path = tmp_path / "gscdb_inputs.pkl"
    _write_input_data(
        input_path,
        [
            {
                "Reaction": "included",
                "Dataset": "D1",
                "Reference": "0.0",
                "Stoichiometry": "1,a",
            },
            {
                "Reaction": "excluded",
                "Dataset": "D2",
                "Reference": "0.0",
                "Stoichiometry": "1,b",
            },
        ],
        dataset_datatypes={"D1": "Thermochemistry"},
    )

    reducer = GSCDBReducer(
        str(input_path),
        excluded_datasets=["D2"],
    )
    metrics = reducer.compute_metrics(
        {
            "a": {"energy": units.kcal / units.mol},
            "b": {"energy": units.kcal / units.mol},
        },
        "test-model",
    )

    assert reducer.reaction_errors.index.tolist() == ["included", "excluded"]
    assert pd.isna(reducer.reaction_errors.loc["excluded", "Datatype"])
    assert metrics["total_count"].tolist() == [1]
