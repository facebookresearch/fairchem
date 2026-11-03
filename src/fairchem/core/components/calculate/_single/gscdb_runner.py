"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import csv
import gzip
import io
import json
import logging
import os
from typing import TYPE_CHECKING, Any, ClassVar

import numpy as np
from tqdm import tqdm

from fairchem.core.common.safe_pickle import safe_pickle_load
from fairchem.core.components.calculate._calculate_runner import CalculateRunner

if TYPE_CHECKING:
    from collections.abc import Sequence

    from ase.calculators.calculator import Calculator


def _species_from_stoichiometry(stoichiometry: str) -> list[str]:
    fields = stoichiometry.split(",")
    if len(fields) % 2:
        raise ValueError(f"Invalid GSCDB stoichiometry: {stoichiometry}")
    return [fields[index].strip() for index in range(1, len(fields), 2)]


GSCDB_INPUT_SCHEMA_VERSION = 1
_REQUIRED_INFO_FILES = {
    "DatasetEval.csv",
    "Datasets.csv",
    "FD_stepsize.csv",
}


def load_gscdb_input_data(input_data: str) -> dict[str, Any]:
    """Load and validate a self-contained GSCDB input pickle."""
    with open(input_data, "rb") as input_file:
        data = safe_pickle_load(input_file)

    if not isinstance(data, dict):
        raise ValueError("GSCDB input data must be a dictionary")
    if data.get("schema_version") != GSCDB_INPUT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported GSCDB input schema version: {data.get('schema_version')!r}"
        )
    if not isinstance(data.get("structures"), dict):
        raise ValueError("GSCDB input data is missing its structures dictionary")
    if not isinstance(data.get("info_files"), dict):
        raise ValueError("GSCDB input data is missing its Info files dictionary")

    missing_info_files = _REQUIRED_INFO_FILES - data["info_files"].keys()
    if missing_info_files:
        raise ValueError(
            "GSCDB input data is missing required Info files: "
            + ", ".join(sorted(missing_info_files))
        )
    return data


class GSCDBRunner(CalculateRunner):
    """Run GSCDB single-point energies from a pinned local data artifact."""

    result_glob_pattern: ClassVar[str] = "gscdb_*-*.json.gz"

    def __init__(
        self,
        calculator: Calculator,
        input_data: str,
        benchmark_name: str = "gscdb",
        excluded_datasets: Sequence[str] | None = None,
    ):
        """Initialize the runner.

        Args:
            calculator: ASE calculator used for each single-point calculation.
            input_data: Path to the self-contained ``gscdb_inputs.pkl`` file.
            benchmark_name: Prefix used for shard output files.
            excluded_datasets: Dataset names that should not be evaluated.
        """
        excluded_datasets = set(excluded_datasets or [])
        gscdb_data = load_gscdb_input_data(input_data)
        structures = gscdb_data["structures"]
        with io.StringIO(gscdb_data["info_files"]["DatasetEval.csv"]) as reaction_file:
            reaction_rows = list(csv.DictReader(reaction_file))

        required_species = {
            species
            for row in reaction_rows
            if row["Dataset"] not in excluded_datasets
            for species in _species_from_stoichiometry(row["Stoichiometry"])
        }
        missing_species = sorted(required_species - structures.keys())
        if missing_species:
            examples = ", ".join(missing_species[:3])
            raise FileNotFoundError(
                f"GSCDB input data is missing {len(missing_species)} required species; "
                f"examples: {examples}"
            )

        species_data = {species: structures[species] for species in required_species}
        self.benchmark_name = benchmark_name
        self.result_glob_pattern = f"{benchmark_name}_*-*.json.gz"
        self.input_keys = sorted(species_data)
        super().__init__(calculator=calculator, input_data=species_data)

    def calculate(
        self, job_num: int = 0, num_jobs: int = 1
    ) -> dict[str, dict[str, Any]]:
        """Calculate one array-job shard and return energies in eV."""
        chunk = np.array_split(self.input_keys, num_jobs)[job_num]
        results: dict[str, dict[str, Any]] = {}

        for species in tqdm(chunk):
            atoms = self.input_data[species].copy()
            atoms.info["charge"] = int(atoms.info["charge"])
            atoms.info["spin"] = int(
                atoms.info.get("multiplicity", atoms.info.get("spin"))
            )
            atoms.calc = self.calculator
            try:
                results[species] = {"energy": float(atoms.get_potential_energy())}
            except ValueError as error:
                logging.warning("GSCDB calculation failed for %s: %s", species, error)
                results[species] = {"error": str(error)}
            finally:
                atoms.calc = None

        return results

    def write_results(
        self,
        results: dict[str, dict[str, Any]],
        results_dir: str,
        job_num: int = 0,
        num_jobs: int = 1,
    ) -> None:
        """Write one compressed JSON result shard."""
        output_path = os.path.join(
            results_dir, f"{self.benchmark_name}_{num_jobs}-{job_num}.json.gz"
        )
        with gzip.open(output_path, "wt") as output_file:
            json.dump(results, output_file)

    def save_state(self, checkpoint_location: str, is_preemption: bool = False) -> bool:
        return True
