"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import gzip
import io
import json
import os
from glob import glob
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from ase import units

from fairchem.core.components.benchmark._benchmark_reducer import BenchmarkReducer
from fairchem.core.components.calculate import GSCDBRunner
from fairchem.core.components.calculate._single.gscdb_runner import (
    load_gscdb_input_data,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


def _finite_difference_energy_to_frequency(
    energy_kcal_per_mol: float, step_size: float
) -> float:
    """Convert a GSCDB V30 finite-difference energy to a frequency in cm^-1."""
    force_constant_to_au = units._me / units._amu * units.Bohr**2
    force_constant = energy_kcal_per_mol * (units.kcal / units.mol) / units.Hartree
    sign = np.sign(force_constant)
    force_constant = abs(force_constant) / step_size**2
    frequency = np.sqrt(force_constant * force_constant_to_au)
    return float(sign * frequency * units.Hartree / units.invcm)


def _reaction_energy(stoichiometry: str, energies: dict[str, float]) -> float:
    fields = stoichiometry.split(",")
    if len(fields) % 2:
        raise ValueError(f"Invalid GSCDB stoichiometry: {stoichiometry}")
    return sum(
        float(fields[index]) * energies[fields[index + 1].strip()]
        for index in range(0, len(fields), 2)
    )


class GSCDBReducer(BenchmarkReducer):
    """Reduce molecular energies into GSCDB errors and category metrics."""

    def __init__(
        self,
        input_data: str,
        benchmark_name: str = "gscdb",
        excluded_datasets: Sequence[str] | None = None,
    ):
        """Initialize the reducer from a self-contained GSCDB input pickle."""
        self.benchmark_name = benchmark_name
        self.excluded_datasets = set(excluded_datasets or [])
        gscdb_data = load_gscdb_input_data(input_data)
        info_files = gscdb_data["info_files"]
        self.reaction_data = pd.read_csv(
            io.StringIO(info_files["DatasetEval.csv"]), index_col="Reaction"
        )
        dataset_info = pd.read_csv(io.StringIO(info_files["Datasets.csv"])).dropna(
            subset=["Name"]
        )
        conflicting_names = dataset_info.groupby("Name")["Datatype"].nunique(
            dropna=False
        )
        conflicting_names = conflicting_names[conflicting_names > 1].index.tolist()
        if conflicting_names:
            raise ValueError(
                "Conflicting GSCDB datatype metadata for: "
                + ", ".join(sorted(conflicting_names))
            )
        self.dataset_info = dataset_info.drop_duplicates("Name").set_index("Name")
        self.fd_step_sizes = pd.read_csv(
            io.StringIO(info_files["FD_stepsize.csv"]), index_col=0
        )
        self.reaction_errors: pd.DataFrame | None = None

    @property
    def runner_type(self) -> type[GSCDBRunner]:
        return GSCDBRunner

    @property
    def glob_pattern(self) -> str:
        return f"{self.benchmark_name}_*-*.json.gz"

    def join_results(self, results_dir: str, glob_pattern: str) -> dict[str, Any]:
        results = {}
        for filepath in glob(os.path.join(results_dir, glob_pattern)):
            with gzip.open(filepath, "rt") as result_file:
                results.update(json.load(result_file))
        return results

    def save_results(self, results: dict[str, Any], results_dir: str) -> None:
        results_path = os.path.join(
            results_dir, f"{self.benchmark_name}_results.json.gz"
        )
        with gzip.open(results_path, "wt") as result_file:
            json.dump(results, result_file)

    def compute_metrics(self, results: dict, run_name: str) -> pd.DataFrame:
        """Compute full-distribution metrics; no percentile trimming is applied."""
        energies = {
            species: result["energy"]
            for species, result in results.items()
            if "energy" in result
        }
        reactions = self.reaction_data.copy()
        reactions["Datatype"] = reactions["Dataset"].map(self.dataset_info["Datatype"])
        missing_datatypes = reactions.loc[
            reactions["Datatype"].isna()
            & ~reactions["Dataset"].isin(self.excluded_datasets),
            "Dataset",
        ].unique()
        if len(missing_datatypes):
            raise ValueError(
                "Missing GSCDB datatype metadata for: "
                + ", ".join(sorted(missing_datatypes))
            )

        predictions = []
        for stoichiometry in reactions["Stoichiometry"]:
            try:
                predictions.append(_reaction_energy(stoichiometry, energies))
            except KeyError:
                predictions.append(np.nan)

        kcal_per_mol = units.kcal / units.mol
        reactions["Reference"] *= units.Hartree / kcal_per_mol
        reactions["Prediction"] = (
            pd.Series(predictions, index=reactions.index) / kcal_per_mol
        )
        reactions["Unit"] = "kcal/mol"

        frequency_mask = reactions["Datatype"] == "Frequency"
        for reaction in reactions.index[frequency_mask]:
            step_size = self.fd_step_sizes.loc[reaction, "fd step"]
            reactions.loc[reaction, "Reference"] = (
                _finite_difference_energy_to_frequency(
                    reactions.loc[reaction, "Reference"], step_size
                )
            )
            if pd.notna(reactions.loc[reaction, "Prediction"]):
                reactions.loc[reaction, "Prediction"] = (
                    _finite_difference_energy_to_frequency(
                        reactions.loc[reaction, "Prediction"], step_size
                    )
                )
            reactions.loc[reaction, "Unit"] = "cm^-1"

        reactions["Error"] = reactions["Prediction"] - reactions["Reference"]
        self.reaction_errors = reactions[
            ["Dataset", "Datatype", "Reference", "Prediction", "Error", "Unit"]
        ]

        metric_errors = self.reaction_errors.loc[
            ~self.reaction_errors["Dataset"].isin(self.excluded_datasets)
        ]
        category_metrics = []
        for (datatype, unit), category in metric_errors.groupby(
            ["Datatype", "Unit"], sort=True
        ):
            errors = category["Error"].dropna().to_numpy()
            category_metrics.append(
                {
                    "run_name": run_name,
                    "category": datatype,
                    "unit": unit,
                    "evaluated_count": len(errors),
                    "total_count": len(category),
                    "coverage": len(errors) / len(category),
                    "mae": np.mean(np.abs(errors)) if len(errors) else np.nan,
                    "rmse": np.sqrt(np.mean(errors**2)) if len(errors) else np.nan,
                    "mean_error": np.mean(errors) if len(errors) else np.nan,
                    "std_error": np.std(errors) if len(errors) else np.nan,
                    "p95_absolute_error": (
                        np.percentile(np.abs(errors), 95) if len(errors) else np.nan
                    ),
                }
            )
        return pd.DataFrame(category_metrics)

    def save_metrics(self, metrics: pd.DataFrame, results_dir: str) -> None:
        if self.reaction_errors is None:
            raise RuntimeError("compute_metrics must be called before save_metrics")
        self.reaction_errors.to_csv(
            os.path.join(results_dir, f"{self.benchmark_name}_reaction_errors.csv")
        )
        metrics.to_csv(
            os.path.join(results_dir, f"{self.benchmark_name}_metrics_by_category.csv"),
            index=False,
        )

    def log_metrics(self, metrics: pd.DataFrame, run_name: str) -> None:
        if self.logger is not None:
            if self.reaction_errors is None:
                raise RuntimeError("compute_metrics must be called before log_metrics")
            self.logger.log_dataframe(
                name=f"{self.benchmark_name}_metrics_by_category", dataframe=metrics
            )
            self.logger.log_dataframe(
                name=f"{self.benchmark_name}_reaction_errors",
                dataframe=self.reaction_errors.reset_index(),
            )

    def save_state(self, checkpoint_location: str, is_preemption: bool = False) -> bool:
        return True

    def load_state(self, checkpoint_location: str | None) -> None:
        pass
