"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import gzip
import json
import math
import os
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from fairchem.core.common.safe_pickle import safe_pickle_load
from fairchem.core.components.benchmark._benchmark_reducer import BenchmarkReducer
from fairchem.core.components.calculate import LiquidDensityRunner
from fairchem.core.components.calculate._single.liquid_density_runner import (
    load_liquid_density_inputs,
)

_REFERENCE_SCHEMA_VERSION = 1


def load_liquid_density_references(path: str | os.PathLike[str]) -> dict[str, float]:
    """Load a trusted pickle containing experimental density references."""
    payload = safe_pickle_load(str(path), warn=False)
    if not isinstance(payload, dict):
        raise ValueError("liquid-density references must be a dictionary")
    if payload.get("schema_version") != _REFERENCE_SCHEMA_VERSION:
        raise ValueError(
            f"liquid-density reference schema_version must be "
            f"{_REFERENCE_SCHEMA_VERSION}"
        )
    raw_references = payload.get("references")
    if not isinstance(raw_references, dict) or not raw_references:
        raise ValueError("liquid-density references must be a non-empty mapping")

    references: dict[str, float] = {}
    for system_id, entry in raw_references.items():
        if not isinstance(system_id, str) or not system_id:
            raise ValueError("liquid-density reference IDs must be non-empty strings")
        if not isinstance(entry, dict) or "experimental_density_g_cm3" not in entry:
            raise ValueError(f"{system_id}: missing experimental_density_g_cm3")
        density = float(entry["experimental_density_g_cm3"])
        if not math.isfinite(density) or density <= 0:
            raise ValueError(f"{system_id}: reference density must be positive")
        references[system_id] = density
    return dict(sorted(references.items()))


class LiquidDensityReducer(BenchmarkReducer):
    """Join liquid-density jobs and evaluate them against experiment."""

    def __init__(
        self,
        benchmark_name: str,
        input_data_path: str,
        reference_data_path: str,
    ):
        self.benchmark_name = benchmark_name
        self.systems = load_liquid_density_inputs(input_data_path)
        self.references = load_liquid_density_references(reference_data_path)
        input_ids = set(self.systems)
        reference_ids = set(self.references)
        if input_ids != reference_ids:
            missing = sorted(input_ids - reference_ids)
            extra = sorted(reference_ids - input_ids)
            raise ValueError(
                f"liquid-density input/reference IDs disagree; missing={missing}, "
                f"extra={extra}"
            )
        for system_id, entry in self.systems.items():
            input_density = float(entry["experimental_density_g_cm3"])
            reference_density = self.references[system_id]
            if not math.isclose(
                input_density, reference_density, rel_tol=0, abs_tol=1e-12
            ):
                raise ValueError(
                    f"{system_id}: JSON density ({input_density}) disagrees with "
                    f"pickle reference ({reference_density})"
                )

    @property
    def runner_type(self) -> type[LiquidDensityRunner]:
        return LiquidDensityRunner

    def join_results(self, results_dir: str, glob_pattern: str) -> pd.DataFrame:
        rows: list[dict[str, Any]] = []
        paths = sorted(glob(os.path.join(results_dir, glob_pattern)))
        if not paths:
            raise ValueError(f"no liquid-density results found in {results_dir}")
        for path in paths:
            with gzip.open(path, "rt") as handle:
                payload = json.load(handle)
            if not isinstance(payload, list):
                raise ValueError(f"{path}: liquid-density result must be a list")
            rows.extend(payload)

        results = pd.DataFrame(rows)
        if "system_id" not in results:
            raise ValueError("liquid-density results are missing system_id")
        duplicates = sorted(
            results.loc[results["system_id"].duplicated(), "system_id"].unique()
        )
        if duplicates:
            raise ValueError(f"duplicate liquid-density results: {duplicates}")

        result_ids = set(results["system_id"])
        expected_ids = set(self.systems)
        if result_ids != expected_ids:
            missing = sorted(expected_ids - result_ids)
            extra = sorted(result_ids - expected_ids)
            raise ValueError(
                f"liquid-density result IDs disagree; missing={missing}, extra={extra}"
            )

        results["experimental_density_g_cm3"] = results["system_id"].map(
            self.references
        )
        results["signed_error_g_cm3"] = (
            results["predicted_density_g_cm3"] - results["experimental_density_g_cm3"]
        )
        results["absolute_error_g_cm3"] = results["signed_error_g_cm3"].abs()
        results["absolute_relative_error_pct"] = (
            100.0
            * results["absolute_error_g_cm3"]
            / results["experimental_density_g_cm3"]
        )
        return results.sort_values("system_id").reset_index(drop=True)

    def save_results(self, results: pd.DataFrame, results_dir: str) -> None:
        results.to_json(
            os.path.join(results_dir, f"{self.benchmark_name}_results.json.gz"),
            orient="records",
        )

    def compute_metrics(self, results: pd.DataFrame, run_name: str) -> pd.DataFrame:
        predicted = pd.to_numeric(results["predicted_density_g_cm3"], errors="coerce")
        valid_mask = results["status"].eq("ok") & np.isfinite(predicted)
        valid = results.loc[valid_mask]
        num_total = len(results)
        num_successful = len(valid)
        num_failed = num_total - num_successful

        metrics: dict[str, float | int] = {
            "num_systems": num_total,
            "num_successful": num_successful,
            "num_failed": num_failed,
            "success_fraction": num_successful / num_total,
        }
        if num_successful:
            signed_error = valid["signed_error_g_cm3"].astype(float)
            absolute_error = valid["absolute_error_g_cm3"].astype(float)
            relative_error = valid["absolute_relative_error_pct"].astype(float)
            metrics.update(
                {
                    "density_mae_g_cm3": float(absolute_error.mean()),
                    "density_rmse_g_cm3": float(
                        np.sqrt(np.mean(np.square(signed_error)))
                    ),
                    "density_mean_absolute_relative_error_pct": float(
                        relative_error.mean()
                    ),
                    "density_bias_g_cm3": float(signed_error.mean()),
                    "density_max_absolute_relative_error_pct": float(
                        relative_error.max()
                    ),
                }
            )
        else:
            metrics.update(
                {
                    "density_mae_g_cm3": float("nan"),
                    "density_rmse_g_cm3": float("nan"),
                    "density_mean_absolute_relative_error_pct": float("nan"),
                    "density_bias_g_cm3": float("nan"),
                    "density_max_absolute_relative_error_pct": float("nan"),
                }
            )
        return pd.DataFrame([metrics], index=[run_name])

    def save_metrics(self, metrics: pd.DataFrame, results_dir: str) -> None:
        metrics.to_json(
            os.path.join(results_dir, f"{self.benchmark_name}_metrics.json.gz")
        )

    def save_parity_plot(self, results: pd.DataFrame, results_dir: str) -> str:
        """
        Save a parity plot of predicted and experimental liquid densities.

        Failed systems and rows with non-finite densities are omitted from the plot.

        Returns:
            Path to the saved PNG file.
        """
        predicted = pd.to_numeric(results["predicted_density_g_cm3"], errors="coerce")
        experimental = pd.to_numeric(
            results["experimental_density_g_cm3"], errors="coerce"
        )
        valid_mask = (
            results["status"].eq("ok")
            & np.isfinite(predicted)
            & np.isfinite(experimental)
        )
        valid_predicted = predicted[valid_mask].to_numpy(dtype=float)
        valid_experimental = experimental[valid_mask].to_numpy(dtype=float)

        figure = Figure(figsize=(6, 6), layout="constrained")
        axes = figure.subplots()
        if valid_predicted.size:
            axes.scatter(valid_experimental, valid_predicted)
            lower = float(min(valid_experimental.min(), valid_predicted.min()))
            upper = float(max(valid_experimental.max(), valid_predicted.max()))
            padding = max(0.05 * (upper - lower), 0.01)
            limits = (max(0.0, lower - padding), upper + padding)
            axes.plot(limits, limits, linestyle="--", color="black", label="Parity")
            axes.set_xlim(limits)
            axes.set_ylim(limits)
            axes.legend()
        else:
            axes.text(
                0.5,
                0.5,
                "No successful density predictions",
                horizontalalignment="center",
                verticalalignment="center",
                transform=axes.transAxes,
            )
        axes.set_aspect("equal", adjustable="box")
        axes.set_xlabel("Experimental density (g/cm³)")
        axes.set_ylabel("Predicted density (g/cm³)")
        axes.set_title("Liquid density parity")
        axes.grid(alpha=0.25)

        path = Path(results_dir) / f"{self.benchmark_name}_parity.png"
        figure.savefig(path, dpi=200)
        return str(path)

    def log_metrics(self, metrics: pd.DataFrame, run_name: str) -> None:
        if self.logger is not None:
            self.logger.log_dataframe(
                name=self.benchmark_name, dataframe=metrics.reset_index()
            )

    def reduce(self) -> None:
        results = self.join_results(
            self.job_config.metadata.results_dir, self.glob_pattern
        )
        self.save_results(results, self.job_config.metadata.results_dir)
        metrics = self.compute_metrics(results, run_name=self.job_config.run_name)
        self.save_metrics(metrics, self.job_config.metadata.results_dir)
        parity_plot_path = self.save_parity_plot(
            results, self.job_config.metadata.results_dir
        )
        if self.logger is not None:
            self.logger.log_dataframe(
                name=f"{self.benchmark_name}-systems", dataframe=results
            )
            self.log_metrics(metrics, run_name=self.job_config.run_name)
            self.logger.log_plots(
                [parity_plot_path], caption="Predicted vs. experimental liquid density"
            )
            self.logger.log_artifact(
                name=f"{self.benchmark_name}-parity",
                type="plot",
                file_location=parity_plot_path,
            )

    def save_state(self, checkpoint_location: str, is_preemption: bool = False) -> bool:
        return False

    def load_state(self, checkpoint_location: str | None) -> None:
        return None
