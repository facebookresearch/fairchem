"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging
import os
from abc import ABCMeta, abstractmethod
from glob import glob
from typing import TYPE_CHECKING, TypeVar

import numpy as np
import pandas as pd

from fairchem.core.common import distutils
from fairchem.core.common.logger import WandBSingletonLogger
from fairchem.core.components.calculate.calculate_runner import (
    CalculateRunner,
)
from fairchem.core.components.reducer import Reducer

if TYPE_CHECKING:
    from collections.abc import Sequence

R = TypeVar("R")
M = TypeVar("M")


class BenchmarkReducer(Reducer, metaclass=ABCMeta):
    """Benchmark reducer interface class.

    Note:
        When running with the `fairchemv2` cli, the `job_config` and `runner_config` attributes are set at
        runtime to those given in the config file.

    See the Reducer interface class for implementation details.

    Attributes:
        job_config (DictConfig): a managed attribute that gives access to the job config
        runner_config (DictConfig): a managed attributed that gives access to the calling runner config
    """

    @property
    def runner_type(self) -> type[CalculateRunner]:
        """The runner type this reducer is associated with."""
        return CalculateRunner

    @property
    def glob_pattern(self):
        """Returns the glob pattern used to find result files from the runner."""
        return self.runner_type.result_glob_pattern

    @property
    def logger(self) -> WandBSingletonLogger | None:
        """Returns a logger instance if conditions are met, otherwise None.

        Returns:
            WandBSingletonLogger or None: Logger instance if running on main rank with logging enabled
        """
        if (
            distutils.is_master()
            and self.job_config is not None
            and not self.job_config.debug
            and self.job_config.logger
        ):
            return WandBSingletonLogger.get_instance()
        else:
            return None

    @abstractmethod
    def join_results(self, results_dir: str, glob_pattern: str) -> R:
        """Join results from multiple files into a single result object.

        Args:
            results_dir: Directory containing result files
            glob_pattern: Pattern to match result files

        Returns:
            Combined results object of type R
        """
        raise NotImplementedError

    @abstractmethod
    def save_results(self, results: R, results_dir: str) -> None:
        """Save joined results to file

        Args:
            results:  results: Combined results from join_results
            results_dir: Directory containing result files
        """
        raise NotImplementedError

    @abstractmethod
    def compute_metrics(self, results: R, run_name: str) -> M:
        """Compute metrics from the joined results.

        Args:
            results: Combined results from join_results
            run_name: Name of the current run

        Returns:
            Metrics object of type M
        """
        raise NotImplementedError

    @abstractmethod
    def save_metrics(self, metrics: M, results_dir: str) -> None:
        """Save computed metrics to disk.

        Args:
            metrics: Metrics object to save
            results_dir: Directory to save metrics to
        """
        raise NotImplementedError

    @abstractmethod
    def log_metrics(self, metrics: M, run_name: str):
        """Log metrics to the configured logger.

        Args:
            metrics: Metrics object to log
            run_name: Name of the current run
        """
        raise NotImplementedError

    @abstractmethod
    def save_state(self, checkpoint_location: str, is_preemption: bool = False) -> bool:
        """Save the current state of the reducer to a checkpoint.

        Args:
            checkpoint_location: Location to save the checkpoint
            is_preemption: Whether the save is due to preemption

        Returns:
            bool: Success status of the save operation
        """
        raise NotImplementedError

    @abstractmethod
    def load_state(self, checkpoint_location: str | None) -> None:
        """Load reducer state from a checkpoint.

        Args:
            checkpoint_location: Location to load the checkpoint from, or None
        """
        raise NotImplementedError

    def reduce(self):
        """Join results, compute metrics, save and log resulting metrics.

        Note:
            Re-implementing this method in derived classes is discouraged.
        """
        # re-implementing  this method in derived classes is discouraged
        logging.info(
            f"Joining calculation results in {self.job_config.metadata.results_dir}"
        )
        results = self.join_results(
            self.job_config.metadata.results_dir, self.glob_pattern
        )
        logging.info(
            f"Saving joined results in {self.job_config.metadata.results_dir}."
        )
        self.save_results(results, self.job_config.metadata.results_dir)
        logging.info("Calculating metrics.")
        metrics = self.compute_metrics(results, run_name=self.job_config.run_name)
        if metrics is not None:
            logging.info(
                f"Saving computed metrics in {self.job_config.metadata.results_dir}"
            )
            self.save_metrics(metrics, self.job_config.metadata.results_dir)
            if self.logger is not None:
                self.log_metrics(metrics, run_name=self.job_config.run_name)
        else:
            logging.info(
                "Evaluation labels not available, skipping metrics computation. Please submit results to the appropriate benchmark leaderboard if applicable."
            )


class JsonDFReducer(BenchmarkReducer):
    """A common pandas DataFrame reducer for benchmarks

    Results are assumed to be saved as json files that can be read into pandas dataframes.
    Only mean absolute error is computed for common columns in the predicted results and target data
    """

    def __init__(
        self,
        benchmark_name: str,
        target_data_path: str | None = None,
        target_data_keys: Sequence[str] | None = None,
        index_name: str | None = None,
    ):
        """Initialize the JsonDFReducer with benchmark data and configuration.

        Args:
            benchmark_name: Name of the benchmark, used for file naming
            target_data_path: Path to the target data JSON file
            target_data_keys: List of target property keys to extract from results file
            index_name: Optional name of the column to use as index
        """

        if target_data_path is None and target_data_keys is None:
            raise ValueError(
                "Either target_data_path or target_data_keys should be provided"
            )

        self.index_name = index_name
        self.benchmark_name = benchmark_name
        self.target_data = (
            self.load_targets(target_data_path, index_name)
            if target_data_path is not None
            else None
        )
        self.target_data_keys = target_data_keys

    @staticmethod
    def load_targets(path: str, index_name: str | None) -> pd.DataFrame:
        """Load target data from a JSON file into a pandas DataFrame.

        Args:
            path: Path to the target JSON file
            index_name: Optional name of the column to use as index

        Returns:
            DataFrame containing the target data, sorted by index
        """
        df_targets = pd.read_json(path, dtype=False)
        if index_name is not None:
            df_targets = df_targets.set_index(index_name)
        return df_targets.sort_index()

    def join_results(self, results_dir: str, glob_pattern: str) -> pd.DataFrame:
        """Join results from multiple JSON files into a single DataFrame.

        Args:
            results_dir: Directory containing result files
            glob_pattern: Pattern to match result files

        Returns:
            Combined DataFrame containing all results
        """
        results = pd.concat(
            [
                pd.read_json(f, dtype=False)
                for f in glob(os.path.join(results_dir, glob_pattern))
            ]
        ).reset_index()

        if self.index_name is not None:
            results = results.set_index("sid").sort_index()
            results.index.name = self.index_name

        return results

    def save_results(self, results: pd.DataFrame, results_dir: str) -> None:
        """Save joined results to a compressed json file

        Args:
            results:  results: Combined results from join_results
            results_dir: Directory containing result files
        """
        results.reset_index().to_json(
            os.path.join(results_dir, f"{self.benchmark_name}_results.json.gz")
        )

    def compute_metrics(self, results: pd.DataFrame, run_name: str) -> pd.DataFrame:
        """Compute mean absolute error metrics for common columns between results and targets.

        Args:
            results: DataFrame containing prediction results
            run_name: Name of the current run, used as index in the metrics DataFrame

        Returns:
            DataFrame containing computed metrics with run_name as index
        """
        """This will just compute MAE of everything that is common in the results and target dataframes"""

        metrics = {}
        if self.target_data is not None:
            common_cols = [
                col for col in results.columns if col in self.target_data.columns
            ]
            metrics.update(
                {
                    f"{col},mae": (results[col] - self.target_data[col]).abs().mean()
                    for col in common_cols
                }
            )

        if self.target_data_keys is not None:
            for target_name in self.target_data_keys:
                # TODO: For now we'll hardcode forces, but a more general
                # approach for arbitrary metrics + properties should be
                # incorporated
                if target_name == "forces":
                    forces = np.concatenate(results[target_name].values)
                    forces_norm = np.linalg.norm(forces, axis=1)
                    forces_target = np.concatenate(
                        results[f"{target_name}_target"].values
                    )
                    forces_target_norm = np.linalg.norm(forces_target, axis=1)

                    metrics[f"{target_name},mae"] = np.mean(
                        np.abs(forces - forces_target)
                    )
                    metrics[f"{target_name},cosine_similarity"] = np.sum(
                        forces_target * forces
                    ) / max(
                        np.linalg.norm(forces_target) * np.linalg.norm(forces), 1e-8
                    )
                    metrics[f"{target_name},magnitude_error"] = np.mean(
                        np.abs(forces_norm - forces_target_norm)
                    )
                else:
                    metrics[f"{target_name},mae"] = (
                        (results[target_name] - results[f"{target_name}_target"])
                        .abs()
                        .mean()
                    )

        return pd.DataFrame([metrics], index=[run_name])

    def save_metrics(self, metrics: pd.DataFrame, results_dir: str) -> None:
        """Save computed metrics to a compressed JSON file.

        Args:
            metrics: DataFrame containing the computed metrics
            results_dir: Directory where metrics will be saved
        """
        metrics.to_json(
            os.path.join(results_dir, f"{self.benchmark_name}_metrics.json.gz")
        )

    def log_metrics(self, metrics: pd.DataFrame, run_name: str) -> None:
        """Log metrics to the configured logger if available.

        Args:
            metrics: DataFrame containing the computed metrics
            run_name: Name of the current run
        """
        if self.logger is not None:
            self.logger.log_dataframe(
                name=self.benchmark_name, dataframe=metrics.reset_index()
            )

    def save_state(self, checkpoint_location: str, is_preemption: bool = False) -> bool:
        pass

    def load_state(self, checkpoint_location: str | None) -> None:
        pass
