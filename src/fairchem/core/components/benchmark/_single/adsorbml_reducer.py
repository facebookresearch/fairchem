"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from typing import TypeVar

import pandas as pd

from fairchem.core.components.benchmark.benchmark_reducer import (
    JsonDFReducer,
)

R = TypeVar("R")
M = TypeVar("M")


class AdsorbMLReducer(JsonDFReducer):
    def __init__(
        self,
        benchmark_name: str,
        target_data_key: str | None = None,
        index_name: str | None = None,
        threshold: float = 0.1,
    ):
        """
        Args:
            benchmark_name: Name of the benchmark, used for file naming
            target_data_key: Key corresponding to the target value in the results
            index_name: Name of the index for the results DataFrame
            threshold: Threshold for success rate calculation
        """
        self.index_name = index_name
        self.benchmark_name = benchmark_name
        self.target_data_key = target_data_key
        self.threshold = threshold

    def compute_metrics(self, results: pd.DataFrame, run_name: str) -> pd.DataFrame:
        """
        Compute mean absolute error metrics for common columns between results and targets.

        Args:
            results: DataFrame containing prediction results
            run_name: Name of the current run, used as index in the metrics DataFrame

        Returns:
            DataFrame containing computed metrics with run_name as index
        """
        """This will just compute MAE of everything that is common in the results and target dataframes"""
        results["diff"] = abs(
            results[f"{self.target_data_key}"]
            - results[f"{self.target_data_key}_target"]
        )
        success_rate = (results["diff"] <= self.threshold).sum() / results["diff"].size
        num_anomalies = results["anomaly_count"].sum()
        metrics = {"success_rate": success_rate, "num_anomalies": num_anomalies}
        return pd.DataFrame([metrics], index=[run_name])
