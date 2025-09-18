"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from typing import TypeVar

import pandas as pd

import numpy as np

from fairchem.core.components.benchmark.benchmark_reducer import (
    JsonDFReducer,
)

R = TypeVar("R")
M = TypeVar("M")


class DeformationReducer(JsonDFReducer):
    def __init__(
        self,
        benchmark_name: str,
        target_ads_key: str | None = None,
        target_int_key: str | None = None,
        target_mof_deform_key: str | None = None,
        index_name: str | None = None,
    ):
        """
        Args:
            benchmark_name: Name of the benchmark, used for file naming
            target_data_key: Key corresponding to the target value in the results
        """
        self.index_name = index_name
        self.benchmark_name = benchmark_name
        self.target_ads_key = target_ads_key
        self.target_int_key = target_int_key
        self.target_mof_deform_key = target_mof_deform_key

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
        ads_pred_col = results["E_ads"].tolist()
        int_pred_col = results["E_int"].tolist()
        mof_deform_pred_col = results["E_mof_deform"].tolist()

        ads_target = results[self.target_ads_key].tolist()
        int_target = results[self.target_int_key].tolist()
        mof_deform_target = results[self.target_mof_deform_key].tolist()
        
        metrics = {
            f"E_ads,mae": np.mean([np.abs(x - y) for x, y in zip(ads_pred_col, ads_target)]),
            f"E_int,mae": np.mean([np.abs(x - y) for x, y in zip(int_pred_col, int_target)]),
            f"E_mof_deform,mae": np.mean([np.abs(x - y) for x, y in zip(mof_deform_pred_col, mof_deform_target)])
        }
        return pd.DataFrame([metrics], index=[run_name])