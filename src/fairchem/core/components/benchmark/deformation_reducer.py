"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from typing import TypeVar

import numpy as np
import pandas as pd

from fairchem.core.components.benchmark.benchmark_reducer import JsonDFReducer

R = TypeVar("R")
M = TypeVar("M")


class DeformationReducer(JsonDFReducer):
    """Reducer for deformation benchmark results."""

    def __init__(
        self,
        benchmark_name: str,
        target_ads_key: str | None = None,
        target_int_key: str | None = None,
        target_mof_deform_key: str | None = None,
        index_name: str | None = None,
    ):
        """
        Initialize DeformationReducer.

        Args:
            benchmark_name: Name of the benchmark, used for file naming
            target_ads_key: Column name for target adsorption energies
            target_int_key: Column name for target interaction energies
            target_mof_deform_key: Column name for target MOF deformation energies
            index_name: Optional index column name for output DataFrame
        """
        self.index_name = index_name
        self.benchmark_name = benchmark_name
        self.target_ads_key = target_ads_key
        self.target_int_key = target_int_key
        self.target_mof_deform_key = target_mof_deform_key

    def compute_metrics(self, results: pd.DataFrame, run_name: str) -> pd.DataFrame:
        """
        Compute mean absolute error metrics for E_ads, E_int, and E_mof_deform.

        Args:
            results: DataFrame containing prediction results and target columns
            run_name: Name of the current run, used as index in the metrics DataFrame

        Returns:
            DataFrame containing computed metrics with run_name as index
        """
        ads_pred_col = results["E_ads"].tolist()
        int_pred_col = results["E_int"].tolist()
        mof_deform_pred_col = results["E_mof_deform"].tolist()

        ads_target = results[self.target_ads_key].tolist()
        int_target = results[self.target_int_key].tolist()
        mof_deform_target = results[self.target_mof_deform_key].tolist()

        metrics = {
            "E_ads,mae": np.mean(
                [abs(x - y) for x, y in zip(ads_pred_col, ads_target)]
            ),
            "E_int,mae": np.mean(
                [abs(x - y) for x, y in zip(int_pred_col, int_target)]
            ),
            "E_mof_deform,mae": np.mean(
                [abs(x - y) for x, y in zip(mof_deform_pred_col, mof_deform_target)]
            ),
        }

        return pd.DataFrame([metrics], index=[run_name])
