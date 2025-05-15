"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import pandas as pd
from monty.dev import requires
from pymatviz.enums import Key

from fairchem.core.components.benchmark.benchmark_reducer import JsonDFReducer
from fairchem.core.components.calculate.kappa_runner import KappaRunner

if TYPE_CHECKING:
    from collections.abc import Sequence

try:
    from matbench_discovery.metrics import phonons

    mbd_installed = True
except ImportError:
    Kappa103_TARGET_DATA_PATH = None
    mbd_installed = False


@requires(mbd_installed, message="Requires `matbench_discovery` to be installed")
class Kappa103Reducer(JsonDFReducer):
    def __init__(
        self,
        benchmark_name: str,
        target_data_path: Optional[str] = None,
        target_data_keys: Sequence[str] | None = None,
        index_name: (
            str | None
        ) = "mp_id",  # bug in matbench-discovery on column name file content mismatch: mp_id vs Key.mat_id
    ):
        """
        Args:
            benchmark_name: Name of the benchmark, used for file naming
            target_data_path: Path to the target data JSON file
            index_name: Optional name of the column to use as index
            corrections: Optional correction class to apply to all entries
            max_error_threshold: Maximum allowed mean absolute formation energy per atom error threshold
        """
        index_name = index_name or str(Key.mat_id)
        super().__init__(benchmark_name, target_data_path, target_data_keys, index_name)

    @property
    def runner_type(self) -> type[KappaRunner]:
        """The runner type this reducer is associated with."""
        return KappaRunner

    def compute_metrics(self, results: pd.DataFrame, run_name: str) -> pd.DataFrame:
        """Compute Matbench discovery metrics for relaxed energy and structure predictions.

        Args:
            results: DataFrame containing prediction results with energy values
            run_name: Identifier for the current evaluation run

        Returns:
            DataFrame containing computed metrics for different material subsets
        """
        target_data = self.target_data.rename(columns={"mp_id": Key.mat_id})
        kappa_metrics = phonons.calc_kappa_metrics_from_dfs(results, target_data)
        metrics = {
            "kappa_sre": kappa_metrics[Key.sre].mean(),
            "kappa_srme": kappa_metrics[Key.srme].mean(),
        }
        return pd.DataFrame([metrics], index=[run_name])

    def save_state(self, checkpoint_location: str, is_preemption: bool = False) -> bool:
        pass

    def load_state(self, checkpoint_location: str | None) -> None:
        pass
