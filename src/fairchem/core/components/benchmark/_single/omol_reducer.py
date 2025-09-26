"""
Copyright (c) Meta, Inc. and its affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import gzip
import json
import os
from glob import glob
from typing import Callable, TypeVar

import pandas as pd

from fairchem.core.components.benchmark.benchmark_reducer import (
    JsonDFReducer,
)

R = TypeVar("R")
M = TypeVar("M")


class OMolReducer(JsonDFReducer):
    def __init__(
        self,
        benchmark_name: str,
        evaluator: Callable | None = None,
        benchmark_labels: str | None = None,
    ):
        """
        Args:
            benchmark_name: Name of the benchmark to run + evaluate
            evaluator: Evaluation code
            benchmark_labels: Benchmark labels to run through evaluator
        """
        self.benchmark_name = benchmark_name
        self.benchmark_labels = benchmark_labels
        self.evaluator = evaluator

    def join_results(self, results_dir: str, glob_pattern: str) -> pd.DataFrame:
        """Join results from multiple JSON files into a single DataFrame.

        Args:
            results_dir: Directory containing result files
            glob_pattern: Pattern to match result files

        Returns:
            Combined DataFrame containing all results
        """
        results = {}
        for filepath in glob(os.path.join(results_dir, glob_pattern)):
            with gzip.open(filepath, "rt") as f:
                data = json.load(f)
                results.update(data)
        return results

    def save_results(self, results: pd.DataFrame, results_dir: str) -> None:
        """Save joined results to a compressed json file

        Args:
            results:  results: Combined results from join_results
            results_dir: Directory containing result files
        """
        results_path = os.path.join(
            results_dir, f"{self.benchmark_name}_results.json.gz"
        )
        with gzip.open(results_path, "wb") as f:
            f.write(json.dumps(results).encode("utf-8"))

    def compute_metrics(self, results: dict, run_name: str) -> pd.DataFrame:
        if self.benchmark_labels and self.evaluator:
            with open(self.benchmark_labels) as f:
                target = json.load(f)
            metrics = self.evaluator(target, results)
            return pd.DataFrame([metrics], index=[run_name])
        return None
