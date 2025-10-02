"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import itertools
import json
import re
import subprocess
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
from functools import cache
from pathlib import Path
from time import perf_counter
from typing import Any, Generator

import click
import numpy as np
from torch.autograd import DeviceType
from torch.cuda import is_available as is_cuda_available
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.collect_env import SystemEnv, get_env_info


class MeasurementStats:
    """
    Builds up statistics over multiple samples of the same measurement.
    """

    def __init__(self) -> None:
        self._values: list[float] = []

    def add_sample(self, value: float) -> None:
        """
        Add a new sample to this measures.

        Args:
            value: The value of the sample to add.
        """
        self._values.append(value)

    def as_dict(self) -> dict[str, int | float]:
        """
        Create a dictionary with the statistics derived from the current
        set of stored sample values.

        Returns:
            Map containing each of the statistical values for this object.
        """

        # Assume anything decorated with @property is a stat
        properties = [
            name for name, value in vars(self.__class__).items()
            if isinstance(value, property)
        ]
        return {
            prop: getattr(self, prop)
            for prop in properties
        }

    @property
    def num_samples(self) -> int:
        """
        Get the total number of samples that are stored on this object.

        Returns:
            The number of invidividual samples stored on this object.
        """
        return len(self._values)

    @property
    def min(self) -> float:
        """
        Get the minimum of all samples that are stored on this object.

        Returns:
            The minimum value over all samples stored on this object.
        """
        return float(np.min(np.array(self._values)))

    @property
    def max(self) -> float:
        """
        Get the maximum of all samples that are stored on this object.

        Returns:
            The maximum value of all samples stored on this object.
        """
        return float(np.max(np.array(self._values)))

    @property
    def mean(self) -> float:
        """
        Get the mean of all samples that are stored on this object.

        Returns:
            The mean value of all samples stored on this object.
        """
        return float(np.mean(np.array(self._values)))

    @property
    def median(self) -> float:
        """
        Get the median of all samples that are stored on this object.

        Returns:
            The median value of all samples stored on this object.
        """
        return float(np.median(np.array(self._values)))

    @property
    def std_dev(self) -> float:
        """
        Get the standard deviation of all samples that are stored on
        this object.

        Returns:
            The standard deviation value of all samples stored on this object.
        """
        return float(np.std(np.array(self._values)))

@dataclass
class MeasurementChange:
    """
    Stores information about the change in a measured value between two
    different performance reports.

    Attributes:
        measurement_name: The name of the measurement.
        stat_name: The name of the statistic being reported.
        value: The current value of the measurement. None if the value is
            not currently measured.
        baseline_value: The baseline value of the measurement. None if the
            value was not measured in the baseline report.
        relative_change: The relative change in values of the measurement.
            None if value or baseline_value is not defined.
    """

    measurement: str
    metric: str
    stat: str
    value: int | float | None
    baseline_value: int | float | None
    relative_change: float | None = field(init=False)

    def __post_init__(self) -> None:

        # Relative change is not defined if value or baseline_value is not set
        if self.value is None or self.baseline_value is None:
            self.relative_change = None
        else:

            # Set to zero if there was no change
            if (difference := self.value - self.baseline_value) == 0:
                self.relative_change = 0

            # Otherwise calculate the change
            else:
                try:
                    self.relative_change = difference / abs(self.baseline_value)
                except ZeroDivisionError:
                    if self.value >= 0:
                        self.relative_change = float("inf")
                    else:
                        self.relative_change = float("-inf")

    def as_dict(self) -> dict[str, Any]:
        """
        Create a dictionary with all of the properties stored on this object.

        Returns:
            A map of each of the values stored on this object.
        """
        return asdict(self)


@dataclass
class MeasurementChanges:
    """
    Stores information about many different changes in measurements between
    two different performance reports.

    Attributes:
        added: Measurements that were added in the target report.
        removed: Measurements that were removed from the baseline report.
        increased: Measurements whose values increased relative to the
            baseline report.
        decreased: Measurements whose values decreased relative to the
            baseline report.
        unchanged: Measurements whose values did not change between reports.
        total_changes: For each unique combination of metric and stat, the
            sum of values across all measurements.
    """

    added: list[MeasurementChange]
    removed: list[MeasurementChange]
    increased: list[MeasurementChange]
    decreased: list[MeasurementChange]
    unchanged: list[MeasurementChange]

    total_changes: list[MeasurementChange] = field(init=False)

    def __post_init__(self) -> None:

        # Sort each of the lists to make the order predictable
        self.added.sort(key=lambda m: (m.measurement, m.metric, m.stat))
        self.removed.sort(key=lambda m: (m.measurement, m.metric, m.stat))
        # "or 0" added to make type checkers happy, but they should
        # not be used since relative change is always defined in the
        # increased/decreased lists
        self.increased.sort(key=lambda m: -(m.relative_change or 0))
        self.decreased.sort(key=lambda m: m.relative_change or 0)
        self.unchanged.sort(key=lambda m: (m.measurement, m.metric, m.stat))

        # For every measurement that was present in both the baseline and
        # target performance reports, sum over all values for each unique
        # combination of metric and stat. For example, this would sum over
        # mean wall times of all measurements.
        totals: dict[tuple[str, str], float] = defaultdict(float)
        baseline_totals: dict[tuple[str, str], float] = defaultdict(float)
        all_changed = self.increased + self.decreased + self.unchanged
        for m in all_changed:
            # "or 0" added to make type checkers happy, but they should
            # not be used since we know these measurements include both
            # new and baseline values
            totals[(m.metric, m.stat)] += m.value or 0
            baseline_totals[(m.metric, m.stat)] += m.baseline_value or 0

        # Sort all totals to make them easier to consume
        self.total_changes = sorted(
            [
                MeasurementChange(
                    measurement=f"sum of '{stat}' values",
                    metric=metric,
                    stat="",
                    value=totals[(metric, stat)],
                    baseline_value=baseline_totals[(metric, stat)],
                )
                for metric, stat in totals
            ],
            key=lambda m: -abs(m.relative_change or 0)
        )

    def as_dict(self) -> dict[str, list[dict[str, int | float]]]:
        """
        Create a dictionary with all of the measurement changes stored on this
        object.

        Returns:
            A dictionary where each key represents a type of change (increase,
            decrease, etc.) and values are all measurements that changed in
            that way.
        """

        # Assume all fields for this dataclass are lists with values that each
        # have their own as_dict() method
        return {
            field.name: [
                m.as_dict()
                for m in getattr(self, field.name)
            ]
            for field in fields(self)
        }


@dataclass
class Measurements:
    """
    Stores performance measurements for a single monitored function.

    Attributes:
        wall_time_sec: Data about the total time spent on the function.
        cpu_time_sec: Data about the time spent on the CPU.
        cuda_time_sec: Data about the time spent with CUDA.
    """

    wall_time_sec: MeasurementStats = field(default_factory=MeasurementStats)
    cpu_time_sec: MeasurementStats = field(default_factory=MeasurementStats)
    cuda_time_sec: MeasurementStats = field(default_factory=MeasurementStats)

    @contextmanager
    def measure(self) -> Generator[None, None, None]:
        """
        When used in a context manager, measures performance of all
        functions called while control is yielded. When multiple calls
        are made, aggregate statistics (e.g. min, max, median, etc.)
        will be available across all of those measurements.

        Example:
            measurements = Measurements()
            with measurements.measure():
                some_expensive_function_call()
        """

        # Always track CPU performance. Also track cuda performance if
        # available.
        activities = [ProfilerActivity.CPU]
        if is_cuda_available():
            activities.append(ProfilerActivity.CUDA)

        # Track performance while control is yielded
        with profile(
            activities=activities
        ) as torch_profile, record_function("wrapper"):
            start = perf_counter()
            yield
            wall_time = perf_counter() - start
        key_averages = torch_profile.key_averages()

        # Wall time is reported in seconds
        self.wall_time_sec.add_sample(wall_time)

        # Logic here to extract time spent on cpu and gpu follows:
        # https://github.com/pytorch/pytorch/blob/c5ec5458a547f7a774468ea0eb2258d3de596492/torch/autograd/profiler_util.py#L1008-L1025
        #
        # These timings are in microseconds and converted to seconds.
        self.cpu_time_sec.add_sample(
            sum(
                e.self_cpu_time_total
                for e in key_averages
            ) / 10**6
        )
        self.cuda_time_sec.add_sample(
            sum(
                e.self_device_time_total
                for e in key_averages
                if e.device_type == DeviceType.CUDA and not e.is_user_annotation
            ) / 10**6
        )

    def as_dict(self) -> dict[str, dict[str, int | float]]:
        """
        Create a dictionary with all of the measurements stored on this object.

        Returns:
            A map of each of the measurements and their derived statistics.
        """

        # Assume all fields for this dataclass have their own as_dict() method
        return {
            field.name: getattr(self, field.name).as_dict()
            for field in fields(self)
        }

    @staticmethod
    def compare(
        target: dict[str, dict[str, dict[str, int | float]]],
        baseline: dict[str, dict[str, dict[str, int | float]]],
        measurement_filter: set[str] | None = None,
        metric_filter: set[str] | None = None,
        stat_filter: set[str] | None = None,
    ) -> MeasurementChanges:
        """
        Compares two dictionaries generated by as_dict() calls on different
        instances.

        Args:
            target: The primary measurements in the comparison.
            baseline: The baseline measurements in the comparison.
            measurement_filter: If not defined, all measurements will be
                included. Otherwise, only those measurements that appear in
                this set will be included in the comparison.
            metric_filter: If not defined, all metrics will be included.
                Otherwise, only those metrics that appear in this set will be
                included in the comparison.
            stat_filter: If not defined, all stats will be included. Otherwise,
                only those stats that appear in this set will be included in
                the comparisons.

        Returns:
            Details about all changes in measurements.
        """

        # Input is a set of nested dictionaries with the structure:
        #   {
        #     "measurement_name": {
        #       "metric_name": {
        #         "stat_name": 0
        #       }
        #     }
        #   }
        #
        # Since the specific keys could change between reports, we need to
        # discover all values present across both reports.
        all_measurements: set[str] = set()
        all_metrics: set[str] = set()
        all_stats: set[str] = set()
        measurements_items = itertools.chain(target.items(), baseline.items())
        for measurement_name, measurement_val in measurements_items:
            all_measurements.add(measurement_name)
            for metric_name, metric_val in measurement_val.items():
                all_metrics.add(metric_name)
                for stat_name in metric_val:
                    all_stats.add(stat_name)

        # Apply filters where defined
        if measurement_filter:
            all_measurements = all_measurements & measurement_filter
        if metric_filter:
            all_metrics = all_metrics & metric_filter
        if stat_filter:
            all_stats = all_stats & stat_filter

        # Organize measurements by the way in which they changed
        added: list[MeasurementChange] = []
        removed: list[MeasurementChange] = []
        increased: list[MeasurementChange] = []
        decreased: list[MeasurementChange] = []
        unchanged: list[MeasurementChange] = []
        stats_iter = itertools.product(all_measurements, all_metrics, all_stats)
        for measurement, metric, stat in stats_iter:

            # Get the measurement stat from both reports
            target_value = target.get(measurement, {}).get(metric, {}).get(stat)
            baseline_value = baseline.get(measurement, {}).get(metric, {}).get(stat)
            change = MeasurementChange(
                measurement=measurement,
                metric=metric,
                stat=stat,
                value=target_value,
                baseline_value=baseline_value,
            )

            # If both the baseline and target are None, there is nothing
            # to do
            if baseline_value is None and target_value is None:
                continue

            # If the baseline is None, the measurement is new
            if baseline_value is None:
                added.append(change)

            # If the target is None, the measurement was removed
            elif target_value is None:
                removed.append(change)

            # Otherwise organize based on the sign of the change
            elif target_value > baseline_value:
                increased.append(change)
            elif target_value < baseline_value:
                decreased.append(change)
            else:
                unchanged.append(change)

        return MeasurementChanges(
            added=added,
            removed=removed,
            increased=increased,
            decreased=decreased,
            unchanged=unchanged,
        )


@dataclass
class EnvironmentChange:
    """
    Stores information about the change in a system environment between
    different performance reports.

    Attributes:
        attribute: The name of the system attribute.
        value: The current value of the system attribute. None if the value
            is not currently gathered.
        baseline_value: The baseline value of the system attribute. None if
            the value was not gathered in the baseline report.
    """

    attribute: str
    value: str | None
    baseline_value: str | None

    def as_dict(self) -> dict[str, Any]:
        """
        Create a dictionary with all of the properties stored on this object.

        Returns:
            A map of each of the values stored on this object.
        """
        return asdict(self)


@dataclass
class EnvironmentChanges:
    """
    Stores information about many different changes in system attributes
    between two different performance reports.

    Attributes:
        added: Attributes that were added in the target report.
        removed: Attributes that were removed from the baseline report.
        changed: Attributes whose values changed relative to the baseline
            report.
        unchanged: Attributes whose values did not change between reports.
    """

    added: list[EnvironmentChange]
    removed: list[EnvironmentChange]
    changed: list[EnvironmentChange]
    unchanged: list[EnvironmentChange]

    def __post_init__(self) -> None:

        # Sort each of the lists to make the order predictable
        self.added.sort(key=lambda e: e.attribute)
        self.removed.sort(key=lambda e: e.attribute)
        self.changed.sort(key=lambda e: e.attribute)
        self.unchanged.sort(key=lambda e: e.attribute)

    def as_dict(self) -> dict[str, list[dict[str, str]]]:
        """
        Create a dictionary with all of the environment changes stored on this
        object.

        Returns:
            A dictionary where each key represents a type of change (changed,
            added, etc.) and values are all attributes that changed in
            that way.
        """

        # Assume all fields for this dataclass are lists with values that each
        # have their own as_dict() method
        return {
            field.name: [
                m.as_dict()
                for m in getattr(self, field.name)
            ]
            for field in fields(self)
        }


# Matches e.g.
#    CPU(s)             24
# And saves the numeric part in a capturing group.
_lscpu_cpu_count_pattern: re.Pattern = re.compile(r"\n\s*CPU\(s\):\s+([0-9]+)\s*[\r\n]")

# Matches e.g.
#    Model name:             Type of CPU
# And saves the "Type of CPU" in a capturing group.
_lscpu_cpu_model_pattern: re.Pattern = re.compile(r"\n\s*Model name:\s+(.*)(?!\s*[\r\n])")


@dataclass
class Environment:
    """
    Stores information about the current environment.
    """

    git_commit_hash: str = field(init=False)

    pytorch_version: str = field(init=False)
    pytorch_is_debug_build: str = field(init=False)
    cuda_version_to_build_pytorch: str = field(init=False)
    rocm_version_to_build_pytorch: str = field(init=False)

    os: str = field(init=False)
    gcc_version: str = field(init=False)
    clang_version: str = field(init=False)
    cmake_version: str = field(init=False)
    libc_version: str = field(init=False)

    python_version: str = field(init=False)
    python_platform: str = field(init=False)
    cuda_runtime_version: str = field(init=False)
    cuda_module_loading: str = field(init=False)
    nvidia_driver_version: str = field(init=False)
    cudnn_version: str = field(init=False)
    hip_runtime_version: str = field(init=False)
    miopen_runtime_version: str = field(init=False)
    xnnpack_available: str = field(init=False)

    libraries: dict[str, str] = field(init=False)

    num_gpus: str = field(init=False)
    gpu_model: str = field(init=False)
    num_cpus: str = field(init=False)
    cpu_model: str = field(init=False)

    def __post_init__(self) -> None:
        system_env = get_torch_env_info()

        self.git_commit_hash = self._get_git_commit_hash()

        self.pytorch_version = system_env.torch_version
        self.pytorch_is_debug_build = system_env.is_debug_build
        self.cuda_version_to_build_pytorch = system_env.cuda_compiled_version
        self.rocm_version_to_build_pytorch = system_env.hip_compiled_version

        self.os = system_env.os
        self.gcc_version = system_env.gcc_version
        self.clang_version = system_env.clang_version
        self.cmake_version = system_env.cmake_version
        self.libc_version = system_env.libc_version

        self.python_version = system_env.python_version
        self.python_platform = system_env.python_platform
        self.cuda_runtime_version = system_env.cuda_runtime_version
        self.cuda_module_loading = system_env.cuda_module_loading
        self.nvidia_driver_version = system_env.nvidia_driver_version
        self.cudnn_version = system_env.cudnn_version
        self.hip_runtime_version = system_env.hip_runtime_version
        self.miopen_runtime_version = system_env.miopen_runtime_version
        self.xnnpack_available = system_env.is_xnnpack_available

        # pip_packages are stored in a multiline string:
        #
        #  mypy_extensions==1.1.0
        #  numpy==2.2.6
        #  nvidia-cublas-cu12==12.4.5.8
        #  nvidia-cuda-cupti-cu12==12.4.127
        #
        # Convert to a map from package name to version. e.g.
        #  {
        #    "mypy_extensions": "1.1.0",
        #    "numpy": "2.2.6"
        #  }
        #
        # Conda packages are also stored in a multiline string:
        #
        #  numpy                     2.2.6                    pypi_0    pypi
        #  nvidia-cublas-cu12        12.4.5.8                 pypi_0    pypi
        #  nvidia-cuda-cupti-cu12    12.4.127                 pypi_0    pypi
        #  nvidia-cuda-nvrtc-cu12    12.4.127                 pypi_0    pypi
        #
        # Also convert them to a map from package name to version:
        #  {
        #    "numpy": "2.2.6",
        #    "nvidia-cublas-cu12 ": "12.4.5.8"
        #  }
        #
        # Then merge both dictionaries.
        self.libraries = {
          package_version[0]: package_version[1]
          for line in system_env.pip_packages.splitlines()
          if len(package_version := line.split("==")) == 2
        } | {
            package_version[0]: package_version[1]
            for line in system_env.conda_packages.splitlines()
            if len(package_version := line.split()) >= 2
        }

        # nvidia_gpu_models is a multiline string with lines for each gpu:
        #
        #  GPU 0: Quadro GV100
        #  GPU 1: Quadro GV100
        #
        # Count the number of GPUs and save the types.
        self.num_gpus = str(len(system_env.nvidia_gpu_models.splitlines()))
        self.gpu_model = (
            # Get the unique GPU types. For any situation in which there is
            # not exactly one type, mark as unknown.
            list(gpu_models)[0]
            if len(gpu_models := {
                parts[1].strip()
                for line in system_env.nvidia_gpu_models.splitlines()
                if len(parts := line.split(":")) > 1
            }) == 1
            else "Unknown"
        )

        # CPU details on linux machines are direct outputs from lscpu. Fetch
        # a subset representing the most important fields.
        self.num_cpus = (
            match.group(1)
            if (match := _lscpu_cpu_count_pattern.search(system_env.cpu_info))
            else "Unknown"
        )
        self.cpu_model = (
            match.group(1)
            if (match := _lscpu_cpu_model_pattern.search(system_env.cpu_info))
            else "Unknown"
        )

    def _get_git_commit_hash(self) -> str:
        """
        Tries to detect the current git commit hash.
        """
        try:
            result = subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                text=True,
            ).strip()
        except Exception:
            result = ""

        # Return Unknown for non-zero exits as well as empty returns
        return result or "Unknown"

    def as_dict(self) -> dict[str, Any]:
        """
        Create a dictionary with information about the environment.

        Returns:
            Map containing details about the environment stored on this object.
        """
        return asdict(self)

    @staticmethod
    def compare(
        target: dict[str, str | dict[str, str]],
        baseline: dict[str, str | dict[str, str]],
    ) -> EnvironmentChanges:
        """
        Compares two dictionaries generated by as_dict() calls on different
        instances.

        Args:
            target: The primary environment in the comparison.
            baseline: The baseline environment in the comparison.

        Returns:
            Details about all changes in environments.
        """

        # Input is a dictionary where values can be strings or dictionaries.
        #  {
        #    "attribute_name_1": "attribute_value_1",
        #    "attribute_name_2": {
        #      "sub_attribute_name": "sub_attribute_value"
        #    }
        #  }
        #
        # Since the specific keys could change between reports, we need to
        # discover all values present across both reports.
        all_attributes: set[str] = set()
        for name, value in itertools.chain(target.items(), baseline.items()):
            if isinstance(value, dict):
                all_attributes.update(f"{name}.{sub}" for sub in value)
            else:
                all_attributes.add(name)

        # Helper function to get an attribute value from the input environment
        # dictionary. Supports nested attribute paths.
        def get_value(
            environment: dict[str, str | dict[str, str]],
            attribute_name: str,
        ) -> str | None:
            path = attribute_name.split(".")

            # Check for a nested attribute
            if isinstance(value := environment.get(path[0]), dict):
                assert len(path) == 2
                return value.get(path[1])

            # This is a nested attribute where the parent does not exist
            if value is None and len(path) == 2:
                return value

            # Otherwise this is a root level attribute
            assert len(path) == 1
            return value

        # Organize attributes by the way in which they changed
        added: list[EnvironmentChange] = []
        removed: list[EnvironmentChange] = []
        changed: list[EnvironmentChange] = []
        unchanged: list[EnvironmentChange] = []
        for attribute in all_attributes:

            # Get the measurement stat from both reports
            target_value = get_value(target, attribute)
            baseline_value = get_value(baseline, attribute)
            change = EnvironmentChange(
                attribute=attribute,
                value=target_value,
                baseline_value=baseline_value,
            )

            # If both the baseline and target are None, there is nothing
            # to do
            if baseline_value is None and target_value is None:
                continue

            # If the baseline is None, the attribute is new
            if baseline_value is None:
                added.append(change)

            # If the target is None, the attribute was removed
            elif target_value is None:
                removed.append(change)

            # Otherwise capture whether changed or not
            elif target_value != baseline_value:
                changed.append(change)
            else:
                unchanged.append(change)

        return EnvironmentChanges(
            added=added,
            removed=removed,
            changed=changed,
            unchanged=unchanged,
        )

@cache
def get_torch_env_info() -> SystemEnv:
    """
    Returns the system information reported by torch. Cached because this
    can be slow to generate.

    Returns:
        SystemEnv instance from torch.
    """
    return get_env_info()


class PerformanceReport:
    """
    Aggregates performance metrics across various tasks and stores them in
    a consistent way.
    """

    def __init__(self) -> None:
        self._environment: Environment = Environment()
        self._measurements: dict[str, Measurements] = defaultdict(Measurements)

    def as_dict(self) -> dict[str, Any]:
        """
        Get a dictionary with all of the captured information.

        Returns:
            A map containing all of the information tracked in this report.
        """
        return {
            "environment": self._environment.as_dict(),
            "measurements": {
                measurement_name: measurement.as_dict()
                for measurement_name, measurement in self._measurements.items()
            }
        }

    @contextmanager
    def measure(self, measurement_name: str) -> Generator[Measurements, None, None]:
        """
        When used in a context manager, measures performance of all functions
        called while control is yielded. Results are organized based on the
        input measurement names. When multiple calls are made with the same
        measurement name, aggregate statistics (e.g. min, max, median, etc.)
        will be available across all of those measurements.

        Example:
            report = PerformanceReport()
            with report.measure("my_test_function"):
                some_expensive_function_call()

        Args:
            measurement_name: A name used to distinguish different measurements
                being tracked in the same report. Aggregate statistics are
                available when the same name is used multiple times.

        Yields:
            The Measurements instance being constructed for the input name.
        """

        # Get existing measurements for the input name or create a new
        # measurement if one does not already exist
        measurements = self._measurements[measurement_name]
        with measurements.measure():
            yield measurements


@click.group()
def cli() -> None:
    """
    Process saved performance reports.
    """


@cli.command
@click.argument(
    "target",
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
)
@click.option(
    "--baseline",
    required=True,
    type=click.Path(exists=True, file_okay=True, dir_okay=False, path_type=Path),
    help=(
        "Path to the baseline performance report. Changes will calculated "
        "relative to this file. For example, if comparing a target report "
        "A with value x=1 to a baseline report B with x=2, the comparison "
        "will indicate that x decreased from 2 to 1."
    ),
)
@click.option(
    "--measurement",
    "measurement_filter",
    type=str,
    multiple=True,
    help=(
        "By default, if this option is not set, all measurements will be "
        "included in the comparison. Use this to limit to a subset of "
        "measurements. This option can be passed multiple times to set more "
        "than one stat name."
    )
)
@click.option(
    "--metric",
    "metric_filter",
    type=str,
    multiple=True,
    help=(
        "By default, if this option is not set, all metrics will be included "
        "in the comparison. Use this to limit to a subset of metrics. This "
        "option can be passed multiple times to set more than one stat name."
    )
)
@click.option(
    "--stat",
    "stat_filter",
    type=str,
    multiple=True,
    help=(
        "By default, if this option is not set, all stats will be included in "
        "the comparison. Use this to limit to a subset of stats. This option "
        "can be passed multiple times to set more than one stat name."
    )
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Print output formatted as a JSON object."
)
def compare(
    target: Path,
    baseline: Path,
    measurement_filter: list[str],
    metric_filter: list[str],
    stat_filter: list[str],
    as_json: bool,
) -> None:
    """
    Compare details from two performance reports.

    TARGET gives the path to the target performance report in the comparison.
    For example, if comparing a target report A with value x=1 to a
    baseline report B with x=2, the comparison will indicate that x
    decreased from 2 to 1.
    """

    # Load each of the performance reports
    target_report = json.loads(target.read_bytes())
    baseline_report = json.loads(baseline.read_bytes())

    # Compare different parts of the performance report
    environment_comparison = Environment.compare(
        target=target_report.get("environment", {}),
        baseline=baseline_report.get("environment", {}),
    )
    measurements_comparison = Measurements.compare(
        target=target_report.get("measurements"),
        baseline=baseline_report.get("measurements"),
        measurement_filter=set(measurement_filter),
        metric_filter=set(metric_filter),
        stat_filter=set(stat_filter),
    )

    # Dump to json if requested
    if as_json:
        data = {
            "measurements": measurements_comparison.as_dict(),
            "environment": environment_comparison.as_dict(),
        }
        print(json.dumps(data, indent=4))

    # Otherwise write in a formatted string
    else:

        def format_measurements(
            header: str,
            measurements: list[MeasurementChange],
            force_relative_change: bool = False,
        ) -> str:

            # Avoid errors below for empty lists
            if not measurements:
                return header

            # Get the max length of each identifier to help with formatting
            max_measurement_len = max([len(m.measurement) for m in measurements])
            max_metric_len = max([len(m.metric) for m in measurements])
            max_stat_len = max([len(m.stat) for m in measurements])

            # Check if any of the metric values changed. We'll include the
            # relative change if they did
            any_changed = any(m.relative_change for m in measurements)

            # Build the measurements line by line
            lines: list[str] = []
            for m in measurements:

                # Add the relative change if needed
                line: str = "  "
                if any_changed or force_relative_change:
                    line += f"{100*(m.relative_change or 0):9.4f}%"

                # Add identifiers
                line += f"{m.measurement.rjust(max_measurement_len+3)}"
                line += f"{m.metric.rjust(max_metric_len+3)}"
                line += f"{m.stat.rjust(max_stat_len+3)}"

                # Add actual values
                values: list[str] = []
                if m.baseline_value is not None:
                    values.append(str(m.baseline_value))
                if m.value is not None:
                    values.append(str(m.value))
                if len(values) > 1 and len(set(values)) == 1:
                    values = values[:1]
                if values:
                    line += f"   {' -> '.join(values)}"

                lines.append(line)

            return "\n".join([header] + lines + [""])

        def format_environment(
            header: str,
            attributes: list[EnvironmentChange],
        ) -> str:

            # Avoid errors below for empty lists
            if not attributes:
                return header

            # Get the max length of each attribute to help with formatting
            max_attribute_len = max([len(a.attribute) for a in attributes])

            # Build the results line by line
            lines: list[str] = []
            for a in attributes:

                # Add attribute name
                line = f"  {a.attribute.rjust(max_attribute_len)}"

                # Add attribute values
                values: list[str] = []
                if a.baseline_value is not None:
                    values.append(str(a.baseline_value))
                if a.value is not None:
                    values.append(str(a.value))
                if len(values) > 1 and len(set(values)) == 1:
                    values = values[:1]
                if values:
                    line += f"   {' -> '.join(values)}"

                lines.append(line)

            return "\n".join([header] + lines + [""])


        print(f"""
--------------
 MEASUREMENTS
--------------

+ {format_measurements('Increased', measurements_comparison.increased)}
+ {format_measurements('Decreased', measurements_comparison.decreased)}
+ {format_measurements('Unchanged', measurements_comparison.unchanged)}
+ {format_measurements('Added', measurements_comparison.added)}
+ {format_measurements('Removed', measurements_comparison.removed)}
+ {format_measurements('Totals (excludes added and removed measurements)', measurements_comparison.total_changes, True)}

-------------
 ENVIRONMENT
-------------

+ {format_environment('Changed', environment_comparison.changed)}
+ {format_environment('Unchanged', environment_comparison.unchanged)}
+ {format_environment('Added', environment_comparison.added)}
+ {format_environment('Removed', environment_comparison.removed)}
""")


if __name__ == "__main__":
    cli()
