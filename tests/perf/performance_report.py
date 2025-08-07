from __future__ import annotations

import itertools
import json
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field, fields
from functools import cache
from pathlib import Path
from typing import Any, Generator

import click
import numpy as np
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.collect_env import get_env_info


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
                    self.relative_change = difference / self.baseline_value
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
    """

    added: list[MeasurementChange]
    removed: list[MeasurementChange]
    increased: list[MeasurementChange]
    decreased: list[MeasurementChange]
    unchanged: list[MeasurementChange]

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
        cpu_time_sec: Data about the time spent on the CPU.
        cuda_time_sec: Data about the time spent with CUDA.
    """

    cpu_time_sec: MeasurementStats = field(default_factory=MeasurementStats)
    cuda_time_sec: MeasurementStats = field(default_factory=MeasurementStats)

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


class Environment:
    """
    Stores information about the current environment.
    """

    def __init__(self) -> None:

        # Read all available information about the environment
        env = get_env_info()

    def as_dict(self) -> dict[str, Any]:
        """
        Create a dictionary with information about the environment.

        Returns:
            Map containing details about the environment stored on this object.
        """
        return {}

@cache
def environment_singleton() -> Environment:
    """
    Cached Environment instance to avoid inspecting the environment
    multiple times (which can be slow).

    Returns:
        Reused Environment instance.
    """
    return Environment()


class PerformanceReport:
    """
    Aggregates performance metrics across various tasks and stores them in
    a consistent way.
    """

    def __init__(self) -> None:
        self._environment: Environment = environment_singleton()
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

    def get_measurement(self, measurement_name: str) -> Measurements:
        """
        Returns information about all measurements taken for the input name.

        Args:
            measurement_name: The name of the measurement to fetch. This should
                exactly match measurement_name values passed to the "measure"
                method.

        Returns:
            Details of all measurements for the input name.

        Raises:
            KeyError: If no measurements were taken for the input name.
        """
        if measurement_name not in self._measurements:
            raise KeyError(
                f"Measurement with name '{measurement_name}' is not "
                "available"
            )
        return self._measurements[measurement_name]

    @contextmanager
    def measure(self, measurement_name: str) -> Generator[None, None, None]:
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
        """

        # Track performance while control is yielded
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        ) as torch_profile, record_function(measurement_name):
            yield
        key_averages = torch_profile.key_averages()

        # Get existing measurements for the input name or create a new
        # measurement if one does not already exist
        measurement = self._measurements[measurement_name]

        # Times are measured in microseconds, convert to seconds
        measurement.cpu_time_sec.add_sample(
            sum(
                e.self_cpu_time_total
                for e in key_averages
            ) / 10**6
        )
        measurement.cuda_time_sec.add_sample(
            sum(
                e.self_device_time_total
                for e in key_averages
                if e.device_type == DeviceType.CUDA
            ) / 10**6
        )


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
    #environment_comparison = Environment.compare(
    #    target=target_report["environment"],
    #    baseline=baseline_report["environment"],
    #)
    measurements_comparison = Measurements.compare(
        target=target_report["measurements"],
        baseline=baseline_report["measurements"],
        measurement_filter=set(measurement_filter),
        metric_filter=set(metric_filter),
        stat_filter=set(stat_filter),
    )

    # Dump to json if requested
    if as_json:
        data = {
            "measurements": measurements_comparison.as_dict(),
        }
        print(json.dumps(data, indent=4))

    # Otherwise write in a formatted string
    else:

        def format_measurements(
            header: str,
            measurements: list[MeasurementChange]
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
                if any_changed:
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

        print(f"""
Measurements
------------

{format_measurements('Increased', measurements_comparison.increased)}
{format_measurements('Decreased', measurements_comparison.decreased)}
{format_measurements('Unchanged', measurements_comparison.unchanged)}
{format_measurements('Added', measurements_comparison.added)}
{format_measurements('Removed', measurements_comparison.removed)}
""")


if __name__ == "__main__":
    cli()
