"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

# Regression-suite records hold NumPy arrays, not pandas objects.
# ruff: noqa: PD011
import copy
import gc
import json
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from fairchem.core import FAIRChemCalculator
from fairchem.core.calculate import pretrained_mlip
from fairchem.core.regression_suite.artifacts.storage import (
    fragment_path,
    load_reference,
    write_json_atomic,
    write_reference,
)
from fairchem.core.regression_suite.evaluation.identity import expected_identity
from fairchem.core.regression_suite.evaluation.performance import (
    build_performance_sidecar,
)
from fairchem.core.regression_suite.execution.environment import execution_environment
from fairchem.core.regression_suite.execution.source import source_provenance
from fairchem.core.regression_suite.spec.hashing import (
    canonical_json,
    stable_hash,
)
from fairchem.core.regression_suite.spec.profiles import (
    CONFIGS,
    build_inference_settings,
    json_settings,
)
from fairchem.core.regression_suite.spec.schema import (
    CaseSpec,
    CommandError,
    MeasurementRecord,
    RegressionSchemaError,
    SuiteCase,
    SuitePlan,
)
from fairchem.core.regression_suite.spec.transforms import apply_case_transform
from fairchem.core.regression_suite.suites import load_suite

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence

    from fairchem.core.regression_suite.suites.base import Suite
    from fairchem.core.units.mlip_unit import InferenceSettings


# Roles of baseline campaign runs; candidates run as comparisons.
ROLES = ("reference", "calibration", "verification")
_RUN_ROLES = {*ROLES, "comparison"}


def _performance_sample(
    suite: Suite,
    case: SuiteCase,
    observable: str,
    values: np.ndarray,
    cluster_id: str,
) -> dict[str, Any] | None:
    metric = suite.performance_metrics.get(observable)
    if metric is None:
        return None
    coordinates = {}
    for coordinate in metric.group_by:
        value = (
            len(case.transformed_atoms)
            if coordinate == "atom_count"
            else case.spec.metadata[coordinate]
        )
        coordinates[coordinate] = (
            value.item() if isinstance(value, np.generic) else value
        )
    return {
        "metric": observable,
        "case_id": case.spec.case_id,
        "transform_id": case.spec.transform.transform_id,
        "cluster_id": cluster_id,
        "coordinates": coordinates,
        "value": float(np.asarray(values).item()),
    }


def _records_for_cases(
    suite: Suite,
    predictor,
    case_specs: Sequence[CaseSpec],
) -> tuple[list[MeasurementRecord], list[dict[str, Any]]]:
    calculators = {}
    source_cache = {}
    records = []
    performance_samples = []
    for case_spec in case_specs:
        source_key = (case_spec.structure, case_spec.structure_index)
        if source_key not in source_cache:
            source_cache[source_key] = suite.source_atoms(case_spec)
        case = apply_case_transform(source_cache[source_key], case_spec)
        calculator = calculators.get(case_spec.task_name)
        if calculator is None:
            calculator = FAIRChemCalculator(predictor, task_name=case_spec.task_name)
            calculators[case_spec.task_name] = calculator
        predictor.set_seed(case_spec.runtime_seed)
        measurements = suite.evaluate(calculator, case)
        unknown = set(measurements) - suite.observables.keys()
        if not measurements or unknown:
            raise ValueError(
                f"suite {suite.name!r} emitted unknown or no observables: {unknown}"
            )
        # Every value of one structure/transform case forms one decision cluster.
        cluster_id = f"{case_spec.case_id}::{case_spec.transform.transform_id}"
        for observable, values in measurements.items():
            spec = suite.observables[observable]
            records.append(
                MeasurementRecord(
                    case_id=case_spec.case_id,
                    transform_id=case_spec.transform.transform_id,
                    observable=observable,
                    values=values,
                    units=spec.units,
                    comparison=spec.comparison,
                    floor=spec.floor,
                    cluster_id=cluster_id,
                    report_only=spec.report_only,
                )
            )
            sample = _performance_sample(suite, case, observable, values, cluster_id)
            if sample is not None:
                performance_samples.append(sample)
    return records, performance_samples


def _performance_sidecar(
    suite: Suite,
    samples: Sequence[dict[str, Any]],
    *,
    identity: Mapping[str, Any],
) -> dict[str, Any] | None:
    """
    Build the optional, additive performance sidecar, if the run has metrics.

    CPU runs omit the metrics they cannot measure, such as GPU memory.
    """
    metric_specs = suite.performance_metrics
    if torch.device(identity["device"]).type != "cuda":
        populated = {sample["metric"] for sample in samples}
        metric_specs = {
            name: spec for name, spec in metric_specs.items() if name in populated
        }
    if not metric_specs:
        return None
    return build_performance_sidecar(
        metric_specs,
        samples,
        protocol=suite.performance_protocol,
        identity=identity,
        environment=identity["execution_environment"],
    )


def _predictor_groups(suite: Suite, cases: Sequence[CaseSpec]) -> list[list[CaseSpec]]:
    groups: dict[Hashable, list[CaseSpec]] = {}
    for case in cases:
        groups.setdefault(suite.predictor_group(case), []).append(case)
    return list(groups.values())


def _validate_effective_settings(predictor, expected: Mapping[str, Any]) -> None:
    actual = json_settings(predictor.inference_settings)
    if actual == expected:
        return
    changed = sorted(
        key
        for key in set(actual) | set(expected)
        if actual.get(key) != expected.get(key)
    )
    raise RegressionSchemaError(
        "inference settings changed during execution; "
        f"the requested profile was not preserved: {changed}"
    )


def _release_predictor(*, compiled: bool) -> None:
    if compiled:
        torch.compiler.reset()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _smoke_cases(
    suite: Suite,
    plan: SuitePlan,
) -> tuple[tuple[CaseSpec, ...], str]:
    """
    Select the suite-owned identity cases used by a local smoke check.
    """
    selected = []
    for case_id in suite.smoke_case_ids:
        matches = [
            case
            for case in plan.cases
            if case.case_id == case_id and case.transform.kind == "identity"
        ]
        if len(matches) != 1:
            raise RegressionSchemaError(
                f"smoke case {case_id!r} must select exactly one identity case; "
                f"found {len(matches)}"
            )
        selected.append(matches[0])
    selection = [
        {
            "case_id": case.case_id,
            "transform_id": case.transform.transform_id,
        }
        for case in selected
    ]
    return tuple(selected), stable_hash({"suite": suite.name, "cases": selection})


def compose_run_config(overrides: Sequence[str]) -> DictConfig:
    """
    Compose the packaged run config with Hydra command-line overrides.
    """
    with hydra.initialize_config_module(
        config_module=CONFIGS,
        version_base="1.3",
    ):
        return hydra.compose(config_name="run", overrides=list(overrides))


def task_config(
    suite: str, role: str, profile: str, values: Mapping[str, Any]
) -> DictConfig:
    """
    Compose the run config of one suite, role, and profile config or label;
    ``values`` become JSON-quoted overrides.
    """
    return compose_run_config(
        [
            f"suite={suite.replace('-', '_')}",
            f"role={role}",
            f"profile={profile.replace('-', '_')}",
            *(f"{key}={json.dumps(value)}" for key, value in values.items()),
        ]
    )


def _prepare_run(
    cfg: DictConfig,
) -> tuple[Suite, InferenceSettings, SuitePlan, dict[str, Any], Path]:
    """
    Load a run's suite, settings, and plan, and describe the run's identity.
    """
    suite = load_suite(str(cfg.suite))
    settings = build_inference_settings(cfg.profile, suite.inference_settings_overrides)
    plan = suite.plan()
    role = str(cfg.role)
    profile = str(cfg.profile.label)
    if role not in _RUN_ROLES:
        raise ValueError(f"role must be one of {sorted(_RUN_ROLES)}")
    if role == "reference" and profile != "fp64-eager":
        raise ValueError("only fp64-eager may produce a reference fragment")
    git_commit, git_dirty = source_provenance()
    identity = {
        "suite": suite.name,
        "profile": profile,
        "role": role,
        "run_id": str(cfg.get("run_id") or uuid.uuid4().hex),
        "checkpoint": str(cfg.checkpoint_id or cfg.checkpoint),
        "seed": int(cfg.seed),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "device": str(cfg.device),
        "scope": str(cfg.get("scope", "full")),
        **expected_identity(suite, plan, settings, cfg.profile),
    }
    return (
        suite,
        settings,
        plan,
        identity,
        fragment_path(cfg.output_dir, suite.name, profile, role),
    )


def _check_output(output: Path) -> None:
    if output.exists():
        raise FileExistsError(f"run already exists: {output}")


def run_suite(config: DictConfig | Mapping[str, Any]) -> dict[str, Any]:
    """
    Execute one suite/profile/role and atomically write its records.

    A smoke scope only selects the suite's smoke cases; the run is saved
    where its role's full run would be. A case shard runs one plan case into a
    launch-private directory.
    """
    cfg = OmegaConf.create(config)
    suite, settings, plan, identity, output = _prepare_run(cfg)
    identity["execution_environment"] = execution_environment(identity["device"])
    smoke = identity["scope"] == "smoke"
    shard_index = cfg.get("case_shard_index")
    case_specs = plan.cases
    if smoke:
        case_specs, selection_id = _smoke_cases(suite, plan)
        identity.update(
            full_case_count=len(plan.cases),
            selection_id=selection_id,
            case_count=len(case_specs),
        )
    elif shard_index is not None:
        case_specs = (plan.cases[shard_index],)
        identity.update(
            case_shard_index=shard_index, case_shard_count=cfg.case_shard_count
        )
        output = Path(str(cfg.case_shard_output))
    if smoke or shard_index is not None:
        identity["selected_cases"] = [
            {"case_id": case.case_id, "transform_id": case.transform.transform_id}
            for case in case_specs
        ]
    _check_output(output)

    records = []
    samples = []
    groups = (
        [case_specs]
        if identity["predictor_scope"] == "suite"
        else _predictor_groups(suite, case_specs)
    )
    for group in groups:
        predictor = pretrained_mlip.get_predict_unit(
            str(cfg.checkpoint),
            inference_settings=copy.deepcopy(settings),
            device=identity["device"],
            seed=identity["seed"],
        )
        group_records, group_samples = _records_for_cases(suite, predictor, group)
        _validate_effective_settings(predictor, identity["inference_settings"])
        records.extend(group_records)
        samples.extend(group_samples)
        del predictor
        _release_predictor(compiled=settings.compile)
    sidecar = None
    if identity["role"] != "reference" and not smoke and shard_index is None:
        sidecar = _performance_sidecar(suite, samples, identity=identity)
    # The atomic records directory is the run's completion marker; only the
    # task that publishes it writes the sidecar, so a duplicate cannot replace
    # the winner's.
    summary = write_reference(output, records, identity=identity)
    if sidecar is not None:
        write_json_atomic(output.with_name("performance.json"), sidecar)
    return summary


def merge_suite_shards(
    config: DictConfig | Mapping[str, Any],
    shard_root: str | Path,
    shard_count: int,
) -> dict[str, Any]:
    """
    Merge case-parallel work into the standard atomic suite artifact.

    Shards are launch-private full-resolution record bundles. Only this reducer
    writes the public reference or run records, so downstream readers
    remain unaware of the scheduling optimization.
    """
    cfg = OmegaConf.create(config)
    _suite, _settings, plan, identity, output = _prepare_run(cfg)
    _check_output(output)
    records = []
    environments = set()
    for index in range(shard_count):
        manifest, shard_records = load_reference(Path(shard_root) / f"{index:04d}")
        environments.add(canonical_json(manifest["identity"]["execution_environment"]))
        records.extend(shard_records)
    merged_cases = {(record.case_id, record.transform_id) for record in records}
    if merged_cases != {
        (case.case_id, case.transform.transform_id) for case in plan.cases
    }:
        raise RegressionSchemaError("merged case population differs from the plan")
    if len(environments) != 1:
        raise RegressionSchemaError("case shards used different execution environments")
    identity["execution_environment"] = json.loads(environments.pop())
    return write_reference(output, records, identity=identity)


def resolve_checkpoint(checkpoint: str, checkpoint_id: str | None) -> tuple[str, str]:
    """
    Return a loadable checkpoint and the identity recorded for it.

    Registered models are their own identity. A checkpoint path requires an
    explicit identity; the command line makes relative paths absolute before
    any job is submitted.
    """
    if checkpoint in pretrained_mlip.available_models:
        if checkpoint_id not in (None, checkpoint):
            raise CommandError(
                "checkpoint_id must match the registered checkpoint model name"
            )
        return checkpoint, checkpoint
    if not checkpoint_id:
        raise CommandError("checkpoint_id is required for a checkpoint path")
    path = Path(checkpoint).expanduser()
    if not path.is_file():
        raise CommandError(f"checkpoint does not exist: {path}")
    return str(path.resolve()), checkpoint_id


def resolve_baseline_checkpoint(
    baseline: Mapping[str, Any],
    checkpoint: str | None,
    checkpoint_id: str | None,
) -> tuple[str, str]:
    """
    Resolve the checkpoint a baseline's check runs, the baseline's own unless
    given, and require the baseline's checkpoint identity.
    """
    checkpoint, checkpoint_id = resolve_checkpoint(
        checkpoint or str(baseline["checkpoint"]), checkpoint_id
    )
    if checkpoint_id != baseline["checkpoint"]:
        raise RegressionSchemaError("checkpoint differs from the baseline's")
    return checkpoint, checkpoint_id
