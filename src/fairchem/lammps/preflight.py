"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License version 2 as
published by the Free Software Foundation. See LICENSE.md in this
directory for the full license.

Validate and benchmark the FAIR-Chem LAMMPS integration.
"""

from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import math
import shlex
import statistics
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import hydra
import numpy as np
import torch
from ase.build import bulk
from ase.calculators.lammps.coordinatetransform import Prism
from ase.io import read, write
from omegaconf import DictConfig, OmegaConf

from fairchem.core.calculate.pretrained_mlip import (
    pretrained_checkpoint_path_from_name,
)
from fairchem.core.common import distutils
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.units.mlip_unit import InferenceSettings, load_predict_unit

if TYPE_CHECKING:
    from ase import Atoms

    from fairchem.core.units.mlip_unit.predict import MLIPPredictUnitProtocol


BRIDGE_ATOL = 1.0e-5
MAX_TIMING_CV = 0.10
MIN_RECOMMENDATION_SPEEDUP = 1.10
DEFAULT_MAX_ENERGY_ERROR_MEV_PER_ATOM = 1.0
DEFAULT_MAX_FORCE_MAE_EV_PER_A = 5.0e-3
DEFAULT_MAX_FORCE_ERROR_EV_PER_A = 2.0e-2


@dataclass(frozen=True)
class InferenceProfile:
    """
    User-relevant inference configuration for LAMMPS MD.
    """

    name: str
    tf32: bool
    compile: bool
    activation_checkpointing: bool

    def settings(self, graph_version: int) -> InferenceSettings:
        """
        Materialize this profile while retaining automatic backend selection.
        """
        return InferenceSettings(
            tf32=self.tf32,
            compile=self.compile,
            activation_checkpointing=self.activation_checkpointing,
            merge_mole=True,
            external_graph_gen=False,
            internal_graph_gen_version=graph_version,
            execution_mode=None,
        )


PROFILES = (
    InferenceProfile("fp32_eager", False, False, False),
    InferenceProfile("tf32_eager", True, False, False),
    InferenceProfile("fp32_compiled", False, True, False),
    InferenceProfile("turbo", True, True, False),
    InferenceProfile("memory_saving", False, False, True),
)
PROFILE_BY_NAME = {profile.name: profile for profile in PROFILES}


@dataclass(frozen=True)
class BenchmarkCase:
    """
    One profile, graph generator, and GPU-worker combination.
    """

    profile: str
    graph_version: int
    workers: int


class RecordingPredictor:
    """
    Record the exact input and output passed through the LAMMPS callback.
    """

    def __init__(self, predictor: MLIPPredictUnitProtocol):
        self.predictor = predictor
        self.last_input: AtomicData | None = None
        self.last_results: dict[str, torch.Tensor] | None = None
        self.last_error: Exception | None = None

    def predict(self, data: AtomicData) -> dict[str, torch.Tensor]:
        self.last_input = data.clone()
        self.last_error = None
        try:
            results = self.predictor.predict(data)
        except Exception as error:
            # ctypes callbacks cannot propagate Python exceptions through
            # LAMMPS. Retain the original failure so the pre-flight report can
            # surface it after LAMMPS returns.
            self.last_error = error
            raise
        self.last_results = {
            key: value.detach().cpu().clone()
            for key, value in results.items()
            if isinstance(value, torch.Tensor)
        }
        return results

    def validate_atoms_data(self, atoms: Atoms, task_name: str) -> None:
        self.predictor.validate_atoms_data(atoms, task_name)

    @property
    def inference_settings(self) -> InferenceSettings:
        return self.predictor.inference_settings

    @property
    def resolved_execution_mode(self) -> str:
        """
        Read the effective backend from the rank that loaded the CUDA model.
        """
        settings = self.predictor.inference_settings
        local_rank0 = getattr(self.predictor, "local_rank0", None)
        if local_rank0 is not None and hasattr(local_rank0, "predict_unit"):
            settings = local_rank0.predict_unit.inference_settings
        return str(settings.execution_mode or "general")


def build_case_matrix(
    gpu_count: int,
    worker_counts: list[int] | None = None,
) -> list[BenchmarkCase]:
    """
    Build the focused GPU benchmark matrix.

    Graph generator v3 is measured only on one GPU. Graph generator v2 is
    measured for every requested GPU count because it is the parallel-oriented
    implementation.
    """
    if gpu_count < 1:
        raise ValueError("At least one CUDA GPU is required.")
    if worker_counts is not None and not worker_counts:
        raise ValueError("Worker counts cannot be empty.")
    counts = (
        [1, *worker_counts]
        if worker_counts is not None
        else list(range(1, gpu_count + 1))
    )
    if not counts or any(count < 1 or count > gpu_count for count in counts):
        raise ValueError(
            f"Worker counts must be between 1 and the {gpu_count} visible GPUs."
        )
    counts = sorted(set(counts))
    cases = [
        BenchmarkCase(profile.name, graph_version=2, workers=workers)
        for workers in counts
        for profile in PROFILES
    ]
    if 1 in counts:
        cases.extend(
            BenchmarkCase(profile.name, graph_version=3, workers=1)
            for profile in PROFILES
        )
    return cases


def projected_runtime_seconds(result: dict[str, Any], expected_steps: int) -> float:
    """
    Project a production run using observed startup, warm-up, and steady state.
    """
    warmup_steps = int(result["warmup_steps"])
    setup = float(result["initialization_seconds"]) + float(
        result["cold_start_seconds"]
    )
    warmup = float(result["warmup_seconds"])
    steady_step = float(result["median_ms_per_step"]) / 1000.0
    if expected_steps <= warmup_steps:
        return setup + warmup * expected_steps / max(warmup_steps, 1)
    return setup + warmup + (expected_steps - warmup_steps) * steady_step


def add_break_even_steps(results: list[dict[str, Any]]) -> None:
    """
    Add steady-state break-even estimates relative to FP32 eager on one GPU.

    The estimate starts after the measured warm-up window and is meaningful
    only for cases with a lower steady-state step time than the baseline.
    """
    baseline = next(
        (
            result
            for result in results
            if result.get("status") == "passed"
            and result["profile"] == "fp32_eager"
            and result["graph_version"] == 2
            and result["workers"] == 1
        ),
        None,
    )
    if baseline is None:
        return
    baseline_startup = sum(
        float(baseline[key])
        for key in ("initialization_seconds", "cold_start_seconds", "warmup_seconds")
    )
    baseline_step = float(baseline["median_ms_per_step"]) / 1000.0
    warmup_steps = int(baseline["warmup_steps"])
    for result in results:
        if result.get("status") != "passed" or "median_ms_per_step" not in result:
            continue
        step = float(result["median_ms_per_step"]) / 1000.0
        if step >= baseline_step:
            result["break_even_steps_vs_fp32_eager"] = None
            continue
        startup = sum(
            float(result[key])
            for key in (
                "initialization_seconds",
                "cold_start_seconds",
                "warmup_seconds",
            )
        )
        additional_steps = max(
            0.0, (startup - baseline_startup) / (baseline_step - step)
        )
        result["break_even_steps_vs_fp32_eager"] = math.ceil(
            warmup_steps + additional_steps
        )


def add_numerical_comparison(
    result: dict[str, Any],
    reference: dict[str, np.ndarray],
    num_atoms: int,
    acceptance: dict[str, float],
) -> None:
    """
    Add accuracy metrics and a pass/fail decision to a completed case.
    """
    energy = np.asarray(result.pop("_energy"))
    forces = np.asarray(result.pop("_forces"))
    energy_ref = reference["energy"]
    forces_ref = reference["forces"]
    energy_error = float(np.max(np.abs(energy - energy_ref)))
    result["energy_absolute_error_eV"] = energy_error
    result["energy_error_meV_per_atom"] = 1000.0 * energy_error / num_atoms
    force_delta = forces - forces_ref
    result["force_mae_eV_per_A"] = float(np.mean(np.abs(force_delta)))
    result["force_max_error_eV_per_A"] = float(np.max(np.abs(force_delta)))
    result["numerical_acceptance"] = acceptance
    result["numerical_passed"] = bool(
        result["energy_error_meV_per_atom"]
        <= acceptance["max_energy_error_meV_per_atom"]
        and result["force_mae_eV_per_A"] <= acceptance["max_force_mae_eV_per_A"]
        and result["force_max_error_eV_per_A"] <= acceptance["max_force_error_eV_per_A"]
    )


def numerical_acceptance(
    *,
    max_energy_error_meV_per_atom: float = DEFAULT_MAX_ENERGY_ERROR_MEV_PER_ATOM,
    max_force_mae_eV_per_A: float = DEFAULT_MAX_FORCE_MAE_EV_PER_A,
    max_force_error_eV_per_A: float = DEFAULT_MAX_FORCE_ERROR_EV_PER_A,
) -> dict[str, float]:
    """Build and validate the accuracy limits used for recommendations."""
    acceptance = {
        "max_energy_error_meV_per_atom": max_energy_error_meV_per_atom,
        "max_force_mae_eV_per_A": max_force_mae_eV_per_A,
        "max_force_error_eV_per_A": max_force_error_eV_per_A,
    }
    if any(value < 0 for value in acceptance.values()):
        raise ValueError("Numerical acceptance limits must be non-negative.")
    return acceptance


def recommend_case(
    results: list[dict[str, Any]], expected_steps: int
) -> dict[str, Any] | None:
    """
    Select a trustworthy case, requiring material benefit for added complexity.
    """
    eligible = [
        result
        for result in results
        if result.get("status") == "passed"
        and result.get("numerical_passed")
        and result.get("timing_cv", math.inf) <= MAX_TIMING_CV
    ]
    if not eligible:
        return None
    for result in eligible:
        result["projected_runtime_seconds"] = projected_runtime_seconds(
            result, expected_steps
        )

    def select_worker_count(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
        """Add GPUs only when the next choice is materially faster."""
        if not candidates:
            return None
        selected = None
        for workers in sorted({result["workers"] for result in candidates}):
            candidate = min(
                (result for result in candidates if result["workers"] == workers),
                key=lambda result: result["projected_runtime_seconds"],
            )
            if selected is None or (
                selected["projected_runtime_seconds"]
                / candidate["projected_runtime_seconds"]
                >= MIN_RECOMMENDATION_SPEEDUP
            ):
                selected = candidate
        return selected

    eager = select_worker_count(
        [
            result
            for result in eligible
            if not PROFILE_BY_NAME[result["profile"]].compile
        ]
    )
    compiled = select_worker_count(
        [result for result in eligible if PROFILE_BY_NAME[result["profile"]].compile]
    )
    if eager is None or (
        compiled is not None
        and eager["projected_runtime_seconds"] / compiled["projected_runtime_seconds"]
        >= MIN_RECOMMENDATION_SPEEDUP
    ):
        fastest = compiled
    else:
        fastest = eager
    if fastest is None:
        return None
    baseline = next(
        (
            result
            for result in eligible
            if result["profile"] == "fp32_eager"
            and result["graph_version"] == 2
            and result["workers"] == 1
        ),
        None,
    )
    if baseline is not None:
        speedup = (
            baseline["projected_runtime_seconds"] / fastest["projected_runtime_seconds"]
        )
        if speedup < MIN_RECOMMENDATION_SPEEDUP:
            fastest = baseline
            speedup = 1.0
    else:
        speedup = None
    return {
        "profile": fastest["profile"],
        "graph_version": fastest["graph_version"],
        "workers": fastest["workers"],
        "resolved_execution_mode": fastest["resolved_execution_mode"],
        "projected_runtime_seconds": fastest["projected_runtime_seconds"],
        "speedup_vs_fp32_eager": speedup,
        "hydra_overrides": hydra_overrides_for_case(fastest),
    }


def hydra_overrides_for_case(result: dict[str, Any]) -> list[str]:
    """
    Produce copyable overrides for the selected LAMMPS configuration.
    """
    profile = PROFILE_BY_NAME[result["profile"]]
    overrides = [
        f"if_settings.tf32={str(profile.tf32).lower()}",
        f"if_settings.compile={str(profile.compile).lower()}",
        "if_settings.merge_mole=true",
        "if_settings.activation_checkpointing="
        f"{str(profile.activation_checkpointing).lower()}",
        f"if_settings.internal_graph_gen_version={result['graph_version']}",
    ]
    if result["workers"] > 1:
        overrides.extend(
            [
                "predict_unit=${parallel_predict_unit}",
                f"parallel_predict_unit.num_workers={result['workers']}",
            ]
        )
    else:
        overrides.append("predict_unit=${local_predict_unit}")
    model = result.get("model")
    if model:
        model_key = (
            "parallel_predict_unit.inference_model_path"
            if result["workers"] > 1
            else "local_predict_unit.path"
        )
        if not Path(model).is_file():
            model_key += ".model_name"
        overrides.append(f"{model_key}={model}")
    return overrides


def _package_version(name: str) -> str | None:
    with contextlib.suppress(importlib.metadata.PackageNotFoundError):
        return importlib.metadata.version(name)
    return None


def hardware_report() -> dict[str, Any]:
    """
    Describe the CUDA and software environment used for the decision.
    """
    devices = []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": properties.name,
                "total_memory_bytes": properties.total_memory,
                "compute_capability": f"{properties.major}.{properties.minor}",
            }
        )
    return {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "visible_gpu_count": len(devices),
        "gpus": devices,
        "fairchem_core": _package_version("fairchem-core"),
        "fairchem_lammps": _package_version("fairchem-lammps"),
        "lammps": _package_version("lammps"),
    }


def generate_fcc_system(num_atoms: int, seed: int = 41) -> Atoms:
    """
    Generate a deterministic, orthogonal periodic carbon benchmark system.
    """
    if num_atoms < 1:
        raise ValueError("The generated atom count must be positive.")
    unit_cell = bulk("C", "fcc", a=3.8, cubic=True)
    repeats = math.ceil((num_atoms / len(unit_cell)) ** (1 / 3))
    atoms = unit_cell.repeat((repeats, repeats, repeats))
    rng = np.random.default_rng(seed)
    indices = np.sort(rng.choice(len(atoms), num_atoms, replace=False))
    return atoms[indices]


def load_system(args: SimpleNamespace) -> tuple[Atoms, dict[str, Any]]:
    """
    Load the user's structure or make the explicitly documented fallback.
    """
    if args.structure is not None:
        atoms = read(args.structure, index=0)
        source = str(args.structure.resolve())
        digest = hashlib.sha256(args.structure.read_bytes()).hexdigest()
    else:
        if args.generated_atoms < 1:
            raise ValueError("--generated-atoms must be positive.")
        atoms = generate_fcc_system(args.generated_atoms)
        source = f"generated_fcc_{args.generated_atoms}"
        digest = hashlib.sha256(
            np.asarray(atoms.positions, dtype=np.float64).tobytes()
        ).hexdigest()
    if not np.asarray(atoms.pbc, dtype=bool).all():
        raise ValueError("Pre-flight currently requires a fully periodic structure.")
    spin = args.spin if args.spin is not None else (1 if args.task == "omol" else 0)
    atoms.info.update(charge=args.charge, spin=spin)
    return atoms, {
        "source": source,
        "sha256": digest,
        "atoms": len(atoms),
        "symbols": sorted(set(atoms.get_chemical_symbols())),
        "task": args.task,
        "charge": args.charge,
        "spin": spin,
    }


def _write_lammps_input(directory: Path, atoms: Atoms, timestep_fs: float) -> Path:
    data_path = directory / "system.data"
    input_path = directory / "preflight.in"
    write(
        data_path,
        atoms,
        format="lammps-data",
        atom_style="atomic",
        masses=True,
    )
    input_path.write_text(
        "\n".join(
            [
                "units metal",
                "atom_style atomic",
                "boundary p p p",
                f"read_data {data_path}",
                f"timestep {timestep_fs / 1000.0}",
                "velocity all create 300.0 12345 mom yes rot no dist gaussian",
                "fix integrator all nve",
                "thermo 1000000000",
                "thermo_style custom step temp pe ke etotal press",
                "run 0",
                "",
            ]
        )
    )
    return input_path


def _cleanup_distributed_state() -> None:
    with contextlib.suppress(Exception):
        distutils.cleanup_gp_ray()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _bridge_checks(
    lmp: Any,
    recording: RecordingPredictor,
    expected_atomic_numbers: np.ndarray,
) -> dict[str, Any]:
    if recording.last_error is not None:
        raise recording.last_error
    if recording.last_results is None or recording.last_input is None:
        raise RuntimeError("LAMMPS did not invoke the FAIR-Chem callback.")
    results = recording.last_results
    energy = results["energy"].numpy()
    forces = results["forces"].numpy()
    nlocal = int(lmp.get_natoms())
    lammps_forces = lmp.numpy.extract_atom("f")[:nlocal].copy()
    lammps_energy = float(lmp.get_thermo("pe"))
    received_atomic_numbers = recording.last_input.atomic_numbers.detach().cpu().numpy()
    atom_ids = lmp.numpy.extract_atom("id")[:nlocal].copy()
    atom_mapping_ok = _atom_mapping_matches(
        received_atomic_numbers, expected_atomic_numbers, atom_ids
    )
    energy_error = abs(lammps_energy - float(np.asarray(energy).reshape(-1)[0]))
    force_error = float(np.max(np.abs(lammps_forces - forces)))
    finite = bool(
        np.isfinite(energy).all()
        and np.isfinite(forces).all()
        and np.isfinite(lammps_forces).all()
        and math.isfinite(lammps_energy)
    )
    return {
        "atom_mapping_passed": atom_mapping_ok,
        "finite_outputs_passed": finite,
        "bridge_energy_error_eV": energy_error,
        "bridge_force_max_error_eV_per_A": force_error,
        "bridge_passed": bool(
            atom_mapping_ok
            and finite
            and energy_error <= BRIDGE_ATOL
            and force_error <= BRIDGE_ATOL
        ),
        "_energy": energy,
        "_forces": forces,
    }


def _atom_mapping_matches(
    received_atomic_numbers: np.ndarray,
    expected_atomic_numbers: np.ndarray,
    lammps_atom_ids: np.ndarray,
) -> bool:
    """Compare elements in LAMMPS local-storage order using stable atom IDs."""
    received = np.asarray(received_atomic_numbers).reshape(-1)
    expected = np.asarray(expected_atomic_numbers).reshape(-1)
    atom_ids = np.asarray(lammps_atom_ids, dtype=np.int64).reshape(-1)
    if (
        len(received) != len(expected)
        or len(atom_ids) != len(expected)
        or np.any(atom_ids < 1)
        or np.any(atom_ids > len(expected))
        or len(np.unique(atom_ids)) != len(expected)
    ):
        return False
    return bool(np.array_equal(received, expected[atom_ids - 1]))


def _direct_reference_checks(
    predictor: MLIPPredictUnitProtocol,
    atoms: Atoms,
    task: str,
    charge: int,
    spin: int,
    callback_energy: np.ndarray,
    callback_forces: np.ndarray,
    acceptance: dict[str, float],
) -> dict[str, Any]:
    """
    Compare the callback result with direct UMA inference on the ASE structure.
    """
    direct_atoms = atoms.copy()
    direct_atoms.info.update(charge=charge, spin=spin)
    direct_data = AtomicData.from_ase(
        direct_atoms,
        task_name=task,
        r_data_keys=["charge", "spin"],
    )
    direct = predictor.predict(direct_data)
    direct_energy = direct["energy"].detach().cpu().numpy()
    direct_forces = direct["forces"].detach().cpu().numpy()
    prism = Prism(np.asarray(atoms.cell), pbc=np.asarray(atoms.pbc))
    direct_forces_lammps = prism.vector_to_lammps(direct_forces)
    energy_error = float(np.max(np.abs(direct_energy - callback_energy)))
    energy_error_per_atom = 1000.0 * energy_error / len(atoms)
    force_delta = direct_forces_lammps - callback_forces
    force_mae = float(np.mean(np.abs(force_delta)))
    force_error = float(np.max(np.abs(force_delta)))
    passed = bool(
        energy_error_per_atom <= acceptance["max_energy_error_meV_per_atom"]
        and force_mae <= acceptance["max_force_mae_eV_per_A"]
        and force_error <= acceptance["max_force_error_eV_per_A"]
    )
    return {
        "direct_energy_error_eV": energy_error,
        "direct_energy_error_meV_per_atom": energy_error_per_atom,
        "direct_force_mae_eV_per_A": force_mae,
        "direct_force_max_error_eV_per_A": force_error,
        "direct_reference_acceptance": acceptance,
        "direct_reference_passed": passed,
    }


def execute_case(
    *,
    atoms: Atoms,
    task: str,
    charge: int,
    spin: int,
    model_path: str,
    case: BenchmarkCase,
    timestep_fs: float,
    dynamics_steps: int,
    device: str = "cuda",
    warmup_steps: int = 0,
    timing_blocks: int = 0,
    block_steps: int = 0,
    acceptance: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Execute one isolated LAMMPS configuration and collect validation/timing data.
    """
    # LAMMPS is an optional dependency of fairchem-core and is required only
    # when a check is actually executed.
    from fairchem.lammps.lammps_fc import (
        run_lammps_with_fairchem,
    )

    profile = PROFILE_BY_NAME[case.profile]
    if acceptance is None:
        acceptance = numerical_acceptance()
    result: dict[str, Any] = asdict(case)
    result["status"] = "failed"
    lmp = None
    predictor = None
    try:
        settings = profile.settings(case.graph_version)
        start = time.perf_counter()
        predictor = load_predict_unit(
            path=model_path,
            device=device,
            inference_settings=settings,
            workers=case.workers,
            seed=41,
        )
        result["initialization_seconds"] = time.perf_counter() - start
        recording = RecordingPredictor(predictor)
        with tempfile.TemporaryDirectory(prefix="fairchem-lammps-preflight-") as tmp:
            input_path = _write_lammps_input(Path(tmp), atoms, timestep_fs)
            start = time.perf_counter()
            lmp = run_lammps_with_fairchem(
                recording,
                str(input_path),
                task,
                charge=charge,
                spin=spin,
                cmdargs=["-nocite", "-log", "none", "-screen", "none"],
            )
            result["cold_start_seconds"] = time.perf_counter() - start
            result.update(_bridge_checks(lmp, recording, atoms.get_atomic_numbers()))
            result.update(
                _direct_reference_checks(
                    predictor,
                    atoms,
                    task,
                    charge,
                    spin,
                    result["_energy"],
                    result["_forces"],
                    acceptance,
                )
            )
            result["resolved_execution_mode"] = recording.resolved_execution_mode
            start = time.perf_counter()
            if warmup_steps:
                lmp.command(f"run {warmup_steps} pre no post no")
            result["warmup_steps"] = warmup_steps
            result["warmup_seconds"] = time.perf_counter() - start
            samples = []
            for _ in range(timing_blocks):
                start = time.perf_counter()
                lmp.command(f"run {block_steps} pre no post no")
                samples.append((time.perf_counter() - start) * 1000 / block_steps)
            if dynamics_steps:
                lmp.command(f"run {dynamics_steps} pre no post no")
            finite_thermo = all(
                math.isfinite(float(lmp.get_thermo(name)))
                for name in ("temp", "pe", "ke", "press")
            )
            result["dynamics_steps"] = dynamics_steps
            result["finite_dynamics_passed"] = finite_thermo
            if samples:
                result["samples_ms_per_step"] = samples
                result["median_ms_per_step"] = statistics.median(samples)
                result["timing_cv"] = (
                    statistics.stdev(samples) / statistics.mean(samples)
                    if len(samples) > 1
                    else 0.0
                )
            result["status"] = (
                "passed"
                if result["bridge_passed"]
                and result["direct_reference_passed"]
                and finite_thermo
                else "failed"
            )
    except Exception as error:
        result["error_type"] = type(error).__name__
        result["error"] = str(error)
        result["out_of_memory"] = (
            isinstance(error, torch.OutOfMemoryError)
            or "out of memory" in str(error).lower()
        )
    finally:
        if lmp is not None:
            with contextlib.suppress(Exception):
                lmp.close()
        del predictor
        _cleanup_distributed_state()
    return result


def resolve_model_path(model: str) -> str:
    """
    Resolve a local checkpoint or a registered FAIR-Chem model name once.
    """
    path = Path(model)
    if path.is_file():
        return str(path.resolve())
    return pretrained_checkpoint_path_from_name(model)


def write_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")


def run_check(args: SimpleNamespace) -> int:
    if args.steps < 0 or args.timestep_fs <= 0:
        raise ValueError("Steps must be non-negative and the timestep positive.")
    hardware = hardware_report()
    if hardware["visible_gpu_count"] < args.workers:
        raise RuntimeError(
            f"Requested {args.workers} workers but found "
            f"{hardware['visible_gpu_count']} visible CUDA GPUs."
        )
    atoms, input_report = load_system(args)
    acceptance = numerical_acceptance(
        max_energy_error_meV_per_atom=args.max_energy_error_meV_per_atom,
        max_force_mae_eV_per_A=args.max_force_mae_eV_per_A,
        max_force_error_eV_per_A=args.max_force_error_eV_per_A,
    )
    model_path = resolve_model_path(args.model)
    case = BenchmarkCase(args.profile, args.graph_version, args.workers)
    result = execute_case(
        atoms=atoms,
        task=input_report["task"],
        charge=input_report["charge"],
        spin=input_report["spin"],
        model_path=model_path,
        case=case,
        timestep_fs=args.timestep_fs,
        dynamics_steps=args.steps,
        acceptance=acceptance,
    )
    for private_key in ("_energy", "_forces"):
        result.pop(private_key, None)
    report = {
        "command": "check",
        "hardware": hardware,
        "input": input_report,
        "model": args.model,
        "case": result,
        "passed": result["status"] == "passed",
    }
    write_report(args.output, report)
    print(
        f"LAMMPS pre-flight: {'PASS' if report['passed'] else 'FAIL'}; "
        f"report={args.output}"
    )
    return 0 if report["passed"] else 1


def run_benchmark(args: SimpleNamespace) -> int:
    if (
        args.expected_steps < 1
        or args.warmup_steps < 1
        or args.timing_blocks < 2
        or args.block_steps < 1
        or args.timestep_fs <= 0
    ):
        raise ValueError(
            "Benchmark steps and timestep must be positive, with at least two "
            "timing blocks."
        )
    hardware = hardware_report()
    gpu_count = hardware["visible_gpu_count"]
    worker_counts = parse_worker_counts(args.worker_counts)
    cases = build_case_matrix(gpu_count, worker_counts)
    atoms, input_report = load_system(args)
    acceptance = numerical_acceptance(
        max_energy_error_meV_per_atom=args.max_energy_error_meV_per_atom,
        max_force_mae_eV_per_A=args.max_force_mae_eV_per_A,
        max_force_error_eV_per_A=args.max_force_error_eV_per_A,
    )
    model_path = resolve_model_path(args.model)
    results = []
    for case in cases:
        print(
            f"Benchmarking {case.profile}, graph v{case.graph_version}, "
            f"workers={case.workers}"
        )
        result = execute_case(
            atoms=atoms,
            task=input_report["task"],
            charge=input_report["charge"],
            spin=input_report["spin"],
            model_path=model_path,
            case=case,
            timestep_fs=args.timestep_fs,
            dynamics_steps=0,
            warmup_steps=args.warmup_steps,
            timing_blocks=args.timing_blocks,
            block_steps=args.block_steps,
            acceptance=acceptance,
        )
        result["model"] = args.model
        results.append(result)
    reference_result = next(
        (
            result
            for result in results
            if result.get("status") == "passed"
            and result["profile"] == "fp32_eager"
            and result["graph_version"] == 2
            and result["workers"] == 1
        ),
        None,
    )
    if reference_result is None:
        reference_result = next(
            (
                result
                for result in results
                if result.get("status") == "passed"
                and not PROFILE_BY_NAME[result["profile"]].tf32
            ),
            None,
        )
    reference = (
        {
            "energy": np.asarray(reference_result["_energy"]),
            "forces": np.asarray(reference_result["_forces"]),
        }
        if reference_result is not None
        else None
    )
    for result in results:
        if result.get("status") == "passed" and reference is not None:
            add_numerical_comparison(
                result,
                reference,
                num_atoms=input_report["atoms"],
                acceptance=acceptance,
            )
        else:
            result.pop("_energy", None)
            result.pop("_forces", None)
            result["numerical_passed"] = False
    add_break_even_steps(results)
    recommendation = recommend_case(results, args.expected_steps)
    report = {
        "command": "benchmark",
        "hardware": hardware,
        "input": input_report,
        "model": args.model,
        "expected_steps": args.expected_steps,
        "selection_policy": {
            "minimum_speedup_for_compile_or_more_workers": (MIN_RECOMMENDATION_SPEEDUP),
            "maximum_timing_coefficient_of_variation": MAX_TIMING_CV,
            "numerical_acceptance": acceptance,
        },
        "cases": results,
        "recommendation": recommendation,
    }
    write_report(args.output, report)
    if recommendation is None:
        print(f"No trustworthy recommendation; report={args.output}")
        return 1
    print("Recommended LAMMPS overrides:")
    print(shlex.join(recommendation["hydra_overrides"]))
    print(f"Full evidence: {args.output}")
    return 0


def parse_worker_counts(value: str | list[int] | None) -> list[int] | None:
    if value is None or value == "all":
        return None
    if isinstance(value, list):
        return [int(item) for item in value]
    try:
        return [int(item) for item in value.split(",")]
    except ValueError as error:
        raise ValueError(
            "Worker counts must be 'all' or comma-separated integers."
        ) from error


def _namespace_from_config(cfg: DictConfig) -> SimpleNamespace:
    values = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(values, dict)
    values["structure"] = (
        Path(values["structure"]) if values["structure"] is not None else None
    )
    values["output"] = Path(
        values["output"]
        or (
            "lammps-preflight-check.json"
            if values["mode"] == "check"
            else "lammps-preflight-benchmark.json"
        )
    )
    return SimpleNamespace(**values)


@hydra.main(version_base=None, config_path=".", config_name="preflight_config")
def main(cfg: DictConfig) -> None:
    """
    Run a Hydra-configured pre-flight check or GPU benchmark.
    """
    args = _namespace_from_config(cfg)
    if args.mode == "check":
        if args.profile not in PROFILE_BY_NAME or args.graph_version not in (2, 3):
            raise ValueError("Invalid profile or graph generator version.")
        status = run_check(args)
    elif args.mode == "benchmark":
        status = run_benchmark(args)
    else:
        raise ValueError("mode must be 'check' or 'benchmark'.")
    if status:
        raise SystemExit(status)


if __name__ == "__main__":
    main()
