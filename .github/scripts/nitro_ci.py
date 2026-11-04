"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

PREDICTION_MODULE = "test_uma_nitro_predict"
FALLBACK_MODULE = "test_uma_nitro_fallbacks"
# Require the actual checkpoint and parameter IDs, rather than only case counts.
# A different passing parametrization cannot replace a required runtime scenario.
REQUIRED_CASES = {
    PREDICTION_MODULE: {
        "test_matches_general": (
            "Cu-omat",
            "NaCl-omc",
            "H2O-omol",
            "H2O+-omol",
            "H-atom",
            "no-edges",
        ),
        "test_matches_general_without_element_references": ("",),
        "test_runs_with_settings_it_ignores_or_supports": (
            "compile",
            "hessian_vmap",
            "edge_chunk_size",
            "use_quaternion_wigner",
            "torch_num_threads",
            "auto_add_default_untrained_tasks",
            "auto_add_default_untrained_tasks,predict_untrained_stress",
            "predict_untrained_forces",
        ),
        "test_default_settings_run_nitro": ("",),
        "test_calculator": ("",),
        "test_recovers_from_stale_edge_capacity": ("",),
    },
    FALLBACK_MODULE: {
        # This constructor check uses a random checkpoint, without parametrization.
        "test_falls_back_on_a_gpu_other_than_the_kernels_were_chosen_for": (None,),
        "test_runtime_contract_break_falls_back_to_multigpu": (
            "composition",
            "charge",
            "spin",
            "task",
            "homogeneous-batch",
        ),
        "test_multigpu_then_falls_back_to_general": ("",),
        "test_homogeneous_first_batch_runs_multigpu": ("",),
        "test_heterogeneous_batch_falls_back_to_multigpu_then_general": ("",),
        "test_keeps_running_for_the_same_reduced_composition": ("",),
        "test_falls_back_on_unsupported_error_during_inference": ("",),
        "test_other_errors_propagate": ("",),
    },
}


def validate_device(
    cuda_available: bool,
    capability: tuple[int, int],
    device_name: str,
    kernel_summary: str,
) -> None:
    """
    Reject unsupported devices and incorrect native Hopper kernel selection.
    """
    if not cuda_available:
        raise ValueError("Nitro CI requires an available CUDA device")
    major, minor = capability
    if not 8 <= major <= 12:
        raise ValueError(f"Nitro CI requires capability 8.x-12.x, got {major}.{minor}")
    if f"CUDA capability {major}.{minor}" not in kernel_summary:
        raise ValueError("Nitro selected kernels for a different CUDA capability")
    if "H100" in device_name.upper() and capability != (9, 0):
        raise ValueError("The H100 lane requires CUDA capability 9.0")
    if major == 9 and not kernel_summary.startswith("native_sm90_plus ("):
        raise ValueError("The Hopper lane requires the native_sm90_plus kernel policy")


def _probe() -> None:
    """
    Import GPU libraries only in the bounded preflight child process.
    """
    import torch

    if not torch.cuda.is_available():
        raise ValueError("Nitro CI requires an available CUDA device")
    if torch.cuda.device_count() != 1:
        raise ValueError("Nitro CI requires exactly one visible CUDA device")
    capability = torch.cuda.get_device_capability(0)
    if not 8 <= capability[0] <= 12:
        raise ValueError(f"Unsupported Nitro CUDA capability: {capability}")

    from fairchem.core.models.uma.uma_nitro import _inference

    device_name = torch.cuda.get_device_name(0)
    summary = _inference.kernel_backend_summary()
    validate_device(True, capability, device_name, summary)
    packages = {}
    for name in (
        "torch",
        "triton",
        "nvalchemi-toolkit-ops",
        "nvidia-cutlass-dsl",
        "cuda-python",
        "pytest",
        "ray",
    ):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = "not installed"
    print(
        json.dumps(
            {
                "device": device_name,
                "capability": capability,
                "torch_cuda": torch.version.cuda,
                "kernel_policy": summary,
                "packages": packages,
            },
            indent=2,
        ),
        flush=True,
    )


def preflight(log: Path) -> int:
    """
    Retain the probe output even if CUDA initialization or library imports hang.
    """
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as stream:
        try:
            result = subprocess.run(
                [sys.executable, __file__, "--probe"],
                stdout=stream,
                stderr=subprocess.STDOUT,
                timeout=60,
                check=False,
            )
            returncode = result.returncode
        except subprocess.TimeoutExpired:
            stream.write("Nitro preflight exceeded 60 seconds\n")
            returncode = 1
    print(log.read_text(), end="", flush=True)
    return returncode if returncode >= 0 else 128 - returncode


def _module(case: ET.Element) -> str:
    """
    Recognize pytest's modern classname and legacy file-based JUnit reports.
    """
    file_name = case.get("file", "").replace("\\", "/").rsplit("/", 1)[-1]
    file_module = file_name.removesuffix(".py")
    classname = case.get("classname", "").split(".")
    for module in REQUIRED_CASES:
        if file_module == module or module in classname:
            return module
    return ""


def verify_report(report: Path) -> dict:
    """
    Require all prediction and supported-GPU fallback sentinels to have passed.
    """
    try:
        root = ET.parse(report).getroot()
    except (OSError, ET.ParseError) as exc:
        raise ValueError(f"Missing or invalid Nitro JUnit report: {report}") from exc
    passed = set()
    seen = set()
    problems = []
    skipped = []
    prediction_count = 0
    for case in root.iter("testcase"):
        module = _module(case)
        name = case.get("name", "")
        function = name.split("[", 1)[0]
        status = next(
            (
                tag
                for tag in ("error", "failure", "skipped")
                if case.find(tag) is not None
            ),
            "passed",
        )
        required = module == PREDICTION_MODULE or function in REQUIRED_CASES.get(
            module, {}
        )
        if status in ("error", "failure") or (required and status == "skipped"):
            problems.append(f"{module or case.get('classname', '')}::{name}: {status}")
        if status == "skipped":
            skipped.append(f"{module or case.get('classname', '')}::{name}")
        if required:
            identity = (module, name)
            if identity in seen:
                problems.append(f"Duplicate Nitro case: {module}::{name}")
            seen.add(identity)
        if module == PREDICTION_MODULE:
            prediction_count += 1
        if status == "passed" and required:
            passed.add((module, name))
    for module, functions in REQUIRED_CASES.items():
        for function, parameters in functions.items():
            for parameter in parameters:
                if parameter is None:
                    name = function
                else:
                    suffix = f"-{parameter}" if parameter else ""
                    name = f"{function}[uma-s-1p2p1{suffix}]"
                if (module, name) not in passed:
                    problems.append(
                        f"{module}::{name}: expected passed case is missing"
                    )
    if prediction_count < 18:
        problems.append(
            f"Expected at least 18 prediction cases, got {prediction_count}"
        )
    if problems:
        raise ValueError("Nitro coverage gate failed:\n" + "\n".join(problems))
    return {
        "prediction_cases": prediction_count,
        "required_passed": len(passed),
        "other_skipped_cases": skipped,
    }


def main() -> int:
    """
    Run either the GPU preflight or the post-test coverage gate.
    """
    parser = argparse.ArgumentParser(description="Validate real UMA Nitro CI coverage")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--preflight", type=Path, metavar="LOG")
    mode.add_argument("--verify", type=Path, metavar="JUNIT")
    mode.add_argument("--probe", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    if args.preflight:
        return preflight(args.preflight)
    if args.probe:
        _probe()
        return 0
    try:
        summary = {"status": "passed", **verify_report(args.verify)}
        returncode = 0
    except ValueError as exc:
        summary = {"status": "failed", "error": str(exc)}
        returncode = 1
    output = json.dumps(summary, indent=2) + "\n"
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(output)
    print(output, end="", flush=True)
    return returncode


if __name__ == "__main__":
    raise SystemExit(main())
