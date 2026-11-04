"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
HELPER = ROOT / ".github/scripts/nitro_ci.py"
SPEC = importlib.util.spec_from_file_location("nitro_ci", HELPER)
nitro_ci = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(nitro_ci)

# Independent report fixture: the coverage gate must not pass if the module
# under test accidentally removes a required prediction or fallback group.
PREDICTIONS = {
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
}
FALLBACKS = {
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
}


def _report(tmp_path: Path, identity: str = "classname") -> tuple[Path, ET.Element]:
    """
    Build a complete report using pytest's two supported JUnit identity formats.
    """
    root = ET.Element("testsuites")
    suite = ET.SubElement(root, "testsuite", tests="999")
    for module, functions in (
        ("test_uma_nitro_predict", PREDICTIONS),
        ("test_uma_nitro_fallbacks", FALLBACKS),
    ):
        for function, parameters in functions.items():
            for parameter in parameters:
                suffix = f"-{parameter}" if parameter else ""
                name = (
                    function
                    if parameter is None
                    else f"{function}[uma-s-1p2p1{suffix}]"
                )
                attributes = {"name": name}
                if identity == "classname":
                    attributes["classname"] = (
                        f"tests.core.models.uma.uma_nitro.{module}"
                    )
                else:
                    attributes["file"] = f"tests/core/models/uma/uma_nitro/{module}.py"
                ET.SubElement(suite, "testcase", **attributes)
    path = tmp_path / "junit.xml"
    ET.ElementTree(root).write(path)
    return path, root


@pytest.mark.parametrize("identity", ["classname", "file"])
def test_complete_prediction_and_runtime_report_passes(tmp_path, identity):
    report, _ = _report(tmp_path, identity)
    assert nitro_ci.verify_report(report) == {
        "prediction_cases": 18,
        "required_passed": 30,
        "other_skipped_cases": [],
    }


@pytest.mark.parametrize("status", ["skipped", "failure", "error"])
@pytest.mark.parametrize("group", ["test_calculator", "test_other_errors_propagate"])
def test_required_case_must_pass(tmp_path, status, group):
    report, root = _report(tmp_path)
    case = next(
        case for case in root.iter("testcase") if case.get("name").startswith(group)
    )
    ET.SubElement(case, status)
    ET.ElementTree(root).write(report)
    with pytest.raises(ValueError, match=status):
        nitro_ci.verify_report(report)


def test_unrelated_cases_cannot_replace_agreement_sentinel(tmp_path):
    report, root = _report(tmp_path)
    for case in root.iter("testcase"):
        if case.get("name").startswith("test_matches_general["):
            case.set("name", "test_unrelated[" + case.get("name") + "]")
    ET.ElementTree(root).write(report)
    with pytest.raises(
        ValueError, match="test_matches_general.*expected passed case is missing"
    ):
        nitro_ci.verify_report(report)


@pytest.mark.parametrize(
    "group",
    ["test_matches_general", "test_runtime_contract_break_falls_back_to_multigpu"],
)
def test_different_parameter_cannot_replace_required_scenario(tmp_path, group):
    report, root = _report(tmp_path)
    case = next(
        case
        for case in root.iter("testcase")
        if case.get("name").startswith(group + "[")
    )
    case.set("name", f"{group}[uma-s-1p2p1-unrelated-scenario]")
    ET.ElementTree(root).write(report)
    with pytest.raises(ValueError, match="expected passed case is missing"):
        nitro_ci.verify_report(report)


def test_different_checkpoint_cannot_replace_pinned_predictions(tmp_path):
    report, root = _report(tmp_path)
    for case in root.iter("testcase"):
        case.set("name", case.get("name").replace("uma-s-1p2p1", "uma-s-1p2"))
    ET.ElementTree(root).write(report)
    with pytest.raises(
        ValueError, match="uma-s-1p2p1.*expected passed case is missing"
    ):
        nitro_ci.verify_report(report)


def test_gate_requires_all_declared_scenarios():
    """
    Keep the required manifest aligned with the actual test parameter IDs.

    Inspect declarations without importing CUDA or downloading checkpoints.
    """
    for module, functions in (
        ("test_uma_nitro_predict", PREDICTIONS),
        ("test_uma_nitro_fallbacks", FALLBACKS),
    ):
        source = ROOT / f"tests/core/models/uma/uma_nitro/{module}.py"
        definitions = {
            node.name: node
            for node in ast.parse(source.read_text()).body
            if isinstance(node, ast.FunctionDef)
        }
        for name, required in functions.items():
            definition = definitions[name]
            if len(required) == 1:
                continue
            parametrization = next(
                decorator
                for decorator in definition.decorator_list
                if isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and decorator.func.attr == "parametrize"
            )
            ids = next(
                keyword.value
                for keyword in parametrization.keywords
                if keyword.arg == "ids"
            )
            if isinstance(ids, ast.Lambda):
                # Settings IDs use the keys in each change dictionary.
                declared = tuple(
                    ",".join(change)
                    for change in ast.literal_eval(parametrization.args[1])
                )
            else:
                declared = tuple(ast.literal_eval(ids))
            assert declared == required, name
            assert nitro_ci.REQUIRED_CASES[module][name] == required


def test_duplicate_cannot_replace_missing_parametrization(tmp_path):
    report, root = _report(tmp_path)
    cases = list(root.iter("testcase"))
    cases[1].set("name", cases[0].get("name"))
    ET.ElementTree(root).write(report)
    with pytest.raises(ValueError, match="Duplicate Nitro case"):
        nitro_ci.verify_report(report)


@pytest.mark.parametrize("contents", [None, "<testsuites>", "<testsuites />"])
def test_missing_invalid_or_empty_report_fails(tmp_path, contents):
    report = tmp_path / "junit.xml"
    if contents is not None:
        report.write_text(contents)
    with pytest.raises(ValueError):
        nitro_ci.verify_report(report)


def test_nonrequired_skips_are_reported(tmp_path):
    report, root = _report(tmp_path)
    case = ET.SubElement(
        root.find("testsuite"),
        "testcase",
        classname="tests.core.models.uma.uma_nitro.test_uma_nitro_kernels",
        name="test_optional_case",
    )
    ET.SubElement(case, "skipped", message="not applicable on this GPU")
    ET.ElementTree(root).write(report)
    summary = nitro_ci.verify_report(report)
    assert summary["required_passed"] == 30
    assert summary["other_skipped_cases"] == [
        "tests.core.models.uma.uma_nitro.test_uma_nitro_kernels::test_optional_case"
    ]


def test_cli_retains_failed_coverage_summary(tmp_path):
    summary = tmp_path / "results/coverage.json"
    result = subprocess.run(
        [
            sys.executable,
            str(HELPER),
            "--verify",
            str(tmp_path / "missing.xml"),
            "--summary",
            str(summary),
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 1
    assert json.loads(summary.read_text())["status"] == "failed"
    assert "Missing or invalid" in result.stdout


@pytest.mark.parametrize("family", ["xunit2", "legacy"])
def test_real_pytest_reports_retain_module_identity(tmp_path, monkeypatch, family):
    """
    Recognize actual pytest XML when the config is outside the test root.
    """
    monkeypatch.setenv("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    probe = tmp_path / "test_uma_nitro_predict.py"
    probe.write_text("def test_calculator():\n    pass\n")
    report = tmp_path / "junit.xml"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            str(probe),
            "-c",
            str(ROOT / "packages/fairchem-core/pyproject.toml"),
            "--rootdir",
            str(tmp_path),
            "--noconftest",
            "-o",
            "addopts=",
            "-o",
            f"junit_family={family}",
            f"--junitxml={report}",
        ],
        capture_output=True,
        text=True,
        timeout=10,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    case = ET.parse(report).find(".//testcase")
    assert nitro_ci._module(case) == "test_uma_nitro_predict"


@pytest.mark.parametrize(
    ("available", "capability", "name", "policy", "error"),
    [
        (False, (0, 0), "", "", "available CUDA"),
        (True, (7, 5), "T4", "portable_pytorch_ieee (CUDA capability 7.5)", "8.x-12.x"),
        (True, (13, 0), "future GPU", "unknown (CUDA capability 13.0)", "8.x-12.x"),
        (
            True,
            (9, 0),
            "H100",
            "portable_pytorch_ieee (CUDA capability 9.0)",
            "native_sm90_plus",
        ),
        (
            True,
            (8, 0),
            "H100",
            "portable_pytorch_ieee (CUDA capability 8.0)",
            "capability 9.0",
        ),
        (
            True,
            (9, 0),
            "H100",
            "native_sm90_plus (CUDA capability 10.0)",
            "different CUDA",
        ),
    ],
)
def test_preflight_rejects_unusable_gpu(available, capability, name, policy, error):
    with pytest.raises(ValueError, match=error):
        nitro_ci.validate_device(available, capability, name, policy)


@pytest.mark.parametrize(
    ("capability", "name", "policy"),
    [
        ((8, 0), "A100", "portable_pytorch_ieee (CUDA capability 8.0)"),
        ((8, 9), "L4", "portable_pytorch_ieee (CUDA capability 8.9)"),
        ((9, 0), "H100", "native_sm90_plus (CUDA capability 9.0)"),
        ((12, 0), "Blackwell", "blackwell_cute_cublas (CUDA capability 12.0)"),
    ],
)
def test_preflight_accepts_supported_gpu(capability, name, policy):
    nitro_ci.validate_device(True, capability, name, policy)


def test_preflight_timeout_retains_diagnostics(tmp_path, monkeypatch):
    def timeout(command, **kwargs):
        kwargs["stdout"].write("CUDA import stalled\n")
        raise subprocess.TimeoutExpired(command, kwargs["timeout"])

    monkeypatch.setattr(nitro_ci.subprocess, "run", timeout)
    log = tmp_path / "results/preflight.log"
    assert nitro_ci.preflight(log) == 1
    assert "CUDA import stalled" in log.read_text()
    assert "exceeded 60 seconds" in log.read_text()


def test_preflight_preserves_child_failure(tmp_path, monkeypatch):
    monkeypatch.setattr(
        nitro_ci.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(returncode=2),
    )
    assert nitro_ci.preflight(tmp_path / "preflight.log") == 2
