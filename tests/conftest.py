"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

# conftest.py
from __future__ import annotations

import random
from contextlib import suppress

import numpy as np
import pytest
import ray
import torch

import fairchem.core.common.gp_utils as gp_utils
from fairchem.core.common import distutils


@pytest.fixture()
def command_line_inference_checkpoint(request):
    return request.config.getoption("--inference-checkpoint")


@pytest.fixture()
def command_line_inference_dataset(request):
    return request.config.getoption("--inference-dataset")


def pytest_addoption(parser):
    parser.addoption(
        "--skip-ocpapi-integration",
        action="store_true",
        default=False,
        help="skip ocpapi integration tests",
    )
    parser.addoption(
        "--inference-checkpoint",
        action="store",
        help="inference checkpoint to run check on",
    )
    parser.addoption(
        "--inference-dataset", action="store", help="inference dataset to run check on"
    )
    parser.addoption(
        "--sweep-model",
        action="store",
        default=None,
        help=(
            "Sweep mode: name (e.g. 'uma-s-1p3') or filesystem path of a "
            "pretrained checkpoint to run all @pretrained tests against. "
            "When set, tests without @pytest.mark.pretrained are deselected "
            "and @calibrated tests are skipped unless the sweep model "
            "matches their declared pretrained(...) models."
        ),
    )
    parser.addoption(
        "--exclude-models",
        action="store",
        default=None,
        help=(
            "Comma-separated model names to exclude. Tests whose "
            "@pretrained(...) models are entirely within this set are "
            "deselected. Tests with no declared models (all-models tests) "
            "are kept. Used by the base CI job to avoid re-running models "
            "that have their own sweep jobs."
        ),
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "ocpapi_integration: ocpapi integration test")
    config.addinivalue_line("markers", "gpu: mark test to run only on GPU workers")
    config.addinivalue_line(
        "markers",
        "compile_gpu: GPU test that uses torch.compile — run in a separate pytest "
        "session to avoid cross-test memory accumulation",
    )
    config.addinivalue_line(
        "markers",
        "serial: mark test to run serially on the CPU runner "
        "(not parallelized with xdist)",
    )
    config.addinivalue_line(
        "markers",
        "subprocess: mark test that spawns subprocesses "
        "(excluded from parallel xdist run)",
    )
    config.addinivalue_line(
        "markers",
        "pretrained: test exercises a pretrained checkpoint. Optional positional "
        "args list model names to parametrize against (e.g. "
        '@pretrained("uma-s-1p1", "uma-s-1p2")). No args means the test runs '
        "against all registered models or uses its own model selection.",
    )
    config.addinivalue_line(
        "markers",
        "calibrated: test asserts exact numerical values calibrated to a specific "
        "pretrained checkpoint. Skipped under --sweep-model when the sweep model "
        "does not match the test's declared pretrained(...) models.",
    )


def _pretrained_model_values(metafunc) -> list[str]:
    """
    Resolve pretrained model names for parametrization.

    When ``--sweep-model`` is set, returns ``[override]``.
    Otherwise reads model names from the closest ``@pretrained(...)`` marker.
    """
    override = metafunc.config.getoption("--sweep-model")
    if override is not None:
        return [override]

    marker = metafunc.definition.get_closest_marker("pretrained")
    if marker is None or not marker.args:
        raise RuntimeError(
            f"{metafunc.function.__name__} uses pretrained_checkpoint / "
            "pretrained_model_name but does not declare "
            "@pytest.mark.pretrained(...)."
        )
    return list(marker.args)


@pytest.fixture(scope="module")
def pretrained_checkpoint(request):
    """
    Name or path of the pretrained checkpoint under test.

    Parametrized by ``pytest_generate_tests`` from the test's
    ``@pytest.mark.pretrained(...)`` declaration, or by ``--sweep-model``
    in sweep mode.
    """
    if hasattr(request, "param"):
        return request.param

    override = request.config.getoption("--sweep-model")
    if override is not None:
        return override

    marker = request.node.get_closest_marker("pretrained")
    if marker is None or not marker.args:
        raise RuntimeError(
            f"{request.node.nodeid} uses pretrained_checkpoint but does not "
            "declare @pytest.mark.pretrained(...)."
        )
    return marker.args[0]


def pytest_generate_tests(metafunc):
    """
    Provide values for pretrained checkpoint parameters:
    - default: checkpoints declared by @pytest.mark.pretrained(...)
    - --sweep-model=...: just the override

    Tests opt in by taking ``pretrained_model_name`` or
    ``pretrained_checkpoint`` as a fixture argument.
    """
    if "pretrained_model_name" in metafunc.fixturenames:
        metafunc.parametrize(
            "pretrained_model_name",
            _pretrained_model_values(metafunc),
            scope="module",
        )
    if "pretrained_checkpoint" in metafunc.fixturenames:
        metafunc.parametrize(
            "pretrained_checkpoint",
            _pretrained_model_values(metafunc),
            indirect=True,
            scope="module",
        )


def pytest_runtest_setup(item):
    # Check if the test has the 'gpu' marker
    if "gpu" in item.keywords and not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping GPU test")
    if "compile_gpu" in item.keywords and not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping compile_gpu test")
    if "dgl" in item.keywords:
        # check dgl is installed
        fairchem_cpp_found = False
        with suppress(ModuleNotFoundError):
            import fairchem_cpp

            unused = (  # noqa: F841
                fairchem_cpp.__file__
            )  # prevent the linter from deleting the import
            fairchem_cpp_found = True
        if not fairchem_cpp_found:
            pytest.skip(
                "fairchem_cpp not found, skipping DGL tests! please install "
                "fairchem if you want to run these"
            )


def pytest_collection_modifyitems(config, items):
    _validate_sweep_model(config)

    if config.getoption("--skip-ocpapi-integration"):
        skip_ocpapi_integration = pytest.mark.skip(reason="skipping ocpapi integration")
        for item in items:
            if "ocpapi_integration_test" in item.keywords:
                item.add_marker(skip_ocpapi_integration)

    if config.getoption("--inference-checkpoint"):
        # Skip all tests not marked with 'inference_check'
        for item in items:
            if "inference_check" not in item.keywords:
                item.add_marker(pytest.mark.skip(reason="skip all but inference check"))
    else:
        # Skip all tests marked with 'inference_check' by default
        skip_inference_check = pytest.mark.skip(
            reason="skipping inference check by default"
        )
        for item in items:
            if "inference_check" in item.keywords:
                item.add_marker(skip_inference_check)

    # --exclude-models: deselect tests whose declared pretrained(...)
    # models are entirely within the excluded set. Tests with no declared
    # models (bare @pretrained or no marker) are kept.
    exclude_raw = config.getoption("--exclude-models")
    if exclude_raw is not None:
        excluded = {t.strip() for t in exclude_raw.split(",") if t.strip()}
        keep, deselect = [], []
        for item in items:
            marker = item.get_closest_marker("pretrained")
            declared = set(marker.args) if marker and marker.args else set()
            if declared and declared.issubset(excluded):
                deselect.append(item)
                continue
            keep.append(item)
        if deselect:
            config.hook.pytest_deselected(items=deselect)
            items[:] = keep

    # --sweep-model: select only @pretrained tests.
    # @calibrated tests are skipped unless the sweep model matches their
    # declared pretrained(...) models.
    sweep_override = config.getoption("--sweep-model")
    if sweep_override is not None:
        keep, deselect = [], []
        for item in items:
            if not item.get_closest_marker("pretrained"):
                deselect.append(item)
                continue
            if item.get_closest_marker("calibrated"):
                marker = item.get_closest_marker("pretrained")
                declared = set(marker.args) if marker and marker.args else set()
                if sweep_override not in declared:
                    item.add_marker(
                        pytest.mark.skip(
                            reason=(
                                f"calibrated: --sweep-model={sweep_override} "
                                f"not in declared models {declared}"
                            )
                        )
                    )
            keep.append(item)
        if deselect:
            config.hook.pytest_deselected(items=deselect)
            items[:] = keep


def seed_everywhere(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _memory_summary() -> str:
    parts = []
    if torch.cuda.is_available():
        free_gpu, total_gpu = torch.cuda.mem_get_info()
        parts.append(f"GPU free {free_gpu / 1024**3:.2f}/{total_gpu / 1024**3:.2f} GB")
    import psutil

    vm = psutil.virtual_memory()
    parts.append(f"CPU free {vm.available / 1024**3:.2f}/{vm.total / 1024**3:.2f} GB")
    return " | ".join(parts)


_test_counts: dict[str, int] = {"current": 0, "total": 0}


def pytest_collection_finish(session):
    _test_counts["total"] = len(session.items)


def pytest_runtest_logreport(report):
    if report.when != "teardown":
        return
    _test_counts["current"] += 1
    current = _test_counts["current"]
    total = _test_counts["total"]
    pct = int(100 * current / total) if total else 0
    summary = _memory_summary()
    if summary:
        print(
            f"\n[mem] [{current}/{total} {pct:3d}%] {report.nodeid}: {summary}",
            flush=True,
        )


@pytest.fixture()
def seed_fixture():
    seed_everywhere(42)  # You can set your desired seed value here


@pytest.fixture()
def compile_reset_state():
    import gc

    torch.compiler.reset()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    yield
    torch.compiler.reset()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(scope="session")
def water_xyz_file(tmp_path_factory):
    """
    Provide a reusable minimal water molecule XYZ file path.

    Returns the filesystem path to a temporary XYZ file containing a 3-atom
    water cluster suitable for quick inference / graph generation tests.
    """
    contents = (
        "3\n"
        "water\n"
        "O 0.000000 0.000000 0.000000\n"
        "H 0.758602 0.000000 0.504284\n"
        "H -0.758602 0.000000 0.504284\n"
    )
    d = tmp_path_factory.mktemp("xyz_inputs")
    fpath = d / "water.xyz"
    fpath.write_text(contents)
    return str(fpath)


def _validate_sweep_model(config) -> None:
    """
    Fail fast when ``--sweep-model`` names a model that cannot be loaded.

    Without this check, an invalid name surfaces deep inside the first
    test as a confusing ``KeyError`` from ``pretrained_mlip.get_predict_unit``.
    Validating at collection time produces a clear ``pytest.UsageError``
    listing the available models.
    """
    import os

    sweep = config.getoption("--sweep-model", default=None)
    if sweep is None:
        return
    if os.path.exists(sweep):
        return
    from fairchem.core import pretrained_mlip

    if sweep in pretrained_mlip.available_models:
        return
    raise pytest.UsageError(
        f"--sweep-model={sweep!r} is neither an existing file path nor a "
        f"registered model name. Registered models: "
        f"{sorted(pretrained_mlip.available_models)}."
    )


def get_predict_unit_for_test(name_or_path, **kwargs):
    """
    Resolve a model name or filesystem path to a ``MLIPPredictUnit``.

    When ``name_or_path`` is an existing file on disk (e.g. passed via
    ``--sweep-model /path/to/file.pt``), loads with
    ``load_predict_unit`` directly.  Otherwise delegates to
    ``pretrained_mlip.get_predict_unit`` which resolves registered names.
    """
    import os
    if os.path.exists(name_or_path):
        from fairchem.core.units.mlip_unit import load_predict_unit

        return load_predict_unit(name_or_path, **kwargs)
    from fairchem.core import pretrained_mlip

    return pretrained_mlip.get_predict_unit(name_or_path, **kwargs)


def models_to_test(config) -> list[str]:
    """
    Return the list of pretrained model names to iterate over in all-models tests.

    Respects the CI partition flags:

    * ``--sweep-model=X`` → ``[X]`` (one model, used by sweep jobs).
    * ``--exclude-models=A,B`` → all registered models minus A, B
      (used by base job to skip models that have their own sweep jobs).
    * Neither flag → every registered pretrained model.

    Import and call from any test module that needs to parametrize
    over the full model catalogue::

        from tests.conftest import models_to_test

        def pytest_generate_tests(metafunc):
            if "my_fixture" in metafunc.fixturenames:
                metafunc.parametrize(
                    "my_fixture",
                    models_to_test(metafunc.config),
                    indirect=True,
                )
    """
    from fairchem.core.calculate import pretrained_mlip

    override = config.getoption("--sweep-model")
    if override is not None:
        return [override]
    excluded = config.getoption("--exclude-models")
    excluded_set = set(excluded.split(",")) if excluded else set()
    return [m for m in pretrained_mlip.available_models if m not in excluded_set]


@pytest.fixture(autouse=True)
def setup_before_each_test():
    if ray.is_initialized():
        ray.shutdown()
    if gp_utils.initialized():
        gp_utils.cleanup_gp()
    distutils.cleanup()
    yield
    if ray.is_initialized():
        ray.shutdown()
    if gp_utils.initialized():
        gp_utils.cleanup_gp()
    distutils.cleanup()
