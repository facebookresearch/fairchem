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

    _validate_model_flags(config)


def sweep_model(config) -> str | None:
    """
    Return ``--sweep-model`` value, or ``None`` if unset/empty.

    Centralized so renames and validation live in one place. Empty
    strings are normalized to ``None`` so downstream callers can use
    a simple truthiness check.
    """
    raw = config.getoption("--sweep-model", default=None)
    if raw is None:
        return None
    raw = raw.strip()
    return raw or None


def excluded_models(config) -> set[str]:
    """
    Return the parsed ``--exclude-models`` set (whitespace-trimmed,
    empties dropped). Returns an empty set when the flag is unset.
    """
    raw = config.getoption("--exclude-models", default=None)
    if not raw:
        return set()
    return {t.strip() for t in raw.split(",") if t.strip()}


def uma_models() -> list[str]:
    """
    Return registered model names belonging to the UMA family.

    Single source of truth for the ``"uma" in name`` heuristic. If UMA
    naming conventions ever change, update this function instead of
    grep-and-replace across fixture files.
    """
    from fairchem.core import pretrained_mlip

    return [name for name in pretrained_mlip.available_models if "uma" in name]


def _pretrained_model_values(metafunc) -> list[str]:
    """
    Resolve pretrained model names for parametrization.

    When ``--sweep-model`` is set, returns ``[override]``.
    Otherwise reads model names from the closest ``@pretrained(...)`` marker.
    """
    override = sweep_model(metafunc.config)
    if override is not None:
        return [override]

    marker = metafunc.definition.get_closest_marker("pretrained")
    if marker is None:
        raise RuntimeError(
            f"{metafunc.function.__name__} uses pretrained_checkpoint / "
            "pretrained_model_name but no @pytest.mark.pretrained(...) marker "
            "is in scope. Decorate the test or run with --sweep-model=<model>."
        )
    if not marker.args:
        raise RuntimeError(
            f"{metafunc.function.__name__} uses pretrained_checkpoint / "
            "pretrained_model_name but its closest @pytest.mark.pretrained "
            "marker has no model arguments. Add models (e.g. "
            "@pytest.mark.pretrained('uma-s-1p1')) or run with "
            "--sweep-model=<model>."
        )
    return list(marker.args)


@pytest.fixture(scope="module")
def pretrained_checkpoint(request):
    """
    Name or path of the pretrained checkpoint under test.

    Always parametrized by ``pytest_generate_tests`` (via ``indirect=True``)
    from the test's ``@pytest.mark.pretrained(...)`` marker or by
    ``--sweep-model``. The value lives in ``request.param``.
    """
    return request.param


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
    excluded = excluded_models(config)
    if excluded:
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
    # declared pretrained(...) models. A bare @pretrained (no args) is
    # treated as "matches everything" for calibrated purposes.
    sweep_override = sweep_model(config)
    if sweep_override is not None:
        keep, deselect = [], []
        for item in items:
            if not item.get_closest_marker("pretrained"):
                deselect.append(item)
                continue
            if item.get_closest_marker("calibrated"):
                marker = item.get_closest_marker("pretrained")
                declared = set(marker.args) if marker and marker.args else set()
                # Bare marker (no declared models) means the test is not
                # locked to any specific checkpoint — let it run.
                if declared and sweep_override not in declared:
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


def _validate_model_flags(config) -> None:
    """
    Fail fast on unusable ``--sweep-model`` / ``--exclude-models`` values.

    Without these checks, an invalid model name surfaces deep inside the
    first test as a confusing ``KeyError`` from
    ``pretrained_mlip.get_predict_unit``, and an unknown ``--exclude-models``
    token silently no-ops (partition violation). Validating at
    configuration time produces a clear ``pytest.UsageError``.

    Rules:

    * ``--sweep-model`` must be a registered model OR an existing file.
    * Every ``--exclude-models`` token must be a registered model.
    * ``--sweep-model=<name>`` may not also appear in ``--exclude-models``
      (zero tests selected).
    * ``--sweep-model=<path>`` may not be combined with non-empty
      ``--exclude-models`` (ambiguous: we cannot tell whether the path's
      weights belong to an excluded model).
    """
    import os

    from fairchem.core import pretrained_mlip

    available = set(pretrained_mlip.available_models)

    excluded = excluded_models(config)
    unknown = excluded - available
    if unknown:
        raise pytest.UsageError(
            f"--exclude-models contains unknown model names: "
            f"{sorted(unknown)}. Registered models: {sorted(available)}."
        )

    sweep = sweep_model(config)
    if sweep is None:
        return

    is_registered = sweep in available
    is_real_file = os.path.isfile(sweep)
    if not (is_registered or is_real_file):
        raise pytest.UsageError(
            f"--sweep-model={sweep!r} is neither a registered model name "
            f"nor an existing file. Registered models: {sorted(available)}."
        )
    if sweep in excluded:
        raise pytest.UsageError(
            f"--sweep-model={sweep!r} is also listed in --exclude-models "
            f"({sorted(excluded)}); the combination would deselect every "
            f"test. Drop one of the flags."
        )
    if is_real_file and not is_registered and excluded:
        raise pytest.UsageError(
            f"--sweep-model={sweep!r} is a filesystem path; combining it "
            f"with --exclude-models={sorted(excluded)} is ambiguous because "
            f"we cannot tell whether the path's weights belong to an "
            f"excluded model. Drop one of the flags."
        )


def get_predict_unit_for_test(name_or_path, **kwargs):
    """
    Resolve a registered model name or filesystem path to a ``MLIPPredictUnit``.

    Tries the registry first to avoid CWD collisions where a file
    happens to share a registered model name. Falls back to
    ``load_predict_unit`` for filesystem paths. If the caller passes
    a kwarg that ``load_predict_unit`` does not accept (e.g.
    ``cache_dir``) on the path branch, the resulting ``TypeError``
    is the right signal — silent dropping would hide intent.
    """
    import os

    from fairchem.core import pretrained_mlip

    if name_or_path in pretrained_mlip.available_models:
        return pretrained_mlip.get_predict_unit(name_or_path, **kwargs)
    if os.path.isfile(name_or_path):
        from fairchem.core.units.mlip_unit import load_predict_unit

        return load_predict_unit(name_or_path, **kwargs)
    # Not registered and not a file — defer to get_predict_unit so the
    # caller sees the registry's own KeyError listing valid names.
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

    override = sweep_model(config)
    if override is not None:
        return [override]
    excluded = excluded_models(config)
    return [m for m in pretrained_mlip.available_models if m not in excluded]


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
