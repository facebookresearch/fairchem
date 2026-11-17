"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

# conftest.py
from __future__ import annotations

import random
from contextlib import suppress
from typing import TYPE_CHECKING

import numpy as np
import pytest
import ray
import torch

import fairchem.core.common.gp_utils as gp_utils
from fairchem.core.common import distutils

if TYPE_CHECKING:
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings


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
            "Sweep mode: select tests that exercise a specific checkpoint. "
            "Accepts a registered model name OR an absolute path to a .pt "
            "checkpoint. "
            "NAME (e.g. 'uma-s-1p2'): strict — tests are kept only when "
            "their @pytest.mark.pretrained(...) args include this name. "
            "PATH (e.g. '/abs/path/candidate.pt'): permissive UMA-family — "
            "any test whose @pretrained args include a UMA model is kept "
            "and runs against your checkpoint. Use this for validating an "
            "unreleased UMA candidate against the existing test suite. "
            "Calibrated tests (test_simple_md, test_single_atom_predict_*, "
            "test_formation_..._against_known_values, "
            "test_stress_old_vs_new_*) assert exact values pinned to a "
            "specific checkpoint; on a candidate they will fail with "
            "value deltas — that delta IS the regression signal. "
            "Tests without @pytest.mark.pretrained are always deselected; "
            "bare @pretrained (no args) is always kept."
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
    parser.addoption(
        "--sweep-refs-from",
        action="store",
        default=None,
        help=(
            "When --sweep-model is a filesystem path, borrow the "
            "atom_refs and form_elem_refs YAMLs from this registered model "
            "name (e.g. 'uma-s-1p2'). Without this flag, path-loaded "
            "checkpoints get empty reference tables, so any test that "
            "drives FormationEnergyCalculator or the single-atom-patch "
            "code path raises KeyError on first task lookup. The "
            "registered model must use the same task heads as the "
            "candidate. No-op (and validated as such) when --sweep-model "
            "is a registered name — name-mode already auto-fetches the "
            "YAMLs from the matching model entry."
        ),
    )
    parser.addoption(
        "--eager-inference",
        action="store_true",
        default=False,
        help=(
            "Run shared inference fixtures without torch.compile or Nitro. "
            "Preserves precision and model options; fails on an unexpected "
            "torch.compile call. Use with -m 'gpu and not compile_gpu'."
        ),
    )
    parser.addoption(
        "--inference-settings",
        action="store",
        default=None,
        help=(
            "Named inference settings (e.g. default, multigpu, batch, traineval) "
            "to run the existing suite in. Shared pretrained-model fixtures load "
            "with it (otherwise high-precision), and tests that opt in, such as "
            "the equivariance tests, use it instead of their own settings. Tests "
            "that pass explicit settings keep them."
        ),
    )


def pytest_configure(config):
    _FIXTURE_SETTINGS["inference_settings"] = "high-precision"
    _FIXTURE_SETTINGS["eager"] = config.getoption("--eager-inference", default=False)
    requested = requested_inference_settings(config)
    if requested is not None:
        from fairchem.core.units.mlip_unit.api.inference import (
            NAME_TO_INFERENCE_SETTING,
        )

        if requested not in NAME_TO_INFERENCE_SETTING:
            raise pytest.UsageError(
                f"--inference-settings={requested!r} is not one of "
                f"{sorted(NAME_TO_INFERENCE_SETTING)}"
            )
        _FIXTURE_SETTINGS["inference_settings"] = requested
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
        "args list model names to lock the test to (e.g. "
        '@pretrained("uma-s-1p1", "uma-s-1p2")) — the test only runs when the '
        "active model is in this list. No args means the test runs against "
        "any model (parametrized via models_to_test or substituted by "
        "--sweep-model).",
    )
    config.addinivalue_line(
        "markers",
        "no_sweep: test asserts behavior/values specific to its declared "
        "@pretrained model(s) (exact reference energies, model-size-specific "
        "logic, or white-box access to the backbone/head internals), so it is "
        "not meaningful under --sweep-model substitution and is deselected there.",
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


def sweep_refs_from(config) -> str | None:
    """
    Return ``--sweep-refs-from`` value, or ``None`` if unset/empty.
    """
    raw = config.getoption("--sweep-refs-from", default=None)
    if raw is None:
        return None
    raw = raw.strip()
    return raw or None


# The named inference settings shared pretrained-model fixtures load with:
# --inference-settings if given (set in pytest_configure), else FP32.
_FIXTURE_SETTINGS = {"inference_settings": "high-precision", "eager": False}


def requested_inference_settings(config) -> str | None:
    """
    Return ``--inference-settings`` value, or ``None`` if unset/empty.
    """
    raw = config.getoption("--inference-settings", default=None)
    if raw is None:
        return None
    raw = raw.strip()
    return raw or None


def fixture_inference_settings() -> str | InferenceSettings:
    """
    The named inference settings shared pretrained-model fixtures load with.
    """
    return inference_settings_for_test(_FIXTURE_SETTINGS["inference_settings"])


def inference_settings_for_test(
    settings: str | InferenceSettings,
) -> str | InferenceSettings:
    """
    Apply the opt-in eager policy without changing production presets.

    Only compilation and Nitro are disabled. Precision, MOLE merging, graph
    generation, derivatives, and backend selection keep the requested values.
    Explicit settings tests should bypass this helper and use compile_gpu when
    they require compilation.
    """
    if not _FIXTURE_SETTINGS["eager"]:
        return settings

    from dataclasses import replace

    from fairchem.core.units.mlip_unit.api.inference import guess_inference_settings

    return replace(guess_inference_settings(settings), compile=False, nitro=False)


@pytest.fixture(scope="session", autouse=True)
def _guard_eager_inference_compilation(request):
    """
    Fail on unclassified compiler use, including during shared fixture setup.

    The guard covers this pytest process. Ray actors receive explicit eager
    settings through their serialized predictors rather than inheriting patches.
    """
    if not request.config.getoption("--eager-inference", default=False):
        yield
        return

    def reject_compile(*args, **kwargs):
        raise AssertionError(
            "torch.compile was called under --eager-inference. Move this case "
            "to compile_gpu or pass explicit eager inference settings."
        )

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(torch, "compile", reject_compile)
        yield


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

    Normally parametrized by ``pytest_generate_tests`` (via
    ``indirect=True``) from the test's ``@pytest.mark.pretrained(...)``
    marker or by ``--sweep-model`` — in that case ``request.param`` is
    set. Falls back to ``--sweep-model`` / marker resolution when the
    fixture is consumed by an unittest.TestCase, where pytest does NOT
    apply ``pytest_generate_tests`` parametrization.
    """
    if hasattr(request, "param"):
        return request.param

    override = sweep_model(request.config)
    if override is not None:
        return override

    marker = request.node.get_closest_marker("pretrained")
    if marker is None or not marker.args:
        raise RuntimeError(
            f"{request.node.nodeid} uses pretrained_checkpoint but its "
            "closest @pytest.mark.pretrained marker is missing or has no "
            "model arguments. Add models (e.g. "
            "@pytest.mark.pretrained('uma-s-1p1')) or run with "
            "--sweep-model=<model>."
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

    # --sweep-model: select only @pretrained tests. Tests with declared
    # @pretrained(M, ...) args are deselected unless the sweep model is
    # in those args. Bare @pretrained (no args) means "any model" and
    # always runs.
    #
    # Path-style sweep (--sweep-model=/abs/path/cand.pt) is the
    # candidate-checkpoint workflow: enable validation of an unreleased
    # checkpoint against the existing test suite. In that mode we keep
    # tests whose declared args include any UMA model (the family check)
    # — a path can plausibly substitute for any UMA-S checkpoint, but
    # not for an allscaip/esen/etc. test locked to a non-UMA model.
    sweep_override = sweep_model(config)
    if sweep_override is not None:
        import os

        sweep_is_path = os.path.isfile(sweep_override)
        uma_set = set(uma_models()) if sweep_is_path else None
        keep, deselect = [], []
        for item in items:
            # no_sweep tests assert model-specific behavior/values; substituting
            # a different checkpoint into them is not meaningful.
            if item.get_closest_marker("no_sweep") is not None:
                deselect.append(item)
                continue
            marker = item.get_closest_marker("pretrained")
            if marker is None:
                deselect.append(item)
                continue
            declared = set(marker.args) if marker.args else None
            if declared is not None:
                if sweep_is_path:
                    # Path-style: keep iff declared intersects the UMA family.
                    if not (declared & uma_set):
                        deselect.append(item)
                        continue
                else:
                    # Name-style: strict match.
                    if sweep_override not in declared:
                        deselect.append(item)
                        continue
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

    refs = sweep_refs_from(config)
    sweep = sweep_model(config)

    # Validate --sweep-refs-from independently of --sweep-model. The
    # "only meaningful with a path-mode --sweep-model" rule applies even
    # when --sweep-model is unset; without this early check the flag
    # silently no-ops.
    if refs is not None:
        if refs not in available:
            raise pytest.UsageError(
                f"--sweep-refs-from={refs!r} is not a registered model. "
                f"Registered models: {sorted(available)}."
            )
        sweep_is_path = sweep is not None and os.path.isfile(sweep)
        if not sweep_is_path:
            raise pytest.UsageError(
                f"--sweep-refs-from={refs!r} is only meaningful when "
                f"--sweep-model is a filesystem path; the registered-name "
                f"branch already auto-fetches reference YAMLs."
            )

    if sweep is None:
        return

    is_registered = sweep in available
    # Distinguish "looks like a path" from "is a real file" so we can
    # give a precise error when the user passed a path (absolute or
    # not) that doesn't resolve, rather than a generic "neither a
    # registered name nor a file" message.
    looks_like_path = (os.sep in sweep) or sweep.endswith(".pt")
    if looks_like_path and not is_registered:
        if not os.path.isabs(sweep):
            raise pytest.UsageError(
                f"--sweep-model={sweep!r} looks like a path but is not "
                f"absolute. Use an absolute path so the same flag value "
                f"resolves identically regardless of pytest's CWD."
            )
        if not os.path.isfile(sweep):
            raise pytest.UsageError(
                f"--sweep-model={sweep!r} is an absolute path but the "
                f"file does not exist."
            )
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


def get_predict_unit_for_test(name_or_path, *, refs_from=None, **kwargs):
    """
    Resolve a registered model name or filesystem path to a ``MLIPPredictUnit``.

    Tries the registry first to avoid CWD collisions where a file
    happens to share a registered model name. Falls back to
    ``load_predict_unit`` for filesystem paths. If the caller passes
    a kwarg that ``load_predict_unit`` does not accept (e.g.
    ``cache_dir``) on the path branch, the resulting ``TypeError``
    is the right signal — silent dropping would hide intent.

    ``refs_from`` (path branch only): name of a registered model whose
    ``atom_refs`` / ``form_elem_refs`` YAMLs should be fetched and
    threaded into the path-loaded predictor. Lets path-mode loads drive
    code paths (``FormationEnergyCalculator``, the single-atom patch)
    that look up entries in those tables. Ignored when ``name_or_path``
    resolves to a registered name — name-mode already fetches them.
    """
    import os

    from fairchem.core import pretrained_mlip

    # FP32 (high-precision) unless --inference-settings names another mode;
    # tests of a specific mode pass it explicitly.
    kwargs.setdefault("inference_settings", fixture_inference_settings())
    if name_or_path in pretrained_mlip.available_models:
        return pretrained_mlip.get_predict_unit(name_or_path, **kwargs)
    if os.path.isfile(name_or_path):
        from fairchem.core.units.mlip_unit import load_predict_unit

        if refs_from is not None:
            from fairchem.core._config import CACHE_DIR
            from fairchem.core.calculate.pretrained_mlip import (
                _MODEL_CKPTS,
                get_reference_energies,
            )

            cache_dir = kwargs.get("cache_dir", CACHE_DIR)
            atom_refs = get_reference_energies(refs_from, "atom_refs", cache_dir)
            if _MODEL_CKPTS.checkpoints[refs_from].form_elem_refs is not None:
                form_elem_refs = get_reference_energies(
                    refs_from, "form_elem_refs", cache_dir
                )["refs"]
            else:
                form_elem_refs = None
            kwargs.setdefault("atom_refs", atom_refs)
            kwargs.setdefault("form_elem_refs", form_elem_refs)
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


# --- Fixtures formerly in tests/core/conftest.py (merged: the core
# suite now has a single conftest) ---
import os
from itertools import product
from random import choice

from ase.calculators.singlepoint import SinglePointCalculator
from ase.db import connect
from ase.io import write
from filelock import FileLock
from pymatgen.core import Structure
from pymatgen.core.periodic_table import Element
from syrupy.extensions.amber import AmberSnapshotExtension

from fairchem.core.calculate import pretrained_mlip
from fairchem.core.datasets.ase_datasets import AseDBDataset, AseReadDataset
from fairchem.core.units.mlip_unit.mlip_unit import (
    UNIT_INFERENCE_CHECKPOINT,
    UNIT_RESUME_CONFIG,
)
from tests.testing_utils import launch_main
from tests.units.mlip_unit.create_fake_dataset import create_fake_uma_dataset

DEFAULT_RTOL = 1.0e-03
DEFAULT_ATOL = 1.0e-03


class Approx:
    """
    Wrapper object for approximately compared numpy arrays.
    """

    def __init__(
        self,
        data: np.ndarray | list,
        *,
        rtol: float | None = None,
        atol: float | None = None,
    ) -> None:
        if isinstance(data, list):
            self.data = np.array(data)
        elif isinstance(data, np.ndarray):
            self.data = data
        else:
            raise TypeError(f"Cannot convert {type(data)} to np.array")

        self.rtol: float = rtol if rtol is not None else DEFAULT_RTOL
        self.atol: float = atol if atol is not None else DEFAULT_ATOL
        self.tol_repr = True

    def __repr__(self) -> str:
        data = np.array_repr(self.data)
        data = "\n".join(f"\t{line}" for line in data.splitlines())
        tol_repr = ""
        if self.tol_repr:
            tol_repr = f", \n\trtol={self.rtol}, \n\tatol={self.atol}"
        return f"Approx(\n{data}{tol_repr}\n)"


class _ApproxNumpyFormatter:
    def __init__(self, data) -> None:
        self.data = data

    def __repr__(self) -> str:
        return Approx(
            self.data.expected,
            rtol=self.data.rel,
            atol=self.data.abs,
        ).__repr__()


def _try_parse_approx(data: SerializableData) -> Approx | None:
    """
    Parse the string representation of an Approx object.
    We can just use eval here, since we know the string is safe.
    """
    if not isinstance(data, str):
        return None

    data = data.strip()
    if not data.startswith("Approx("):
        return None

    approx = eval(
        data.replace("dtype=", "dtype=np."),
        {"Approx": Approx, "np": np},
        {"array": np.array},
    )
    if not isinstance(approx, Approx):
        return None

    return approx


class ApproxExtension(AmberSnapshotExtension):
    """
    By default, syrupy uses the __repr__ of the expected (snapshot) and actual values
    to serialize them into strings. Then, it compares the strings to see if they match.

    However, this behavior is not ideal for comparing floats/ndarrays. For example,
    if we have a snapshot with a float value of 0.1, and the actual value is 0.10000000000000001,
    then the strings will not match, even though the values are effectively equal.

    To work around this, we override the serialize method to serialize the expected value
    into a special representation. Then, we override the matches function (which originally does a
    simple string comparison) to parse the expected and actual values into numpy arrays.
    Finally, we compare the arrays using np.allclose.
    """

    def matches(
        self,
        *,
        serialized_data: SerializableData,
        snapshot_data: SerializableData,
    ) -> bool:
        # if both serialized_data and snapshot_data are serialized Approx objects,
        # then we can load them as numpy arrays and compare them using np.allclose
        serialized_approx = _try_parse_approx(serialized_data)
        snapshot_approx = _try_parse_approx(snapshot_data)
        if serialized_approx is not None and snapshot_approx is not None:
            return np.allclose(
                snapshot_approx.data,
                serialized_approx.data,
                rtol=serialized_approx.rtol,
                atol=serialized_approx.atol,
            )

        return super().matches(
            serialized_data=serialized_data, snapshot_data=snapshot_data
        )

    def serialize(self, data, **kwargs):
        # we override the existing serialization behavior
        # of the `pytest.approx()` object to serialize it into a special string.
        if isinstance(data, type(pytest.approx(np.array(0.0)))):
            return super().serialize(_ApproxNumpyFormatter(data), **kwargs)
        elif isinstance(data, type(pytest.approx(0.0))):
            raise NotImplementedError("Scalar approx not implemented yet")
        return super().serialize(data, **kwargs)


@pytest.fixture()
def snapshot(snapshot):
    return snapshot.use_extension(ApproxExtension)


@pytest.fixture()
def torch_deterministic():
    # Setup
    torch.use_deterministic_algorithms(True)
    yield True  # Usability: prints `torch_deterministic=True` if a test fails
    # Tear down
    torch.use_deterministic_algorithms(False)


@pytest.fixture()
def torch_deterministic_warn():
    # warn_only=True: for drift/consistency tests that want reduced nondeterminism
    # but tolerate ops without a deterministic CUDA kernel (e.g. UMA-MoE's
    # index_reduce_ in set_MOLE_coefficients, which would otherwise raise).
    torch.use_deterministic_algorithms(True, warn_only=True)
    yield True
    # Tear down (prevents leaking the global flag into later tests)
    torch.use_deterministic_algorithms(False)


@pytest.fixture(autouse=True)
def _restore_determinism():
    # Safety net: restore the global deterministic-algorithms flag after every
    # test so an inline toggle in one test cannot leak into subsequent tests.
    prev = torch.are_deterministic_algorithms_enabled()
    yield
    torch.use_deterministic_algorithms(prev)


@pytest.fixture(scope="session")
def dummy_element_refs():
    # create some dummy elemental energies from ionic radii (ignore deuterium and tritium included in pmg)
    return np.concatenate(
        [[0], [e.average_ionic_radius for e in Element if e.name not in ("D", "T")]]
    )


@pytest.fixture(scope="session")
def dummy_binary_dataset_path(tmpdir_factory, dummy_element_refs):
    # a dummy dataset with binaries with energy that depends on composition only plus noise
    # Limit to first 83 elements (up to Bismuth) to avoid CUDA indexing errors with rare/synthetic elements
    common_elements = [Element.from_Z(z) for z in range(1, 84)]  # H to Bi
    all_binaries = list(product(common_elements, repeat=2))
    rng = np.random.default_rng(seed=0)

    tmpdir = tmpdir_factory.mktemp("dataset")
    with connect(str(tmpdir / "dummy.aselmdb")) as db:
        for i in range(10):
            elements = choice(all_binaries)
            structure = Structure.from_prototype("cscl", species=elements, a=2.0)
            energy = (
                sum(e.average_ionic_radius for e in elements)
                + 0.05 * rng.random() * dummy_element_refs.mean()
            )
            atoms = structure.to_ase_atoms()
            atoms.calc = SinglePointCalculator(
                atoms,
                energy=energy,
                forces=rng.random((2, 3)),
                stress=rng.random((3, 3)),
            )
            # write to the lmdb file
            db.write(atoms, data={"sid": f"structure_{i}"})

            # write it as a cif file as well
            write(str(tmpdir / f"structure_{i}.cif"), atoms)

    return tmpdir


@pytest.fixture(scope="session", params=["asedb", "cif"])
def dummy_binary_dataset(dummy_binary_dataset_path, request):
    config = dict(src=str(dummy_binary_dataset_path))

    if request.param == "cif":
        config["pattern"] = "*.cif"
        return AseReadDataset(config=config)
    else:
        return AseDBDataset(config=config)


@pytest.fixture(scope="session")
def dummy_binary_db_dataset(dummy_binary_dataset_path):
    config = dict(src=str(dummy_binary_dataset_path))
    return AseDBDataset(config=config)


@pytest.fixture(autouse=True)
def run_around_tests():
    yield
    torch.cuda.empty_cache()


@pytest.fixture(scope="session")
def direct_mole_checkpoint(fake_uma_dataset, tmp_path_factory):
    # Use filelock so only one xdist worker trains the model
    root_tmp = tmp_path_factory.getbasetemp().parent
    temp_dir = str(root_tmp / "direct_mole_checkpoint")
    os.makedirs(temp_dir, exist_ok=True)
    timestamp_id = "12345"
    device = "CPU"

    with FileLock(str(root_tmp / "direct_mole_checkpoint.lock"), timeout=600):
        marker = root_tmp / "direct_mole_checkpoint.done"
        if not marker.is_file():
            sys_args = [
                "--config",
                "tests/core/units/mlip_unit/test_mlip_train.yaml",
                "num_experts=8",
                "checkpoint_every=10000",
                "datasets=aselmdb",
                f"+job.run_dir={temp_dir}",
                f"datasets.data_root_dir={fake_uma_dataset}",
                f"job.device_type={device}",
                f"+job.timestamp_id={timestamp_id}",
                "optimizer=savegrad",
                "max_steps=2",
                "max_epochs=null",
                "expected_loss=null",
                "act_type=gate",
                "ff_type=spectral",
            ]
            launch_main(sys_args)
            marker.touch()

    # Now resume from checkpoint_step and should get the same result
    # TODO, should get the run config and get checkpoint location from there
    checkpoint_dir = os.path.join(temp_dir, timestamp_id, "checkpoints", "step_0")
    checkpoint_state_yaml = os.path.join(checkpoint_dir, UNIT_RESUME_CONFIG)
    inference_checkpoint_pt = os.path.join(checkpoint_dir, UNIT_INFERENCE_CHECKPOINT)
    assert os.path.isdir(checkpoint_dir)
    assert os.path.isfile(checkpoint_state_yaml)
    assert os.path.isfile(inference_checkpoint_pt)

    return inference_checkpoint_pt, checkpoint_state_yaml


@pytest.fixture(scope="session")
def direct_checkpoint(fake_uma_dataset, tmp_path_factory):
    # Use filelock so only one xdist worker trains the model
    root_tmp = tmp_path_factory.getbasetemp().parent
    temp_dir = str(root_tmp / "direct_checkpoint")
    os.makedirs(temp_dir, exist_ok=True)
    timestamp_id = "12345"
    device = "CPU"

    with FileLock(str(root_tmp / "direct_checkpoint.lock"), timeout=600):
        marker = root_tmp / "direct_checkpoint.done"
        if not marker.is_file():
            sys_args = [
                "--config",
                "tests/core/units/mlip_unit/test_mlip_train.yaml",
                "num_experts=0",
                "checkpoint_every=10000",
                "datasets=aselmdb",
                f"+job.run_dir={temp_dir}",
                f"datasets.data_root_dir={fake_uma_dataset}",
                f"job.device_type={device}",
                f"+job.timestamp_id={timestamp_id}",
                "optimizer=savegrad",
                "max_steps=2",
                "max_epochs=null",
                "expected_loss=null",
                "act_type=gate",
                "ff_type=spectral",
            ]
            launch_main(sys_args)
            marker.touch()

    # Now resume from checkpoint_step and should get the same result
    # TODO, should get the run config and get checkpoint location from there
    checkpoint_dir = os.path.join(temp_dir, timestamp_id, "checkpoints", "step_0")
    checkpoint_state_yaml = os.path.join(checkpoint_dir, UNIT_RESUME_CONFIG)
    inference_checkpoint_pt = os.path.join(checkpoint_dir, UNIT_INFERENCE_CHECKPOINT)
    assert os.path.isdir(checkpoint_dir)
    assert os.path.isfile(checkpoint_state_yaml)
    assert os.path.isfile(inference_checkpoint_pt)

    return inference_checkpoint_pt, checkpoint_state_yaml


@pytest.fixture(scope="session")
def conserving_mole_checkpoint(fake_uma_dataset, tmp_path_factory):
    # Use filelock so only one xdist worker trains the model
    root_tmp = tmp_path_factory.getbasetemp().parent
    temp_dir = str(root_tmp / "conserving_mole_checkpoint")
    os.makedirs(temp_dir, exist_ok=True)
    timestamp_id = "12345"
    device = "CPU"

    with FileLock(str(root_tmp / "conserving_mole_checkpoint.lock"), timeout=600):
        marker = root_tmp / "conserving_mole_checkpoint.done"
        if not marker.is_file():
            sys_args = [
                "--config",
                "tests/core/units/mlip_unit/test_mlip_train_conserving.yaml",
                "num_experts=8",
                "heads.energyandforcehead.module=fairchem.core.models.uma.escn_moe.DatasetSpecificSingleHeadWrapper",
                "checkpoint_every=10000",
                "datasets=aselmdb_conserving",
                f"+job.run_dir={temp_dir}",
                f"datasets.data_root_dir={fake_uma_dataset}",
                f"job.device_type={device}",
                f"+job.timestamp_id={timestamp_id}",
                "optimizer=savegrad",
                "max_steps=2",
                "max_epochs=null",
                "expected_loss=null",
                "act_type=gate",
                "ff_type=spectral",
            ]
            launch_main(sys_args)
            marker.touch()

    # Now resume from checkpoint_step and should get the same result
    # TODO, should get the run config and get checkpoint location from there
    checkpoint_dir = os.path.join(temp_dir, timestamp_id, "checkpoints", "step_0")
    checkpoint_state_yaml = os.path.join(checkpoint_dir, UNIT_RESUME_CONFIG)
    inference_checkpoint_pt = os.path.join(checkpoint_dir, UNIT_INFERENCE_CHECKPOINT)
    assert os.path.isdir(checkpoint_dir)
    assert os.path.isfile(checkpoint_state_yaml)
    assert os.path.isfile(inference_checkpoint_pt)

    return inference_checkpoint_pt, checkpoint_state_yaml


@pytest.fixture(scope="session")
def fake_uma_dataset(tmp_path_factory):
    # Use filelock so only one xdist worker creates the dataset
    root_tmp = tmp_path_factory.getbasetemp().parent
    data_dir = str(root_tmp / "fake_uma_dataset")
    os.makedirs(data_dir, exist_ok=True)

    with FileLock(str(root_tmp / "fake_uma_dataset.lock"), timeout=600):
        marker = root_tmp / "fake_uma_dataset.done"
        if not marker.is_file():
            create_fake_uma_dataset(data_dir)
            marker.touch()

    return data_dir


# _LOCAL_CHECKPOINT = "/checkpoint/ocp/shared/uma_checkpoints/uma_sm_1p2.pt"
@pytest.fixture(scope="session")
def uma_predict_unit(request):
    """
    Predict unit for some UMA model — sweep-aware and exclude-aware.

    * If ``--sweep-model`` is set, loads that checkpoint (name or path).
    * Otherwise picks the first registered UMA model NOT in
      ``--exclude-models``. This ensures the base CI job (which excludes
      ``uma-s-1p1`` and ``uma-s-1p2`` so the sweep jobs own those
      models) does not silently load an excluded model — which would
      both violate the partition contract and double-run that model.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sweep = sweep_model(request.config)
    if sweep:
        return get_predict_unit_for_test(
            sweep,
            device=device,
            refs_from=sweep_refs_from(request.config),
        )
    excluded = excluded_models(request.config)
    candidates = [m for m in uma_models() if m not in excluded]
    if not candidates:
        pytest.skip(
            "No UMA model available outside --exclude-models — sweep jobs "
            "should be used instead."
        )
    return pretrained_mlip.get_predict_unit(
        candidates[0], device=device, inference_settings=fixture_inference_settings()
    )


@pytest.fixture(scope="session")
def uma_predict_unit_alt(request):
    """
    Predict unit for a UMA model different from the one ``uma_predict_unit`` uses.

    When ``--sweep-model`` is a registered UMA model, picks any other UMA
    model. When sweep is unset, picks the second UMA model alphabetically.
    Skips when ``--sweep-model`` is a filesystem path — we cannot reliably
    tell which registered model (if any) corresponds to the path, so we
    cannot guarantee this fixture returns a *different* one.
    """
    available_uma = uma_models()
    if len(available_uma) < 2:
        pytest.skip("Fewer than 2 UMA models available")
    sweep = sweep_model(request.config)
    if sweep and sweep not in available_uma:
        # path-style sweep — primary model identity is opaque
        pytest.skip("--sweep-model is a path; cannot pick a distinct UMA model")
    candidates = (
        [m for m in available_uma if m != sweep] if sweep else available_uma[1:]
    )
    if not candidates:
        pytest.skip("No UMA model available that differs from sweep model")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return pretrained_mlip.get_predict_unit(
        candidates[0], device=device, inference_settings=fixture_inference_settings()
    )


@pytest.fixture(scope="module")
def declared_predict_unit(pretrained_checkpoint, request):
    """
    Predict unit for the model(s) declared in the test's ``@pretrained(...)`` marker.

    Parametrized by the root conftest ``pytest_generate_tests`` hook, which reads
    model names from the ``@pytest.mark.pretrained("uma-s-1p1", ...)`` decorator.

    Shared across test modules — import via fixture dependency (just add
    ``declared_predict_unit`` to the test function signature).  Module-scoped so
    the model is loaded once per (module, model-name) pair.
    """
    return get_predict_unit_for_test(
        pretrained_checkpoint,
        refs_from=sweep_refs_from(request.config),
    )
