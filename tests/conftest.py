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
        "--uma-checkpoint",
        action="store",
        default=None,
        help=(
            "Sweep mode: name (e.g. 'uma-s-1p3') or filesystem path of the "
            "UMA checkpoint to run all UMA-using tests against. When set, "
            "tests without @pytest.mark.uses_uma are deselected and tests "
            "with @pytest.mark.checkpoint_specific are skipped."
        ),
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "ocpapi_integration: ocpapi integration test")
    config.addinivalue_line("markers", "gpu: mark test to run only on GPU workers")
    config.addinivalue_line(
        "markers",
        "serial: mark test to run serially on the CPU runner (not parallelized with xdist)",
    )
    config.addinivalue_line(
        "markers",
        "subprocess: mark test that spawns subprocesses (excluded from parallel xdist run)",
    )
    config.addinivalue_line(
        "markers",
        "uses_uma: test exercises a UMA pretrained checkpoint. Required for "
        "tests to be selected under --uma-checkpoint sweep mode.",
    )
    config.addinivalue_line(
        "markers",
        "uma_models(*names): UMA model names to run this test against when "
        "--uma-checkpoint is not set.",
    )
    config.addinivalue_line(
        "markers",
        "checkpoint_specific: test asserts values calibrated to a specific "
        "UMA checkpoint and is skipped under --uma-checkpoint sweep mode.",
    )


def _uma_model_values(metafunc) -> list[str]:
    override = metafunc.config.getoption("--uma-checkpoint")
    if override is not None:
        return [override]

    marker = metafunc.definition.get_closest_marker("uma_models")
    if marker is None or not marker.args:
        raise RuntimeError(
            f"{metafunc.function.__name__} uses uma_model_name/uma_checkpoint "
            "but does not declare @pytest.mark.uma_models(...)."
        )
    return list(marker.args)


@pytest.fixture(scope="module")
def uma_checkpoint(request):
    """
    Name or path of the UMA checkpoint under test.

    This fixture is parametrized by pytest_generate_tests from the test's
    @pytest.mark.uma_models(...) declaration, or by --uma-checkpoint in sweep mode.
    """
    if hasattr(request, "param"):
        return request.param

    override = request.config.getoption("--uma-checkpoint")
    if override is not None:
        return override

    marker = request.node.get_closest_marker("uma_models")
    if marker is None or not marker.args:
        raise RuntimeError(
            f"{request.node.nodeid} uses uma_checkpoint but does not declare "
            "@pytest.mark.uma_models(...)."
        )
    return marker.args[0]


def pytest_generate_tests(metafunc):
    """
    Provide values for UMA checkpoint parameters:
    - default: checkpoints declared by @pytest.mark.uma_models(...)
    - --uma-checkpoint=...: just the override

    Tests opt in by taking `uma_model_name` or `uma_checkpoint` as an argument.
    """
    if "uma_model_name" in metafunc.fixturenames:
        metafunc.parametrize(
            "uma_model_name", _uma_model_values(metafunc), scope="module"
        )
    if "uma_checkpoint" in metafunc.fixturenames:
        metafunc.parametrize(
            "uma_checkpoint",
            _uma_model_values(metafunc),
            indirect=True,
            scope="module",
        )


def pytest_runtest_setup(item):
    # Check if the test has the 'gpu' marker
    if "gpu" in item.keywords and not torch.cuda.is_available():
        pytest.skip("CUDA not available, skipping GPU test")
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
                "fairchem_cpp not found, skipping DGL tests! please install fairchem if you want to run these"
            )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--skip-ocpapi-integration"):
        skip_ocpapi_integration = pytest.mark.skip(reason="skipping ocpapi integration")
        for item in items:
            if "ocpapi_integration_test" in item.keywords:
                item.add_marker(skip_ocpapi_integration)
        return
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

    # Sweep mode: --uma-checkpoint deselects non-UMA tests and skips
    # checkpoint_specific tests so the named checkpoint runs against the
    # full set of model-agnostic UMA tests.
    uma_override = config.getoption("--uma-checkpoint")
    if uma_override is not None:
        skip_specific = pytest.mark.skip(
            reason=f"checkpoint_specific test skipped under --uma-checkpoint={uma_override}"
        )
        keep, deselect = [], []
        for item in items:
            if not item.get_closest_marker("uses_uma"):
                deselect.append(item)
                continue
            if item.get_closest_marker("checkpoint_specific"):
                item.add_marker(skip_specific)
            keep.append(item)
        if deselect:
            config.hook.pytest_deselected(items=deselect)
            items[:] = keep


def seed_everywhere(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@pytest.fixture()
def seed_fixture():
    seed_everywhere(42)  # You can set your desired seed value here


@pytest.fixture()
def compile_reset_state():
    torch.compiler.reset()
    yield
    torch.compiler.reset()


@pytest.fixture(scope="session")
def water_xyz_file(tmp_path_factory):
    """Provide a reusable minimal water molecule XYZ file path.

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


def get_predict_unit_for_test(name_or_path, **kwargs):
    """
    Wrapper that handles both registered model names and filesystem paths.

    When ``name_or_path`` is an existing file on disk (e.g. passed via
    ``--uma-checkpoint /path/to/file.pt``), we load with
    ``load_predict_unit`` directly.  Otherwise we delegate to the
    production ``get_predict_unit`` which resolves registered names.
    """
    import os

    if os.path.exists(name_or_path):
        from fairchem.core.units.mlip_unit import load_predict_unit

        return load_predict_unit(
            name_or_path, atom_refs=None, form_elem_refs=None, **kwargs
        )
    from fairchem.core import pretrained_mlip

    return pretrained_mlip.get_predict_unit(name_or_path, **kwargs)


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
