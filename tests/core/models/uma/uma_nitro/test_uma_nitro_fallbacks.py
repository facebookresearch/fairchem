"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

UMA Nitro's automatic fallbacks, through MLIPPredictUnit.

Every request Nitro cannot serve warns, naming the settings now in use, and
runs the same settings with nitro=False, which for the default and nitro presets
is multigpu: another checkpoint, no CUDA, an unvalidated GPU, a missing
installation, unsupported settings or multi-worker inference at load time;
inputs outside the merged-weights contract and UMANitroNotSupportedError at run
time. The multigpu preset then falls back to the general model as it always has
(for example, on a heterogeneous batch). Any other error propagates.
"""

from __future__ import annotations

import logging
import re
import sys

import pytest
import torch
from ase import Atoms
from ase.build import bulk, molecule

from fairchem.core.units.mlip_unit import MLIPPredictUnit
from fairchem.core.units.mlip_unit.api.inference import (
    UMANitroNotSupportedError,
    describe_inference_settings,
    inference_settings_batch,
)
from fairchem.core.units.mlip_unit.predict import _REDUCED_PRECISION_WARNING
from tests.core.models.uma.uma_nitro.nitro_test_utils import (
    UNSUPPORTED_SETTINGS,
    assert_close,
    batch,
    checkpoint_path,
    general_predictor,
    load_time_reason,
    nitro_settings,
    random_checkpoint,
    requires_nitro_gpu,
)

pytest.importorskip("triton")

_NITRO_FALLBACK = "UMA Nitro cannot serve this request: "
_TO_MULTIGPU = "Now running with nitro=False"
_MULTIGPU_FALLBACK = "Falling back to a less optimized version"


def _assert_on_multigpu(predictor, caplog, reason: str) -> None:
    """
    Nitro fell back, with ``reason``, to multigpu: the preset with nitro=False.
    """
    assert _NITRO_FALLBACK in caplog.text
    assert reason in caplog.text
    assert _TO_MULTIGPU in caplog.text
    settings = predictor.inference_settings
    assert settings.nitro is False
    assert settings.merge_mole is True
    assert settings.compile is True
    assert type(predictor.model.module).__name__ == "HydraModel"


# --------------------------------------------------------------- load time


def test_falls_back_without_cuda(nitro_random_checkpoint_path, caplog):
    with caplog.at_level(logging.WARNING):
        predictor = MLIPPredictUnit(
            nitro_random_checkpoint_path, "cpu", inference_settings=nitro_settings()
        )
    _assert_on_multigpu(predictor, caplog, "nitro mode requires a CUDA device")
    # TF32 does not apply on the CPU: no reduced-precision warning.
    assert "(FP32)." in caplog.text
    assert _REDUCED_PRECISION_WARNING not in caplog.text


@pytest.mark.parametrize(
    ("config_changes", "name"),
    [
        ({"model_id": None, "model_version": 1.1}, "UMA 1.1"),
        ({"model_id": "UMA-S-1.2"}, "UMA-S-1.2"),
    ],
    ids=["uma-1.1", "uma-s-1.2"],
)
def test_other_uma_versions_fall_back_to_multigpu(
    tmp_path, caplog, config_changes, name
):
    """
    Nitro mode names the unsupported UMA version, says to use UMA 1.2.1 for the
    speedup, and runs multigpu. Decided from the checkpoint config, so it applies
    on any device.
    """
    path = tmp_path / "uma.pt"
    torch.save(random_checkpoint(**config_changes), path)
    with caplog.at_level(logging.WARNING):
        predictor = MLIPPredictUnit(
            str(path), "cpu", inference_settings=nitro_settings()
        )
    _assert_on_multigpu(
        predictor,
        caplog,
        f"{name} is not supported in nitro mode; use UMA 1.2.1 (uma-s-1p2p1) "
        "for the nitro speedup",
    )


def test_parallel_predict_unit_falls_back_to_multigpu(monkeypatch, caplog):
    """
    Multi-worker inference builds its workers with the nitro preset minus nitro.
    """
    pytest.importorskip("ray")
    from fairchem.core.units.mlip_unit import predict

    class _Built(Exception):
        pass

    built = {}

    def record(*args, inference_settings=None, **kwargs):
        built["settings"] = inference_settings
        raise _Built

    monkeypatch.setattr(predict, "MLIPPredictUnit", record)
    with caplog.at_level(logging.WARNING), pytest.raises(_Built):
        predict.ParallelMLIPPredictUnit(
            "unused.pt", inference_settings=nitro_settings(), num_workers=2
        )
    assert "does not support multi-worker" in caplog.text
    assert "Now running with nitro=False (FP32)." in caplog.text
    assert built["settings"] == nitro_settings(nitro=False)


@pytest.mark.gpu()
@pytest.mark.parametrize(("change", "reason"), UNSUPPORTED_SETTINGS)
def test_falls_back_for_unsupported_settings(
    nitro_random_checkpoint_path, caplog, change, reason
):
    """
    Settings Nitro cannot honor run without nitro, exactly as requested.
    """
    with caplog.at_level(logging.WARNING):
        predictor = MLIPPredictUnit(
            nitro_random_checkpoint_path,
            "cuda",
            inference_settings=nitro_settings(**change),
        )
    assert _NITRO_FALLBACK in caplog.text
    assert load_time_reason(reason) in caplog.text
    assert predictor.inference_settings.nitro is False
    for key, value in change.items():
        assert getattr(predictor.inference_settings, key) == value, key
    assert type(predictor.model.module).__name__ == "HydraModel"
    # float64 or tf32=False make the fallback full precision; otherwise
    # tf32=True keeps it reduced.
    full_precision = (
        change.get("base_precision_dtype") == torch.float64
        or change.get("tf32") is False
    )
    assert caplog.text.count(_REDUCED_PRECISION_WARNING) == (not full_precision)
    precision = "(FP64)" if "base_precision_dtype" in change else "(FP32)"
    assert (precision if full_precision else "(TF32)") in caplog.text


@pytest.mark.gpu()
def test_falls_back_on_an_unvalidated_gpu(
    nitro_random_checkpoint_path, monkeypatch, caplog
):
    """
    A GPU outside Nitro's supported compute capabilities runs multigpu.
    """
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda device=None: (7, 5))
    with caplog.at_level(logging.WARNING):
        predictor = MLIPPredictUnit(
            nitro_random_checkpoint_path, "cuda", inference_settings=nitro_settings()
        )
    _assert_on_multigpu(
        predictor, caplog, "supports only CUDA compute capability 8.x-12.x"
    )
    assert "not 7.5" in caplog.text


@pytest.mark.gpu()
@requires_nitro_gpu
def test_falls_back_on_a_gpu_other_than_the_kernels_were_chosen_for(
    nitro_random_checkpoint_path, monkeypatch, caplog
):
    """
    The kernels are chosen once, for the GPU current at import; a predictor on
    a different supported GPU (a mixed node) runs multigpu.
    """
    from fairchem.core.models.uma.uma_nitro import _inference as inference

    actual = torch.cuda.get_device_capability()
    other = (9, 0) if actual[0] != 9 else (10, 0)
    monkeypatch.setattr(inference, "_CUDA_CAPABILITY", other)
    with caplog.at_level(logging.WARNING):
        predictor = MLIPPredictUnit(
            nitro_random_checkpoint_path, "cuda", inference_settings=nitro_settings()
        )
    _assert_on_multigpu(predictor, caplog, "UMA Nitro chose its kernels for")


@pytest.mark.gpu()
def test_falls_back_for_a_modified_uma_1p2p1_file(nitro_random_checkpoint_path, caplog):
    """
    A file tagged UMA-S-1.2.1 that is not on the MD5 allowlist.
    """
    with caplog.at_level(logging.WARNING):
        predictor = MLIPPredictUnit(
            nitro_random_checkpoint_path, "cuda", inference_settings=nitro_settings()
        )
    _assert_on_multigpu(
        predictor, caplog, load_time_reason("is not supported in nitro mode (md5 ")
    )


@pytest.mark.gpu()
def test_falls_back_when_not_installed(
    nitro_random_checkpoint_path, monkeypatch, caplog
):
    monkeypatch.setitem(sys.modules, "fairchem.core.models.uma.uma_nitro", None)
    with caplog.at_level(logging.WARNING):
        predictor = MLIPPredictUnit(
            nitro_random_checkpoint_path, "cuda", inference_settings=nitro_settings()
        )
    _assert_on_multigpu(predictor, caplog, "kernels could not be imported")


# ---------------------------------------------------------------- run time

_COPPER = bulk("Cu", "fcc", a=3.6) * (2, 2, 2)
_O2 = Atoms("O2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 1.21]])


@pytest.fixture(scope="module")
def general(pretrained_model_name):
    """
    The general, unmerged HydraModel with tf32=True: what multigpu falls back to,
    and the reference for multigpu's results.
    """
    return general_predictor(checkpoint_path(pretrained_model_name), tf32=True)


@pytest.mark.gpu()
@pytest.mark.pretrained("uma-s-1p2p1", tasks=("omat", "omc", "omol"))
@pytest.mark.parametrize(
    ("first", "second", "reason"),
    [
        (
            (_COPPER, "omat"),
            [(bulk("NaCl", "rocksalt", a=5.6), "omat")],
            "Compositions",
        ),
        ((molecule("H2O"), "omol", 0, 1), [(molecule("H2O"), "omol", 1, 2)], "Charge"),
        ((_O2, "omol", 0, 3), [(_O2, "omol", 0, 1)], "Spin"),
        ((_COPPER, "omat"), [(_COPPER, "omc")], "Task"),
        ((_COPPER, "omat"), [(_COPPER, "omat"), (_COPPER, "omat")], "one system"),
    ],
    ids=["composition", "charge", "spin", "task", "homogeneous-batch"],
)
@requires_nitro_gpu
def test_runtime_contract_break_falls_back_to_multigpu(
    pretrained_model_name, general, caplog, first, second, reason
):
    """
    After Nitro has run, an input outside its merged weights runs multigpu, which
    merges for that input; the switch is permanent for the predictor.
    """
    predictor = MLIPPredictUnit(
        checkpoint_path(pretrained_model_name),
        "cuda",
        inference_settings=nitro_settings(),
    )
    predictor.predict(batch(first))
    assert predictor.inference_settings.nitro is True

    with caplog.at_level(logging.WARNING):
        output = predictor.predict(batch(*second))
    _assert_on_multigpu(predictor, caplog, reason)
    assert _MULTIGPU_FALLBACK not in caplog.text
    assert_close(output, general.predict(batch(*second)))


@pytest.mark.gpu()
@pytest.mark.pretrained("uma-s-1p2p1", tasks=("omat",))
@requires_nitro_gpu
def test_multigpu_then_falls_back_to_general(pretrained_model_name, general, caplog):
    """
    Once on multigpu, a further composition change takes multigpu's own fallback.
    """
    predictor = MLIPPredictUnit(
        checkpoint_path(pretrained_model_name),
        "cuda",
        inference_settings=nitro_settings(),
    )
    predictor.predict(batch((_COPPER, "omat")))
    nacl = (bulk("NaCl", "rocksalt", a=5.6), "omat")
    predictor.predict(batch(nacl))  # nitro -> multigpu, merged for NaCl
    with caplog.at_level(logging.WARNING):
        output = predictor.predict(batch((_COPPER, "omat")))
    assert _MULTIGPU_FALLBACK in caplog.text
    assert predictor.inference_settings.merge_mole is False
    assert predictor.inference_settings.compile is False
    assert_close(output, general.predict(batch((_COPPER, "omat"))))


@pytest.mark.gpu()
@pytest.mark.pretrained("uma-s-1p2p1", tasks=("omat",))
@requires_nitro_gpu
def test_homogeneous_first_batch_runs_multigpu(pretrained_model_name, general, caplog):
    predictor = MLIPPredictUnit(
        checkpoint_path(pretrained_model_name),
        "cuda",
        inference_settings=nitro_settings(),
    )
    systems = [(_COPPER, "omat"), (bulk("Cu", "fcc", a=3.6), "omat")]
    with caplog.at_level(logging.WARNING):
        output = predictor.predict(batch(*systems))
    _assert_on_multigpu(predictor, caplog, "one system")
    assert output["energy"].shape == (2,)
    assert_close(output, general.predict(batch(*systems)))


@pytest.mark.gpu()
@pytest.mark.pretrained("uma-s-1p2p1", tasks=("omat",))
@requires_nitro_gpu
def test_heterogeneous_batch_falls_back_to_multigpu_then_general(
    pretrained_model_name, general, caplog
):
    """
    The multigpu preset's MOLE merge rejects a batch of different compositions
    too, so the same call continues to the general model.
    """
    predictor = MLIPPredictUnit(
        checkpoint_path(pretrained_model_name),
        "cuda",
        inference_settings=nitro_settings(),
    )
    systems = [(_COPPER, "omat"), (bulk("NaCl", "rocksalt", a=5.6), "omat")]
    with caplog.at_level(logging.WARNING):
        output = predictor.predict(batch(*systems))
    assert _NITRO_FALLBACK in caplog.text
    assert _TO_MULTIGPU in caplog.text
    assert _MULTIGPU_FALLBACK in caplog.text
    settings = predictor.inference_settings
    assert (settings.nitro, settings.merge_mole, settings.compile) == (
        False,
        False,
        False,
    )
    assert output["energy"].shape == (2,)
    assert_close(output, general.predict(batch(*systems)))


@pytest.mark.gpu()
@pytest.mark.pretrained("uma-s-1p2p1", tasks=("omat",))
@requires_nitro_gpu
def test_keeps_running_for_the_same_reduced_composition(
    pretrained_model_name, general, caplog
):
    """
    A larger cell of the same reduced composition stays on Nitro.
    """
    predictor = MLIPPredictUnit(
        checkpoint_path(pretrained_model_name),
        "cuda",
        inference_settings=nitro_settings(),
    )
    with caplog.at_level(logging.WARNING):
        for cell in (_COPPER, bulk("Cu", "fcc", a=3.6) * (3, 3, 3)):
            output = predictor.predict(batch((cell, "omat")))
            expected = general.predict(batch((cell, "omat")))
            assert_close(output, expected)
    assert _NITRO_FALLBACK not in caplog.text
    assert type(predictor.model.module).__name__ == "UMANitroModel"


@pytest.mark.gpu()
@pytest.mark.pretrained("uma-s-1p2p1", tasks=("omat",))
@requires_nitro_gpu
def test_falls_back_on_unsupported_error_during_inference(
    pretrained_model_name, general, monkeypatch, caplog
):
    """
    UMANitroNotSupportedError raised by the forward (e.g. a radius graph that
    keeps outgrowing Nitro's capacity) runs multigpu.
    """
    from fairchem.core.models.uma.uma_nitro import _inference as inference

    def unsupported(*_args, **_kwargs):
        raise UMANitroNotSupportedError("radius-graph capacity still overflowing")

    predictor = MLIPPredictUnit(
        checkpoint_path(pretrained_model_name),
        "cuda",
        inference_settings=nitro_settings(),
    )
    monkeypatch.setattr(inference, "_nitro_forward", unsupported)
    with caplog.at_level(logging.WARNING):
        output = predictor.predict(batch((_COPPER, "omat")))
    _assert_on_multigpu(predictor, caplog, "radius-graph capacity")
    assert_close(output, general.predict(batch((_COPPER, "omat"))))


@pytest.mark.gpu()
@pytest.mark.pretrained("uma-s-1p2p1", tasks=("omat",))
@requires_nitro_gpu
def test_other_errors_propagate(pretrained_model_name, monkeypatch):
    """
    Failures that are not "unsupported" (e.g. non-finite outputs) raise.
    """
    from fairchem.core.models.uma.uma_nitro import _inference as inference

    def non_finite(*_args, **_kwargs):
        raise RuntimeError(inference._NONFINITE_OUTPUT_MSG)

    predictor = MLIPPredictUnit(
        checkpoint_path(pretrained_model_name),
        "cuda",
        inference_settings=nitro_settings(),
    )
    monkeypatch.setattr(inference, "_nitro_forward", non_finite)
    with pytest.raises(RuntimeError, match="non-finite"):
        predictor.predict(batch((_COPPER, "omat")))
    assert predictor.inference_settings.nitro is True


# ------------------------------------------------- reported inference settings


def test_first_predict_logs_the_inference_settings_once(
    nitro_random_checkpoint_path, caplog
):
    predictor = MLIPPredictUnit(
        nitro_random_checkpoint_path,
        "cpu",
        inference_settings=inference_settings_batch(),
    )
    water = (molecule("H2O"), "omol", 0, 1)
    with caplog.at_level(logging.WARNING):
        predictor.predict(batch(water))
        predictor.predict(batch(water))
    assert caplog.text.count(" with inference settings: ") == 1
    assert re.search(r"Running \(.+\) with inference settings: batch\n", caplog.text)


def test_settings_after_a_nitro_fallback_describe_as_multigpu(
    nitro_random_checkpoint_path,
):
    predictor = MLIPPredictUnit(
        nitro_random_checkpoint_path, "cpu", inference_settings=nitro_settings()
    )
    assert describe_inference_settings(predictor.inference_settings) == "multigpu"
