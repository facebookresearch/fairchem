"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

CPU tests for the standalone UMA Nitro model: checkpoint loading, the MOLE merge
and weight layouts against HydraModel, settings checks, and the merged-input
contract. Fallback behavior through the predict unit is in
test_uma_nitro_fallbacks.py.
"""

from __future__ import annotations

import copy
from types import SimpleNamespace

import hydra
import pytest
import torch
from ase import Atoms
from ase.build import bulk, molecule

from fairchem.core.units.mlip_unit.api.inference import (
    InferenceSettings,
    MergeMoleConsistencyError,
    UMANitroNotSupportedError,
)
from tests.core.models.uma.uma_nitro.nitro_test_utils import (
    UNSUPPORTED_SETTINGS,
    batch,
    nitro_settings,
)

pytest.importorskip("triton")

# Nitro's module imports Triton, so it is imported after the skip above.
from fairchem.core.models.uma.uma_nitro import UMANitroModel  # noqa: E402
from fairchem.core.models.uma.uma_nitro._inference import (  # noqa: E402
    _validate_settings,
)


def _reference_tensors(checkpoint, data) -> dict[str, torch.Tensor]:
    """
    HydraModel's merged, fast-PyTorch-prepared tensors under Nitro's names.
    """
    config = copy.deepcopy(checkpoint.model_config)
    config["backbone"]["execution_mode"] = "umas_fast_pytorch"
    model = hydra.utils.instantiate(config)
    model.load_state_dict(checkpoint.model_state_dict)
    model.eval()
    model.prepare_for_inference(
        data, InferenceSettings(merge_mole=True, activation_checkpointing=False)
    )
    head_prefix = "output_heads.energyandforcehead.head."
    tensors = dict(model.named_parameters())
    tensors.update(model.named_buffers())
    tensors = {k.replace(head_prefix, "head."): v for k, v in tensors.items()}
    backbone = model.backbone
    tensors["backbone.wigner_c_l2"] = backbone.wigner_data.custom_kernels.C_l2
    for i, block in enumerate(backbone.blocks):
        for c in (1, 2):
            conv = getattr(block.edge_wise, f"so2_conv_{c}")
            for j, m_conv in enumerate(conv.so2_m_conv):
                name = f"backbone.blocks.{i}.edge_wise.so2_conv_{c}.so2_m_conv.{j}"
                tensors[f"{name}._w_block"] = m_conv._w_block
    return tensors


@pytest.mark.parametrize(
    "system",
    [
        (bulk("Cu", "fcc", a=3.6) * (2, 1, 1), "omat"),
        (bulk("NaCl", "rocksalt", a=5.6), "omat"),
        (molecule("H2O"), "omol", 0, 1),
        (molecule("H2O"), "omol", 1, 2),
        (Atoms("H", positions=[[0.0, 0.0, 0.0]]), "omol", 0, 2),
    ],
    ids=["Cu-omat", "NaCl-omat", "H2O-omol", "H2O+-omol", "H-atom-omol"],
)
def test_weights_match_hydra_merge(nitro_random_checkpoint, system):
    """
    Nitro's merged and laid-out weights equal HydraModel's bit for bit.
    """
    data = batch(system)
    reference = _reference_tensors(nitro_random_checkpoint, data)

    nitro = UMANitroModel.from_checkpoint(nitro_random_checkpoint)
    nitro.prepare_for_inference(data, nitro_settings())
    for block in nitro.backbone.blocks:
        for conv in (block.edge_wise.so2_conv_1, block.edge_wise.so2_conv_2):
            for m_conv in conv.so2_m_conv:
                m_conv._build_w_block()

    tensors = dict(nitro.named_parameters())
    tensors.update(nitro.named_buffers())
    assert tensors
    for name, value in tensors.items():
        assert name in reference, name
        assert value.dtype == reference[name].dtype, name
        assert torch.equal(value.detach(), reference[name].detach()), name


def test_loads_direct_energy_head(nitro_direct_head_checkpoint):
    """
    Direct single-dataset EFS head tensors load without a wrapper namespace.
    """
    nitro = UMANitroModel.from_checkpoint(nitro_direct_head_checkpoint)
    for name, value in nitro.head.named_parameters():
        checkpoint_name = f"module.output_heads.efs.{name}"
        assert torch.equal(
            value.detach(),
            nitro_direct_head_checkpoint.ema_state_dict[checkpoint_name],
        )


def test_verify_checkpoint_file_rejects_other_checkpoints(tmp_path):
    """
    Files outside the pinned checkpoint allowlist fail the checksum gate.
    """
    from fairchem.core.models.uma.uma_nitro import verify_checkpoint_file

    path = tmp_path / "other.pt"
    torch.save({"weights": torch.zeros(3)}, path)
    with pytest.raises(UMANitroNotSupportedError, match="uma-s-1p2p1"):
        verify_checkpoint_file(path)


def test_regress_config_is_conservative(nitro_random_checkpoint):
    """
    The backbone reports the gradients the kernels compute.
    """
    from fairchem.core.models.uma.escn_md import GradRegressConfig

    assert UMANitroModel.from_checkpoint(
        nitro_random_checkpoint
    ).backbone.regress_config == (
        GradRegressConfig(
            direct_forces=False,
            direct_stress=False,
            forces=True,
            stress=True,
            hessian=False,
            hessian_vmap=False,
        )
    )


@pytest.mark.parametrize(("change", "reason"), UNSUPPORTED_SETTINGS)
def test_rejects_unsupported_settings(change, reason):
    with pytest.raises(UMANitroNotSupportedError, match=reason):
        _validate_settings(nitro_settings(**change))


def test_reports_every_unsupported_setting():
    settings = nitro_settings(merge_mole=False, activation_checkpointing=True)
    match = "merge_mole=False, activation checkpointing"
    with pytest.raises(UMANitroNotSupportedError, match=match):
        _validate_settings(settings)


@pytest.mark.parametrize(
    "change",
    [
        {},
        {"compile": False},
        {"hessian_vmap": True},
        {"hessian_vmap": False},
        {"edge_chunk_size": 1000},
        {"use_quaternion_wigner": False},
        {"torch_num_threads": 2},
        {"auto_add_default_untrained_tasks": False},
        {"predict_untrained_forces": {"omol"}},
        {"predict_untrained_stress": {"omol"}},
    ],
    ids=lambda change: ",".join(change) or "preset",
)
def test_accepts_settings_it_ignores_or_supports(change):
    """
    Settings that only tune the HydraModel, or tasks Nitro computes anyway.
    """
    _validate_settings(nitro_settings(**change))


def test_merged_input_contract(nitro_random_checkpoint):
    """
    Inputs the merged weights do not cover raise the fast-path fallback error.
    """
    nitro = UMANitroModel.from_checkpoint(nitro_random_checkpoint)
    copper = bulk("Cu", "fcc", a=3.6)
    nitro.prepare_for_inference(batch((copper, "omat")), nitro_settings())

    # Same reduced composition, charge, spin and task: covered.
    nitro.on_predict_check(batch((copper * (2, 2, 1), "omat")))

    for data, reason in [
        (batch((molecule("H2O"), "omat")), "Compositions"),
        (batch((copper, "omat", 1, 0)), "Charge"),
        (batch((copper, "omat", 0, 1)), "Spin"),
        (batch((copper, "omol")), "Task"),
        (batch((copper, "omat"), (copper, "omat")), "one system"),
    ]:
        with pytest.raises(MergeMoleConsistencyError, match=reason):
            nitro.on_predict_check(data)


def test_merge_rejects_batches(nitro_random_checkpoint):
    copper = bulk("Cu", "fcc", a=3.6)
    with pytest.raises(MergeMoleConsistencyError, match="one system"):
        UMANitroModel.from_checkpoint(nitro_random_checkpoint).prepare_for_inference(
            batch((copper, "omat"), (copper, "omat")), nitro_settings()
        )


def test_tasks_match_backbone_defaults(nitro_random_checkpoint):
    """
    Nitro adds the same untrained stress tasks the UMA backbone adds.
    """
    from fairchem.core.models.uma.escn_md import eSCNMDBackbone

    nitro = UMANitroModel.from_checkpoint(nitro_random_checkpoint)
    settings = nitro_settings()
    added = nitro.backbone.get_default_untrained_tasks(nitro.tasks, settings)
    expected = eSCNMDBackbone.get_default_untrained_tasks(
        SimpleNamespace(regress_config=SimpleNamespace(direct_forces=False)),
        nitro.tasks,
        settings,
    )
    assert sorted(t.name for t in added) == sorted(t.name for t in expected)
    assert [t.name for t in added] == ["omol_stress"]

    nitro.add_tasks(added)
    assert sorted(t.property for t in nitro.dataset_to_tasks["omol"]) == [
        "energy",
        "forces",
        "stress",
    ]


def test_rejects_hessian_tasks(nitro_random_checkpoint):
    from fairchem.core.units.mlip_unit.mlip_unit import OutputSpec, Task

    hessian = Task(
        name="omat_hessian",
        level="system",
        property="hessian",
        out_spec=OutputSpec(dim=[1], dtype=torch.float32),
        normalizer=None,
        datasets=["omat"],
    )
    with pytest.raises(ValueError, match="energy, forces and stress only"):
        UMANitroModel.from_checkpoint(nitro_random_checkpoint).add_tasks([hessian])
