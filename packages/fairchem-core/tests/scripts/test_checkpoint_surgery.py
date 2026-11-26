"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ase.calculators.calculator import Calculator, all_changes

from fairchem.core.scripts import uma_1p2_surgery, uma_1p2p1_surgery


@pytest.mark.parametrize("surgery", [uma_1p2_surgery, uma_1p2p1_surgery])
def test_surgery_preserves_trained_experts_and_removes_unused_dataset(
    tmp_path, surgery
):
    embedding_key = "backbone.dataset_embedding.dataset_emb_dict.omat.weight"
    head_keys = [
        f"output_heads.energyandforcehead.head.energy_block.{layer}.weights"
        for layer in (0, 2, 4)
    ]
    weights = {embedding_key: torch.tensor([[1.0, 2.0]])}
    weights.update({key: torch.arange(16).reshape(8, 2) for key in head_keys})
    checkpoint = SimpleNamespace(
        model_config={
            "backbone": {"dataset_list": ["omat", "omat_rattle"]},
            "heads": {"energyandforcehead": {"dataset_names": ["omat", "omat_rattle"]}},
        },
        model_state_dict=weights,
        ema_state_dict={
            f"module.{key}": value.clone() for key, value in weights.items()
        },
        tasks_config=[{"datasets": ["omat"]}, {"datasets": ["omat_rattle"]}],
    )
    input_path = tmp_path / "input.pt"
    torch.save(checkpoint, input_path)
    stage1, stage2 = surgery.uma_1p2_surgery(str(input_path), str(tmp_path / "output"))
    first = torch.load(stage1, weights_only=False)
    second = torch.load(stage2, weights_only=False)
    rattle_key = embedding_key.replace("omat.weight", "omat_rattle.weight")
    assert torch.equal(first.model_state_dict[rattle_key], weights[embedding_key])
    assert (
        first.model_state_dict[rattle_key].data_ptr()
        != first.model_state_dict[embedding_key].data_ptr()
    )
    assert rattle_key not in second.model_state_dict
    assert f"module.{rattle_key}" not in second.ema_state_dict
    assert second.tasks_config == [{"datasets": ["omat"]}]
    assert second.model_config["supports_single_atoms"]
    mapping = second.model_config["backbone"]["dataset_mapping"]
    assert "omat_rattle" not in mapping
    assert mapping["omat"] == "omat"
    assert len(set(mapping.values())) == 7
    for key in head_keys:
        assert torch.equal(second.model_state_dict[key], weights[key][:7])
        assert torch.equal(second.ema_state_dict[f"module.{key}"], weights[key][:7])
    systems = surgery.create_test_systems()
    assert not systems["nopbc"].pbc.any()
    assert systems["pbc"].pbc.all()
    assert systems["nopbc"].info == {"charge": 0, "spin": 1}


@pytest.mark.parametrize("surgery", [uma_1p2_surgery, uma_1p2p1_surgery])
def test_checkpoint_comparison_detects_energy_and_force_changes(monkeypatch, surgery):
    class ConstantCalculator(Calculator):
        implemented_properties = ("energy", "forces")

        def __init__(self, value):
            super().__init__()
            self.value = value

        def calculate(self, atoms=None, properties=None, system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            self.results = {
                "energy": float(self.value),
                "forces": np.full((len(atoms), 3), self.value),
            }

    monkeypatch.setattr(
        surgery.FAIRChemCalculator,
        "from_model_checkpoint",
        lambda path, **kwargs: ConstantCalculator(int(path)),
    )
    assert surgery.compare_checkpoints("1", "1")
    assert surgery.compare_checkpoints("0", "0")
    assert not surgery.compare_checkpoints("1", "2")
