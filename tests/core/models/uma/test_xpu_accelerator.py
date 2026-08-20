"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

End-to-end check that a UMA/eSEN-family backbone runs on whatever accelerator
is present -- in particular Intel GPUs (XPU), which stock fairchem refused.

Deliberately runs both forward AND backward. A missing accelerator autograd
kernel produces a model that imports, loads, and returns an energy, but cannot
do MD or training -- the worst kind of pass.
"""

from __future__ import annotations

import pytest
import torch
from ase import build

from fairchem.core.common import device_utils as du
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets.collaters.simple_collater import data_list_collater
from fairchem.core.models.base import HydraModelV2
from fairchem.core.models.uma.escn_md import MLP_EFS_Head, eSCNMDBackbone

ACCEL = du.get_available_accelerator()
pytestmark = pytest.mark.skipif(
    ACCEL is None, reason="no cuda/xpu accelerator on this node"
)

CUTOFF = 6.0


def _backbone_config(direct_forces: bool, regress_stress: bool) -> dict:
    return {
        "max_num_elements": 100,
        "sphere_channels": 16,
        "lmax": 2,
        "mmax": 2,
        "otf_graph": False,
        "max_neighbors": 300,
        "cutoff": CUTOFF,
        "edge_channels": 16,
        "num_layers": 2,
        "hidden_channels": 16,
        "norm_type": "rms_norm_sh",
        "act_type": "gate",
        "ff_type": "spectral",
        "activation_checkpointing": False,
        "chg_spin_emb_type": "pos_emb",
        "cs_emb_grad": False,
        "dataset_emb_grad": False,
        "dataset_mapping": {"omat": "omat"},
        "regress_stress": regress_stress,
        "direct_forces": direct_forces,
        "regress_forces": True,
    }


def _cu_batch(device: str, sizes=(1, 2)):
    """Cu bulk -- the metal this project's electrocatalysis work targets."""
    samples = []
    for size in sizes:
        atoms = build.bulk("Cu", "fcc", a=3.58, cubic=True).repeat((size,) * 3)
        sample = AtomicData.from_ase(
            atoms, max_neigh=300, radius=CUTOFF, r_edges=True
        )
        sample.natoms = torch.tensor(len(atoms))
        sample.charge = torch.LongTensor([0])
        sample.spin = torch.LongTensor([0])
        sample.dataset = "omat"
        samples.append(sample)
    return data_list_collater(samples, otf_graph=True).to(device)


def _build(direct_forces: bool, regress_stress: bool, device: str):
    backbone = eSCNMDBackbone(**_backbone_config(direct_forces, regress_stress))
    model = HydraModelV2(
        backbone, {"efs_head": MLP_EFS_Head(backbone, wrap_property=False)}
    )
    return model.to(device)


def _unwrap(out):
    return out["efs_head"] if "efs_head" in out else out


def test_conservative_forces_via_accelerator_autograd():
    """Forces = -dE/dx, so a finite result proves backward works on-device."""
    model = _build(direct_forces=False, regress_stress=True, device=ACCEL).eval()
    out = _unwrap(model(_cu_batch(ACCEL)))
    du.synchronize(ACCEL)

    energy, forces = out["energy"], out["forces"]
    assert energy.device.type == ACCEL
    assert forces.device.type == ACCEL
    assert torch.isfinite(energy).all()
    assert torch.isfinite(forces).all()
    # Newton's third law: internal forces on an isolated periodic cell sum to 0.
    assert torch.allclose(
        forces.sum(dim=0), torch.zeros(3, device=forces.device), atol=1e-3
    )


def test_parameter_gradients_flow_on_accelerator():
    """The training path: every parameter must receive a finite gradient."""
    model = _build(direct_forces=True, regress_stress=False, device=ACCEL)
    model.train()
    out = _unwrap(model(_cu_batch(ACCEL)))
    loss = sum(
        v.pow(2).sum()
        for k, v in out.items()
        if torch.is_tensor(v) and v.requires_grad and k != "embeddings"
    )
    loss.backward()
    du.synchronize(ACCEL)

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no parameter received a gradient"
    assert all(torch.isfinite(g).all() for g in grads)
    assert any(torch.count_nonzero(g) > 0 for g in grads)


def test_accelerator_matches_cpu_within_tolerance():
    """Same weights, same input -> same energy on CPU and on the accelerator."""
    torch.manual_seed(0)
    model_cpu = _build(direct_forces=False, regress_stress=True, device="cpu").eval()
    model_acc = _build(direct_forces=False, regress_stress=True, device="cpu").eval()
    model_acc.load_state_dict(model_cpu.state_dict())
    model_acc = model_acc.to(ACCEL)

    e_cpu = _unwrap(model_cpu(_cu_batch("cpu")))["energy"].detach().cpu()
    e_acc = _unwrap(model_acc(_cu_batch(ACCEL)))["energy"].detach().cpu()
    du.synchronize(ACCEL)

    assert torch.allclose(e_cpu, e_acc, atol=1e-3, rtol=1e-3), (
        f"cpu={e_cpu.tolist()} {ACCEL}={e_acc.tolist()}"
    )


# --------------------------------------------------------------------------
# The full MLIPPredictUnit path -- this is the code that used to hard-stop.
# --------------------------------------------------------------------------


def _predict_batch():
    """A single small molecule in the format the mole test checkpoint expects."""
    from fairchem.core.datasets.atomic_data import AtomicData
    from fairchem.core.datasets.collaters.simple_collater import data_list_collater

    atoms = build.molecule("H2O")
    atoms.pbc = False
    sample = AtomicData.from_ase(
        atoms,
        max_neigh=10,
        radius=100,
        r_energy=False,
        r_forces=False,
        r_edges=False,
        r_data_keys=["spin", "charge"],
    )
    sample["dataset"] = "oc20"
    return data_list_collater([sample], otf_graph=True)


def test_predict_unit_accepts_accelerator(conserving_mole_checkpoint):
    """MLIPPredictUnit on the accelerator, end to end.

    Stock fairchem asserted `device in ["cpu", "cuda"]` here, so device="xpu"
    raised outright; and because anything not "cuda" resolved to CPU, merely
    widening the assert would have produced a silently slow CPU run instead.
    This asserts the unit actually lands on the accelerator.
    """
    from fairchem.core.units.mlip_unit.predict import MLIPPredictUnit

    inference_checkpoint_pt, _ = conserving_mole_checkpoint
    predictor = MLIPPredictUnit(inference_checkpoint_pt, device=ACCEL)

    assert torch.device(predictor.device).type == ACCEL, (
        f"requested {ACCEL} but predict unit resolved to {predictor.device} -- "
        "this is the silent CPU downgrade the port exists to prevent"
    )

    # Device placement is lazy upstream: _lazy_init() runs on the first
    # predict(), so the model sits on CPU until then. Drive a real prediction
    # rather than inspecting parameters early.
    out = predictor.predict(_predict_batch())
    du.synchronize(ACCEL)

    assert next(predictor.model.parameters()).device.type == ACCEL
    assert "energy" in out and "forces" in out
    assert torch.isfinite(out["energy"]).all()
    assert torch.isfinite(out["forces"]).all()


def test_predict_unit_auto_selects_accelerator(conserving_mole_checkpoint):
    """device="auto" resolves to the hardware actually present."""
    from fairchem.core.units.mlip_unit.predict import MLIPPredictUnit

    inference_checkpoint_pt, _ = conserving_mole_checkpoint
    predictor = MLIPPredictUnit(inference_checkpoint_pt, device="auto")
    assert torch.device(predictor.device).type == ACCEL


def test_predict_unit_refuses_absent_accelerator(conserving_mole_checkpoint):
    """Asking for hardware this node lacks must raise, not fall back to CPU."""
    from fairchem.core.units.mlip_unit.predict import MLIPPredictUnit

    absent = [
        d for d in du.ACCELERATOR_DEVICE_TYPES if not du.accelerator_is_available(d)
    ]
    if not absent:
        pytest.skip("this node has every supported accelerator")
    inference_checkpoint_pt, _ = conserving_mole_checkpoint
    with pytest.raises((RuntimeError, ValueError, AssertionError)):
        MLIPPredictUnit(inference_checkpoint_pt, device=absent[0])
