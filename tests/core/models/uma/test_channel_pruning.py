"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import hydra
import pytest
import torch
from ase.build import molecule as get_molecule
from omegaconf import OmegaConf

from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.models.uma import channel_pruning as CP
from tests.core.testing_utils import launch_main

# scripts/ is not an importable package -> load the compactor by path.
_CC_PATH = Path(__file__).parents[4] / "scripts" / "compact_channels.py"
_spec = importlib.util.spec_from_file_location("compact_channels", _CC_PATH)
compact_channels = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compact_channels)


def _model_cfg() -> OmegaConf:
    return OmegaConf.create(
        {
            "_target_": "fairchem.core.models.base.HydraModel",
            "backbone": {
                "model": "fairchem.core.models.uma.escn_md.eSCNMDBackbone",
                "max_num_elements": 100,
                "sphere_channels": 16,
                "lmax": 2,
                "mmax": 2,
                "otf_graph": True,
                "edge_channels": 16,
                "num_distance_basis": 32,
                "use_dataset_embedding": True,
                "dataset_list": ["omol"],
                "num_layers": 2,
                "hidden_channels": 16,
                "norm_type": "rms_norm_sh",
                "act_type": "s2",
                "ff_type": "grid",
                "regress_forces": True,
                "direct_forces": True,
                "always_use_pbc": False,
                "distance_function": "gaussian",
            },
            "heads": {
                "omol_energy": {
                    "module": "fairchem.core.models.uma.escn_md.MLP_Energy_Head"
                },
                "forces": {
                    "module": "fairchem.core.models.uma.escn_md.Linear_Force_Head"
                },
            },
        }
    )


def _routeb_cfg(kept: int) -> OmegaConf:
    """
    A rand_emb sphere model configured Route-B style: the RMSNorm computes its
    over-channel statistics over the TARGET kept width (`norm_stats_num_channels`),
    so a model pruned to `kept` channels compacts to sphere_channels=kept EXACTLY.
    """
    cfg = _model_cfg()
    cfg.backbone.chg_spin_emb_type = "rand_emb"  # cleanly sliceable csd front-end
    cfg.backbone.cs_emb_grad = True
    cfg.backbone.norm_stats_num_channels = kept
    return cfg


@pytest.fixture()
def model_and_cfg(seed_fixture):
    cfg = _model_cfg()
    model = hydra.utils.instantiate(cfg)
    model.eval()
    return model, cfg


def _sample(atoms=None) -> AtomicData:
    atoms = atoms if atoms is not None else get_molecule("H2O")
    return AtomicData.from_ase(
        atoms,
        max_neigh=50,
        radius=12,
        task_name="omol",
        r_edges=False,
        r_data_keys=["spin", "charge"],
    )


def _forces(out) -> torch.Tensor:
    return out["forces"]["forces"]


def _energy(out) -> torch.Tensor:
    return out["omol_energy"]["energy"]


# sphere compaction is approximate (RMSNorm centering over the dropped channels,
# absorbed by the heal phase in real training and by the norm-stats override).
_TOL = {"sphere": 5e-2}


@pytest.mark.parametrize("mode", ["sphere"])
def test_validate_spec(model_and_cfg, mode):
    model, _ = model_and_cfg
    rep = CP.validate_spec(model, mode)
    assert rep["params"] > 0
    assert rep["specs"] >= rep["params"]
    assert rep["width"] == 16


@pytest.mark.parametrize("mode", ["sphere"])
def test_importance_and_mask(model_and_cfg, mode):
    model, _ = model_and_cfg
    imp = CP.channel_importance(model, mode)
    assert imp.shape == (16,)
    assert (imp > 0).all(), "a fresh model should have all-nonzero channel importance"
    drop = torch.argsort(imp)[:5]
    CP.apply_channel_mask(model, mode, drop)
    imp2 = CP.channel_importance(model, mode)
    assert float(imp2[drop].abs().max()) == 0.0, "dropped channels must be zeroed"
    keep = torch.tensor([c for c in range(16) if c not in set(drop.tolist())])
    assert float(imp2[keep].min()) > 0.0, "survivors must stay nonzero"


@pytest.mark.parametrize("mode", ["sphere"])
def test_compact_identity(model_and_cfg, mode):
    """Compact with keep-all channels -> outputs match the original exactly."""
    model, cfg = model_and_cfg
    g0 = _sample()
    ref = model(g0)
    keep = torch.arange(16)
    cm, rep = compact_channels.compact(model, cfg, mode, keep=keep)
    cm.eval()
    assert rep["kept"] == 16
    out = cm(_sample())
    assert torch.allclose(_forces(out), _forces(ref), atol=1e-5)
    assert torch.allclose(_energy(out), _energy(ref), atol=1e-5)


@pytest.mark.parametrize("mode", ["sphere"])
def test_compact_roundtrip(model_and_cfg, mode):
    """Prune K channels, compact, and match the pruned (masked) model's output."""
    model, cfg = model_and_cfg
    imp = CP.channel_importance(model, mode)
    drop = torch.argsort(imp)[:4]
    CP.apply_channel_mask(model, mode, drop)
    ref = model(_sample())
    cm, rep = compact_channels.compact(model, cfg, mode)
    cm.eval()
    assert rep["kept"] == 12
    assert rep["param_reduction"] > 0
    out = cm(_sample())
    tol = _TOL[mode]
    assert float((_forces(out) - _forces(ref)).abs().max()) < tol
    assert float((_energy(out) - _energy(ref)).abs().max()) < tol


@pytest.mark.parametrize("mode", ["sphere"])
def test_compacted_equivariance(model_and_cfg, mode):
    """A compacted model is SO(3)-equivariant: energy invariant, forces covariant."""
    model, cfg = model_and_cfg
    imp = CP.channel_importance(model, mode)
    CP.apply_channel_mask(model, mode, torch.argsort(imp)[:4])
    cm, _ = compact_channels.compact(model, cfg, mode)
    cm.eval()

    atoms = get_molecule("H2O")
    r, _ = torch.linalg.qr(torch.randn(3, 3))
    r = r * torch.det(r).sign()  # proper rotation
    rotated = atoms.copy()
    rotated.set_positions(atoms.get_positions() @ r.T.numpy())

    base = cm(_sample(atoms))
    rot = cm(_sample(rotated))
    assert float((_energy(rot) - _energy(base)).abs().max()) < 1e-4
    assert float((_forces(rot) - _forces(base) @ r.T).abs().max()) < 1e-4


def test_routeb_sphere_compaction_near_exact(seed_fixture):
    """
    Route B: with the RMSNorm over-channel stats set to the kept width
    (norm_stats_num_channels), the compacted norm reproduces the pruned model's
    centering + normalization, removing the dominant sphere-compaction error (the
    RMSNorm-centering shift). The result is NEAR-EXACT -- far tighter than the naive
    sphere tolerance (5e-2) -- and round-trips as a uniform sphere_channels=K model.
    (A small residual ~1e-3 remains from the sphere-specific SO2/radial channel
    slicing, absorbed by the heal phase.)
    """
    kept = 12  # sphere_channels 16 -> 12
    cfg = _routeb_cfg(kept)
    model = hydra.utils.instantiate(cfg)
    model.eval()

    imp = CP.channel_importance(model, "sphere")
    drop = torch.argsort(imp)[: 16 - kept]
    CP.apply_channel_mask(model, "sphere", drop)
    ref = model(_sample())

    cm, rep = compact_channels.compact(model, cfg, "sphere")
    cm.eval()
    assert rep["kept"] == kept
    assert cm.backbone.sphere_channels == kept  # uniform-width, standard-loadable
    out = cm(_sample())
    # near-exact: far tighter than the naive sphere APPROX tolerance (_TOL 5e-2)
    assert float((_forces(out) - _forces(ref)).abs().max()) < 2e-3
    assert float((_energy(out) - _energy(ref)).abs().max()) < 1e-2


def test_norm_stats_default_is_behavior_preserving(seed_fixture):
    """The stats_num_channels override defaults to num_channels -> identical output."""
    from fairchem.core.models.uma.nn.layer_norm import (
        EquivariantRMSNormArraySphericalHarmonicsV2,
    )

    torch.manual_seed(0)
    norm = EquivariantRMSNormArraySphericalHarmonicsV2(lmax=2, num_channels=16).eval()
    x = torch.randn(5, 9, 16)
    assert norm.stats_num_channels == 16
    out = norm(x)
    # explicit stats=num_channels must match the default
    norm2 = EquivariantRMSNormArraySphericalHarmonicsV2(
        lmax=2, num_channels=16, stats_num_channels=16
    ).eval()
    norm2.load_state_dict(norm.state_dict())
    assert torch.allclose(out, norm2(x), atol=1e-6)


def test_cubic_schedule():
    vals = [CP.cubic_target(s, 10, 60, 0.5) for s in range(0, 65, 5)]
    assert vals[0] == 0.0
    assert CP.cubic_target(5, 10, 60, 0.5) == 0.0  # inside warmup
    assert abs(CP.cubic_target(60, 10, 60, 0.5) - 0.5) < 1e-9  # reaches target
    assert abs(CP.cubic_target(200, 10, 60, 0.5) - 0.5) < 1e-9  # flat after
    assert all(b >= a - 1e-9 for a, b in zip(vals, vals[1:])), "must be monotonic"


@pytest.mark.parametrize("mode", ["sphere"])
def test_callback_prune_then_heal_freeze(model_and_cfg, mode):
    model, _ = model_and_cfg
    cb = CP.ChannelPruningCallback(
        mode, target_sparsity=0.5, warmup_steps=10, healing_start_step=60
    )
    cb._resolved = True
    unit = SimpleNamespace(
        model=model,
        ema_model=None,
        logger=None,
        train_progress=SimpleNamespace(num_steps_completed=5),
    )
    # warmup: nothing dropped
    cb.on_train_step_end(None, unit)
    assert cb.drop_channels is None

    # prune phase: drops ramp toward target
    unit.train_progress.num_steps_completed = 40
    cb.on_train_step_end(None, unit)
    assert cb.drop_channels is not None
    n_at_40 = cb.drop_channels.numel()
    assert 0 < n_at_40 <= 8

    # heal phase: set frozen
    unit.train_progress.num_steps_completed = 62
    cb.on_train_step_end(None, unit)
    frozen = cb.drop_channels.clone()
    unit.train_progress.num_steps_completed = 90
    cb.on_train_step_end(None, unit)
    assert torch.equal(cb.drop_channels, frozen), "heal must freeze the dropped set"
    # frozen channels stay exactly zero on the live model
    imp = CP.channel_importance(model, mode)
    assert float(imp[frozen].abs().max()) == 0.0


@pytest.mark.parametrize("mode", ["sphere"])
def test_chanprune_smoke_train_cpu(fake_uma_dataset, mode):
    """End-to-end: the callback prunes/heals inside the CPU training loop."""
    launch_main(
        [
            "--config",
            "tests/core/units/mlip_unit/test_mlip_train_chanprune.yaml",
            "datasets=aselmdb_conserving",
            f"datasets.data_root_dir={fake_uma_dataset}",
            f"channel_prune_mode={mode}",
            "runner.max_steps=4",
        ]
    )
