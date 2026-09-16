"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

Tests:  The umas_flash execution backend, in four groups. Guards: every model
        shape and inference setting the fused path assumes must be rejected
        with a message rather than silently mis-executed. Parity: node
        embeddings, energies and forces against the general backend, plus the
        fused radial stage-1 kernel against its torch spelling. Graph
        parallelism: the partitioned scatter kernels directly on one GPU, and
        the full two-rank path on two. Device selection, since Triton launches
        on the ambient current device rather than the tensors' own.
Models: a small locally constructed eSCNMDBackbone (16 sphere channels) for
        everything except test_flash_forces_match_general_e2e, which uses the
        conserving_mole_checkpoint fixture.
CI:     test_gpu_sweep (units shard). Three tests need a second GPU and skip
        otherwise: both graph-parallel cases and the cross-device launch test.
"""

from __future__ import annotations

import os

import pytest
import torch
from ase.build import bulk, molecule

from fairchem.core.common import gp_utils
from fairchem.core.common.test_utils import (
    PGConfig,
    init_pg_and_rank_and_launch_test,
    spawn_multi_process,
)
from fairchem.core.datasets.ase_datasets import AseDBDataset
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.datasets.collaters.simple_collater import data_list_collater

# Registers the fairchem::flash_* ops, which the backend imports lazily.
from fairchem.core.models.uma import flash  # noqa: F401
from fairchem.core.models.uma.escn_md import eSCNMDBackbone
from fairchem.core.models.uma.nn.execution_backends import UMASFlashBackend
from fairchem.core.units.mlip_unit import MLIPPredictUnit
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

SPHERE_CHANNELS = 16
CUTOFF = 6.0


def _settings(**overrides) -> InferenceSettings:
    kwargs = {
        "merge_mole": True,
        "activation_checkpointing": False,
        "external_graph_gen": False,
        "execution_mode": "umas_flash",
    }
    kwargs.update(overrides)
    return InferenceSettings(**kwargs)


def _make_backbone(
    execution_mode: str,
    device: str = "cuda",
    seed: int = 42,
    external_graph: bool = False,
):
    """
    Build a small backbone with reproducible weights, prepared for inference.

    Both backends are constructed from the same seed so their weights match
    bit for bit and any output difference comes from the execution path.
    """
    torch.manual_seed(seed)
    backbone = eSCNMDBackbone(
        max_num_elements=100,
        sphere_channels=SPHERE_CHANNELS,
        hidden_channels=SPHERE_CHANNELS,
        lmax=2,
        mmax=2,
        num_layers=2,
        otf_graph=not external_graph,
        edge_channels=SPHERE_CHANNELS,
        num_distance_basis=32,
        cutoff=CUTOFF,
        use_dataset_embedding=False,
        always_use_pbc=False,
        execution_mode=execution_mode,
    ).to(device)
    backbone.eval()
    return backbone.prepare_for_inference(
        None, _settings(execution_mode=execution_mode)
    )


def _make_batch(atoms, device: str = "cuda", external_graph: bool = False):
    sample = AtomicData.from_ase(
        input_atoms=atoms,
        max_neigh=20,
        radius=CUTOFF,
        task_name="test",
        r_edges=external_graph,
        r_data_keys=["spin", "charge"],
    )
    return data_list_collater([sample], otf_graph=not external_graph).to(device)


def _water():
    atoms = molecule("H2O")
    atoms.info.update({"charge": 0, "spin": 1})
    atoms.set_cell([12.0, 12.0, 12.0])
    atoms.center()
    atoms.pbc = [False, False, False]
    return atoms


# =============================================================================
# Validation
# =============================================================================


@pytest.mark.gpu()
def test_flash_validation_requires_lmax_mmax_2():
    with pytest.raises(ValueError, match="lmax==2 and mmax==2"):
        UMASFlashBackend.validate(lmax=3, mmax=2, settings=_settings())
    with pytest.raises(ValueError, match="lmax==2 and mmax==2"):
        UMASFlashBackend.validate(lmax=2, mmax=1, settings=_settings())


@pytest.mark.gpu()
def test_flash_validation_requires_merge_mole():
    with pytest.raises(ValueError, match="merge_mole=True"):
        UMASFlashBackend.validate(lmax=2, mmax=2, settings=_settings(merge_mole=False))


@pytest.mark.gpu()
def test_flash_validation_accepts_activation_checkpointing():
    """
    Honoured, but by wrapping each fused layer rather than chunking edges the
    way the eager backbone does -- see test_flash_checkpointing_matches.
    """
    UMASFlashBackend.validate(
        lmax=2, mmax=2, settings=_settings(activation_checkpointing=True)
    )


@pytest.mark.gpu()
def test_flash_validation_accepts_supported_config():
    UMASFlashBackend.validate(lmax=2, mmax=2, settings=_settings())


@pytest.mark.gpu()
def test_flash_validation_rejects_float64():
    """
    The kernels accumulate in fp32, so float64 would be quietly demoted.
    Refusing is better than returning float32 accuracy in a float64 tensor.
    """
    with pytest.raises(ValueError, match="float32 only"):
        UMASFlashBackend.validate(
            lmax=2, mmax=2, settings=_settings(base_precision_dtype=torch.float64)
        )


@pytest.mark.gpu()
def test_flash_validation_rejects_hessians():
    with pytest.raises(ValueError, match="hessians"):
        UMASFlashBackend.validate(
            lmax=2, mmax=2, settings=_settings(predict_untrained_hessian={"omol"})
        )


def _raw_backbone(device: str = "cuda", **overrides):
    """
    A backbone in eval mode that has NOT been through prepare_for_inference,
    so the model-level guards can be provoked one at a time.
    """
    torch.manual_seed(0)
    kwargs = {
        "max_num_elements": 100,
        "sphere_channels": SPHERE_CHANNELS,
        "hidden_channels": SPHERE_CHANNELS,
        "lmax": 2,
        "mmax": 2,
        "num_layers": 1,
        "otf_graph": True,
        "edge_channels": SPHERE_CHANNELS,
        "num_distance_basis": 32,
        "cutoff": CUTOFF,
        "use_dataset_embedding": False,
        "always_use_pbc": False,
        "execution_mode": "umas_flash",
    }
    kwargs.update(overrides)
    backbone = eSCNMDBackbone(**kwargs).to(device)
    backbone.eval()
    return backbone


@pytest.mark.gpu()
def test_flash_rejects_mismatched_hidden_channels():
    backbone = _raw_backbone(hidden_channels=2 * SPHERE_CHANNELS)
    with pytest.raises(ValueError, match="hidden_channels == sphere_channels"):
        backbone.prepare_for_inference(None, _settings())


@pytest.mark.gpu()
def test_flash_rejects_training_mode():
    """
    Inference only, and not merely untested: the kernels drop the activations
    a training backward needs, and their backward is not differentiable.
    """
    backbone = _raw_backbone()
    backbone.train()
    with pytest.raises(ValueError, match="inference-only"):
        backbone.prepare_for_inference(None, _settings())


@pytest.mark.gpu()
def test_flash_rejects_grid_ff_but_accepts_spectral():
    """
    ff_type is genuinely free -- the block's own atom_wise module is reused --
    so neither value may be rejected. This pins that down, since the guards
    around it are deliberately strict.
    """
    for ff_type in ("spectral", "grid"):
        backbone = _raw_backbone(ff_type=ff_type)
        backbone.prepare_for_inference(None, _settings())
        assert hasattr(backbone, "_flash_features")


@pytest.mark.gpu()
def test_flash_rejects_non_gate_activation():
    backbone = _raw_backbone(act_type="s2")
    with pytest.raises(ValueError, match="gate activation"):
        backbone.prepare_for_inference(None, _settings())


@pytest.mark.gpu()
def test_flash_rejects_float64_parameters():
    backbone = _raw_backbone().double()
    with pytest.raises(ValueError, match="float32 only"):
        backbone.prepare_for_inference(None, _settings())


@pytest.mark.gpu()
def test_flash_rejects_a_different_envelope():
    """
    The geometry kernel inlines the exponent-5 polynomial rather than calling
    model.envelope, so a changed exponent would be silently ignored.
    """
    from fairchem.core.models.uma.nn.radial import PolynomialEnvelope

    backbone = _raw_backbone()
    backbone.envelope = PolynomialEnvelope(exponent=6)
    with pytest.raises(ValueError, match="polynomial envelope"):
        backbone.prepare_for_inference(None, _settings())


@pytest.mark.gpu()
def test_flash_rejects_a_non_gaussian_radial_basis():
    backbone = _raw_backbone()
    backbone.distance_expansion = torch.nn.Identity()
    with pytest.raises(ValueError, match="Gaussian radial basis"):
        backbone.prepare_for_inference(None, _settings())


@pytest.mark.gpu()
def test_flash_rejects_a_cpu_model():
    """
    validate() only checks that a GPU exists, not that the model is on it.

    The check has to live in the forward: prepare_for_inference runs before
    the predict unit moves the model to its device, so a CPU model there is
    normal rather than an error.
    """
    backbone = _raw_backbone(device="cpu")
    backbone.prepare_for_inference(None, _settings())
    batch = _make_batch(_water(), device="cpu")
    with pytest.raises(ValueError, match="needs CUDA tensors"):
        backbone(batch)


@pytest.mark.gpu()
def test_flash_rejects_an_unexpected_radial_width():
    """
    The gather kernel reads twelve C-wide slices with unchecked pointer
    arithmetic, so a narrower radial output would read out of bounds.
    """
    backbone = _raw_backbone()
    net = backbone.blocks[0].edge_wise.so2_conv_1.rad_func.net
    net[-1] = torch.nn.Linear(net[-1].in_features, 4 * SPHERE_CHANNELS).cuda()
    with pytest.raises(ValueError, match="12 \\* sphere_channels"):
        backbone.prepare_for_inference(None, _settings())


@pytest.mark.gpu()
def test_flash_rejects_unmerged_mole_layers():
    """
    merge_mole is required by validate(), but a model that reached the
    repacking with MOLE layers still in place must say so rather than dying on
    a missing .weight attribute.
    """
    backbone = _raw_backbone()
    backbone.blocks[0].edge_wise.so2_conv_1.fc_m0 = torch.nn.Identity()
    with pytest.raises(ValueError, match="MOLE experts were not merged"):
        backbone.prepare_for_inference(None, _settings())


# =============================================================================
# Activation checkpointing
# =============================================================================


@pytest.mark.gpu()
def test_flash_checkpointing_matches_and_saves_memory():
    """
    Recomputing each layer must not change the answer beyond the backend's own
    atomic noise, and must hold less.

    The point of checkpointing is the second half, so this asserts on retained
    memory rather than just parity: without the size check a no-op wrapper
    would pass.
    """
    import gc

    # Rattled: a pristine lattice sits at a symmetry point where every force
    # vanishes, so the gradient comparison below would be noise against noise.
    atoms = bulk("Cu", "fcc", a=3.6, cubic=True).repeat((2, 2, 2))
    atoms.rattle(stdev=0.05, seed=0)
    atoms.info.update({"charge": 0, "spin": 0})

    results, retained = {}, {}
    for flag in (False, True):
        backbone = _make_backbone("umas_flash")
        backbone._flash_features.checkpointing = flag
        batch = _make_batch(atoms)
        batch["pos"].requires_grad_(True)

        gc.collect()
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        emb = backbone(batch)["node_embedding"]
        torch.cuda.synchronize()
        retained[flag] = torch.cuda.memory_allocated() - base
        (grad,) = torch.autograd.grad(emb.square().sum(), batch["pos"])
        results[flag] = (emb.detach().clone(), grad.clone())

    emb_off, grad_off = results[False]
    emb_on, grad_on = results[True]
    # Not bit equality: the scatter accumulates with fp32 atomics, so two runs
    # of the same code already differ by ~1e-6 relative. Checkpointing must not
    # move the answer by more than that noise floor.
    # atol + rtol * scale, not a pure ratio: a quantity that is near zero has
    # a reference of pure rounding, and dividing by it turns 1e-7 into 40%.
    for got, want, name in (
        (emb_on, emb_off, "node embeddings"),
        (grad_on, grad_off, "position gradients"),
    ):
        delta = (got - want).abs().max()
        assert delta < 1e-6 + 1e-5 * want.abs().max(), f"{name} differ by {delta}"

    assert retained[True] < retained[False], (
        f"checkpointing retained {retained[True] / 2**20:.1f} MB, "
        f"no better than {retained[False] / 2**20:.1f} MB without it"
    )


# =============================================================================
# Fused radial stage 1
# =============================================================================


def _stage1_inputs(E, B, H, device="cuda", seed=0):
    gen = torch.Generator(device=device).manual_seed(seed)
    rand = lambda *shape: torch.randn(*shape, device=device, generator=gen)  # noqa: E731
    return {
        "gauss": rand(E, B).requires_grad_(True),
        "W_gauss": rand(H, B) * 0.1,
        "bias": rand(H),
        "table_src": rand(100, H),
        "table_tgt": rand(100, H),
        "z_i": torch.randint(0, 100, (E,), device=device, generator=gen),
        "z_j": torch.randint(0, 100, (E,), device=device, generator=gen),
        "gamma": rand(H),
        "beta": rand(H),
        "eps": 1e-5,
    }


def _stage1_eager(a):
    """The torch spelling the kernel replaces."""
    import torch.nn.functional as F

    from fairchem.core.models.uma.flash.custom_ops import layer_norm_silu

    h = F.linear(a["gauss"], a["W_gauss"], a["bias"])
    h = h + a["table_src"][a["z_i"]] + a["table_tgt"][a["z_j"]]
    return layer_norm_silu(h, a["gamma"], a["beta"], a["eps"])


@pytest.mark.gpu()
@pytest.mark.parametrize(("B", "H"), [(32, 128), (24, 96)])
def test_radial_stage1_matches_eager(B, H):
    """
    Forward and backward against the torch chain, at a power-of-two shape and
    a padded one -- tl.dot needs powers of two, so the padded case exercises
    the masking that keeps the contraction exact.
    """
    from fairchem.core.models.uma.flash.custom_ops import radial_stage1

    E = 4096
    args = _stage1_inputs(E, B, H)
    eager_args = dict(args)
    eager_args["gauss"] = args["gauss"].detach().clone().requires_grad_(True)

    reference = _stage1_eager(eager_args)
    fused = radial_stage1(
        args["gauss"],
        args["W_gauss"],
        args["bias"],
        args["table_src"],
        args["table_tgt"],
        args["z_i"],
        args["z_j"],
        args["gamma"],
        args["beta"],
        args["eps"],
    )
    assert fused.shape == (E, H)

    scale = reference.abs().max()
    assert (fused - reference).abs().max() < 1e-5 * scale

    cotangent = torch.randn_like(reference)
    (g_ref,) = torch.autograd.grad(reference, eager_args["gauss"], cotangent)
    (g_fused,) = torch.autograd.grad(fused, args["gauss"], cotangent)
    g_scale = g_ref.abs().max()
    assert (g_fused - g_ref).abs().max() < 1e-5 * g_scale


@pytest.mark.gpu()
def test_radial_stage1_accepts_empty_edge_set():
    from fairchem.core.models.uma.flash.custom_ops import radial_stage1

    args = _stage1_inputs(0, 32, 128)
    out = radial_stage1(*[args[k] for k in _STAGE1_ORDER])
    assert out.shape == (0, 128)


_STAGE1_ORDER = (
    "gauss",
    "W_gauss",
    "bias",
    "table_src",
    "table_tgt",
    "z_i",
    "z_j",
    "gamma",
    "beta",
    "eps",
)


@pytest.mark.gpu()
def test_radial_stage1_ignores_tf32():
    """
    The one tl.dot in the backend is pinned to ieee, so unlike the cuBLAS
    GEMMs around it this kernel must give bit-identical results either way.
    Without the pin it would quietly lose mantissa bits whenever some other
    caller had set a global matmul precision.
    """
    from fairchem.core.models.uma.flash.custom_ops import radial_stage1

    args = _stage1_inputs(2048, 32, 128)
    call = lambda: radial_stage1(*[args[k] for k in _STAGE1_ORDER])  # noqa: E731

    original = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        exact = call().detach().clone()
        torch.set_float32_matmul_precision("high")
        reduced = call().detach().clone()
    finally:
        torch.set_float32_matmul_precision(original)
    assert torch.equal(exact, reduced)


# =============================================================================
# Precision
# =============================================================================


@pytest.mark.gpu()
@pytest.mark.skipif(
    torch.cuda.get_device_capability()[0] < 8, reason="tf32 needs sm_80 or newer"
)
def test_flash_honours_the_tf32_setting():
    """
    The kernels hold no tf32 switch of their own. The only tl.dot in the
    backend is the radial stage-1 contraction, and it is pinned to ieee. What
    tf32 reaches is the cuBLAS GEMMs: the packed SO(2) blocks and the radial
    MLP's second and third linears, all plain F.linear.

    So the property to pin is that the setting arrives at all. Under "high"
    the result must move (otherwise tf32 is being ignored) and must stay close
    (otherwise something other than mantissa truncation changed).
    """
    backbone = _make_backbone("umas_flash")
    batch = _make_batch(_water())

    original = torch.get_float32_matmul_precision()
    try:
        torch.set_float32_matmul_precision("highest")
        exact = backbone(batch)["node_embedding"].detach().clone()
        torch.set_float32_matmul_precision("high")
        reduced = backbone(batch)["node_embedding"].detach().clone()
    finally:
        torch.set_float32_matmul_precision(original)

    delta = (exact - reduced).abs().max()
    scale = exact.abs().max()
    assert delta > 0, "tf32 did not reach the flash GEMMs"
    assert delta < 1e-3 * scale, f"tf32 moved the result by {delta / scale:.2e}"


# =============================================================================
# Parity with the general backend
# =============================================================================


@pytest.mark.gpu()
@pytest.mark.parametrize(
    "atoms_fn", [_water, lambda: bulk("Cu", "fcc", a=3.6, cubic=True)]
)
def test_flash_node_embeddings_and_grads_match_general(atoms_fn):
    """
    Same weights, same input: the fused path must reproduce the eager one,
    including the gradient w.r.t. positions that forces are built from.
    """
    atoms = atoms_fn()
    if "charge" not in atoms.info:
        atoms.info.update({"charge": 0, "spin": 0})

    outputs = {}
    for mode in ("general", "umas_flash"):
        backbone = _make_backbone(mode)
        batch = _make_batch(atoms)
        batch["pos"].requires_grad_(True)
        emb = backbone(batch)["node_embedding"]
        (grad,) = torch.autograd.grad(emb.square().sum(), batch["pos"])
        outputs[mode] = (emb.detach(), grad.detach())
        del backbone

    emb_ref, grad_ref = outputs["general"]
    emb_flash, grad_flash = outputs["umas_flash"]
    assert emb_flash.shape == emb_ref.shape
    # atol + rtol rather than either alone: weights are random and untrained so
    # activations are O(1..10) and fp32 re-association is visible in absolute
    # terms, while a symmetric crystal has a gradient that vanishes by symmetry
    # and would make a pure relative bound divide noise by noise.
    assert torch.allclose(
        emb_flash, emb_ref, rtol=0, atol=1e-5 + 5e-4 * emb_ref.abs().max()
    ), f"node embedding mismatch: {(emb_flash - emb_ref).abs().max()}"
    assert torch.allclose(
        grad_flash, grad_ref, rtol=0, atol=1e-5 + 1e-3 * grad_ref.abs().max()
    ), f"position gradient mismatch: {(grad_flash - grad_ref).abs().max()}"


@pytest.mark.gpu()
def test_flash_forces_match_general_e2e(conserving_mole_checkpoint, fake_uma_dataset):
    """
    End to end through a predict unit, so MOLE merging and the heads are in
    the loop as well.
    """
    checkpoint_pt, _ = conserving_mole_checkpoint
    db = AseDBDataset(config={"src": os.path.join(fake_uma_dataset, "oc20")})
    sample = AtomicData.from_ase(
        db.get_atoms(0),
        max_neigh=10,
        radius=100,
        r_energy=False,
        r_forces=False,
        r_edges=False,
        r_data_keys=["spin", "charge"],
    )
    sample["dataset"] = "oc20"
    batch = data_list_collater([sample], otf_graph=True)

    baseline = MLIPPredictUnit(
        checkpoint_pt, "cuda", inference_settings=_settings(execution_mode="general")
    ).predict(batch.clone())
    fused = MLIPPredictUnit(
        checkpoint_pt, "cuda", inference_settings=_settings()
    ).predict(batch.clone())

    assert torch.allclose(
        baseline["energy"], fused["energy"], rtol=5e-4, atol=5e-5
    ), f"energy mismatch: {baseline['energy']} vs {fused['energy']}"
    assert torch.allclose(
        baseline["forces"], fused["forces"], rtol=5e-4, atol=5e-5
    ), f"force mismatch: {(baseline['forces'] - fused['forces']).abs().max()}"
    # Stress exercises the cell gradient, which reaches the kernels only
    # through edge_distance_vec.
    assert "stress" in baseline, "expected the fixture to regress stress"
    assert torch.allclose(
        baseline["stress"], fused["stress"], rtol=5e-4, atol=5e-5
    ), f"stress mismatch: {(baseline['stress'] - fused['stress']).abs().max()}"


# =============================================================================
# Graph parallel contract, on a single GPU
# =============================================================================


def _random_scatter_inputs(num_nodes, num_edges, channels, device, seed=0):
    gen = torch.Generator(device=device).manual_seed(seed)
    rand = lambda *shape: torch.randn(  # noqa: E731
        *shape, device=device, generator=gen, requires_grad=True
    )
    return {
        "Z_m0": rand(num_edges, 3 * channels),
        "Z_m1": rand(num_edges, 4 * channels),
        "Z_m2": rand(num_edges, 2 * channels),
        "wigner": rand(34, num_edges),
        "env": rand(num_edges),
        "scatter_target": torch.randint(
            0, num_nodes, (num_edges,), device=device, generator=gen
        ),
    }


@pytest.mark.gpu()
def test_scatter_partitioned_matches_full_graph():
    """
    A rank owning a slice of the nodes, and the edges targeting them, must
    produce exactly the rows the unpartitioned scatter produces, forward and
    backward. This is the whole partition contract, with no collectives.

    The local targets are built the way upstream's _generate_graph builds
    ``scatter_target``: global target minus the partition start, which for a
    contiguous slice is what its global_to_local table amounts to.
    """
    device, C, N, E = "cuda", 8, 12, 60
    args = _random_scatter_inputs(N, E, C, device)
    x_res = torch.zeros(N, 9, C, device=device)

    full = torch.ops.fairchem.flash_scatter_fwd(
        args["Z_m0"],
        args["Z_m1"],
        args["Z_m2"],
        args["wigner"],
        args["env"],
        x_res,
        args["scatter_target"],
    )
    (g_full,) = torch.autograd.grad(
        full.square().sum(), args["Z_m0"], retain_graph=True
    )

    partitions = [(0, N // 2), (N // 2, N)]
    rows, grads = [], torch.zeros_like(g_full)
    for start, end in partitions:
        mask = (args["scatter_target"] >= start) & (args["scatter_target"] < end)
        idx = mask.nonzero(as_tuple=True)[0]
        local = torch.ops.fairchem.flash_scatter_fwd(
            args["Z_m0"][idx],
            args["Z_m1"][idx],
            args["Z_m2"][idx],
            args["wigner"][:, idx].contiguous(),
            args["env"][idx],
            x_res[start:end],
            args["scatter_target"][idx] - start,
        )
        rows.append(local)
        (g_local,) = torch.autograd.grad(
            (local * full[start:end].detach() * 2).sum(),
            args["Z_m0"],
            retain_graph=True,
        )
        grads += g_local

    stitched = torch.cat(rows, dim=0)
    assert (
        (stitched - full).abs().max() < 1e-5
    ), f"partitioned scatter differs: {(stitched - full).abs().max()}"
    assert (
        (grads - g_full).abs().max() < 1e-4
    ), f"partitioned scatter gradient differs: {(grads - g_full).abs().max()}"


@pytest.mark.gpu()
def test_init_scatter_partitioned_matches_full_graph():
    device, C, N, E, num_systems = "cuda", 8, 12, 60, 1
    gen = torch.Generator(device=device).manual_seed(1)
    rad_out = torch.randn(E, 3 * C, device=device, generator=gen, requires_grad=True)
    wigner = torch.randn(34, E, device=device, generator=gen)
    env = torch.randn(E, device=device, generator=gen)
    idx_j = torch.randint(0, N, (E,), device=device, generator=gen)
    Z = torch.randint(1, 90, (N,), device=device, generator=gen)
    batch = torch.zeros(N, dtype=torch.long, device=device)
    W_sphere = torch.randn(100, C, device=device, generator=gen)
    csd = torch.randn(num_systems, C, device=device, generator=gen)

    full = torch.ops.fairchem.flash_init_scatter_fwd(
        rad_out, wigner, env, Z, batch, W_sphere, csd, idx_j, N, 0.2
    )

    rows = []
    for start, end in [(0, N // 2), (N // 2, N)]:
        mask = (idx_j >= start) & (idx_j < end)
        idx = mask.nonzero(as_tuple=True)[0]
        rows.append(
            torch.ops.fairchem.flash_init_scatter_fwd(
                rad_out[idx],
                wigner[:, idx].contiguous(),
                env[idx],
                Z[start:end],
                batch[start:end],
                W_sphere,
                csd,
                idx_j[idx] - start,
                end - start,
                0.2,
            )
        )
    stitched = torch.cat(rows, dim=0)
    assert (
        (stitched - full).abs().max() < 1e-5
    ), f"partitioned edge-degree scatter differs: {(stitched - full).abs().max()}"


@pytest.mark.gpu()
def test_flash_ops_accept_empty_edge_set():
    """
    A graph parallel rank can end up owning no edges; autotuned kernels cannot
    be benchmarked on an empty grid, so the ops must short-circuit.
    """
    device, C, N = "cuda", 8, 4
    empty = lambda *shape: torch.zeros(*shape, device=device)  # noqa: E731
    idx_j = torch.zeros(0, dtype=torch.long, device=device)

    out = torch.ops.fairchem.flash_scatter_fwd(
        empty(0, 3 * C),
        empty(0, 4 * C),
        empty(0, 2 * C),
        empty(34, 0),
        empty(0),
        empty(N, 9, C),
        idx_j,
    )
    assert out.shape == (N, 9, C)
    assert torch.count_nonzero(out) == 0

    gauss, wigner, env = torch.ops.fairchem.flash_geom_fwd(
        empty(0, 3), torch.linspace(0, CUTOFF, 32, device=device), -0.5, CUTOFF
    )
    assert gauss.shape == (0, 32)
    assert wigner.shape == (34, 0)
    assert env.shape == (0,)

    init = torch.ops.fairchem.flash_init_scatter_fwd(
        empty(0, 3 * C),
        empty(34, 0),
        empty(0),
        torch.ones(N, dtype=torch.long, device=device),
        torch.zeros(N, dtype=torch.long, device=device),
        empty(100, C),
        empty(1, C),
        idx_j,
        N,
        0.2,
    )
    assert init.shape == (N, 9, C)


def _flash_gp_rank_output(external_graph: bool = False):
    """
    Run the flash backbone on one graph parallel rank and hand back the local
    node embeddings plus this rank's contribution to dE/dpos.
    """
    device = f"cuda:{gp_utils.get_gp_rank()}" if gp_utils.initialized() else "cuda:0"
    torch.cuda.set_device(device)
    backbone = _make_backbone(
        "umas_flash", device=device, external_graph=external_graph
    )
    batch = _make_batch(_water(), device=device, external_graph=external_graph)
    batch["pos"].requires_grad_(True)
    emb = backbone(batch)["node_embedding"]
    (grad,) = torch.autograd.grad(emb.square().sum(), batch["pos"])
    return emb.detach().cpu(), grad.detach().cpu()


@pytest.mark.gpu()
@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="needs two GPUs: see the note below"
)
@pytest.mark.parametrize(
    "external_graph",
    [
        False,
        pytest.param(
            True,
            marks=pytest.mark.skip(
                reason="externally supplied graphs are not filtered by node "
                "partition under GP -- an upstream gap in _generate_graph, not "
                "a flash one; see docs/gp-external-graph-filtering.md"
            ),
        ),
    ],
)
def test_flash_graph_parallel_matches_single_rank(external_graph):
    """
    Concatenating the per-rank node embeddings must reproduce the unpartitioned
    result, and the per-rank position gradients must sum to it.

    Two real GPUs and nccl, matching how upstream tests the CUDA graph-parallel
    path. This used to run as two gloo ranks sharing one device, which is
    cheaper, but gp_utils now issues its collectives through
    torch.distributed._functional_collectives, and funcol's all_gather
    segfaults on CUDA tensors under gloo -- reproducible in six lines with no
    fairchem involved. Plain dist.all_gather is unaffected.

    The external_graph case is skipped: with otf_graph=False, _generate_graph
    consumes data_dict["edge_index"] unfiltered, so under GP every rank sees
    every edge. That affects all backends, not just this one -- see the report.
    """
    reference_emb, reference_grad = _flash_gp_rank_output(external_graph)

    config = PGConfig(backend="nccl", world_size=2, gp_group_size=2, use_gp=True)
    per_rank = spawn_multi_process(
        config,
        _flash_gp_rank_output,
        init_pg_and_rank_and_launch_test,
        external_graph,
    )

    stitched = torch.cat([emb for emb, _ in per_rank], dim=0)
    assert stitched.shape == reference_emb.shape
    assert (stitched - reference_emb).abs().max() < 1e-4, (
        f"graph parallel node embeddings differ: "
        f"{(stitched - reference_emb).abs().max()}"
    )

    summed_grad = sum(grad for _, grad in per_rank)
    assert (summed_grad - reference_grad).abs().max() < 1e-3, (
        f"graph parallel position gradients differ: "
        f"{(summed_grad - reference_grad).abs().max()}"
    )


# =============================================================================
# Device selection
# =============================================================================


@pytest.mark.gpu()
def test_device_guard_is_free_on_the_current_device():
    """
    The guard must be a no-op when the tensor already lives on the current
    device, which is the single-GPU path.
    """
    import contextlib

    from fairchem.core.models.uma.flash.custom_ops import _device_of

    current = torch.empty(1, device=f"cuda:{torch.cuda.current_device()}")
    assert isinstance(_device_of(current), contextlib.nullcontext)
    assert isinstance(_device_of(torch.empty(1)), contextlib.nullcontext)


@pytest.mark.gpu()
@pytest.mark.skipif(
    torch.cuda.device_count() < 2, reason="needs a second GPU to be meaningful"
)
def test_flash_ops_run_on_non_current_device():
    """
    Triton resolves the launch device from the ambient current device, so an
    op called with tensors on another device must set it first.
    """
    device, C, N, E = "cuda:1", 8, 12, 40
    torch.cuda.set_device(0)
    args = _random_scatter_inputs(N, E, C, device)
    out = torch.ops.fairchem.flash_scatter_fwd(
        args["Z_m0"],
        args["Z_m1"],
        args["Z_m2"],
        args["wigner"],
        args["env"],
        torch.zeros(N, 9, C, device=device),
        args["scatter_target"],
    )
    assert out.device.index == 1
    assert torch.isfinite(out).all()
    assert torch.cuda.current_device() == 0, "the guard must restore the device"
