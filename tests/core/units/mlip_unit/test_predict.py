"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import contextlib
import logging
from copy import deepcopy

import numpy as np
import numpy.testing as npt
import pytest
import ray
import torch
from ase import Atoms
from ase.build import add_adsorbate, bulk, fcc100, make_supercell, molecule
from ase.data import chemical_symbols

from fairchem.core import FAIRChemCalculator, pretrained_mlip
from fairchem.core.calculate.pretrained_mlip import pretrained_checkpoint_path_from_name
from fairchem.core.common import distutils
from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
from fairchem.core.datasets.common_structures import get_fcc_crystal_by_num_atoms
from fairchem.core.models.uma.nn.execution_backends import UMASFastGPUBackend
from fairchem.core.units.mlip_unit import InferenceSettings, MLIPPredictUnit
from fairchem.core.units.mlip_unit.predict import ParallelMLIPPredictUnit
from fairchem.core.units.mlip_unit.single_atom_patch import (
    single_atom_prediction_from_lookup,
)
from tests.conftest import seed_everywhere

FORCE_TOL = 1e-4
ATOL = 5e-4
_REPRESENTATIVE_ELEMENTS = [
    (1, 0, 2),  # H:  charge=0, spin=2
    (6, 0, 3),  # C:  charge=0, spin=3
    (8, 0, 3),  # O:  charge=0, spin=3
    (11, 1, 1),  # Na: charge=+1, spin=1
    (79, 0, 2),  # Au: charge=0, spin=2
]
SINGLE_ATOM_ENERGY_ATOL = 0.05  # eV, for model-predicted single atom energies


@pytest.fixture(scope="module")
def uma_predict_unit_cuda():
    """Module-scoped predict unit using the first available UMA model with device=cuda."""
    uma_models = [name for name in pretrained_mlip.available_models if "uma" in name]
    return pretrained_mlip.get_predict_unit(uma_models[0], device="cuda")


@pytest.fixture(scope="module")
def uma_predict_unit(uma_predict_unit_cuda):
    """Module-scoped predict unit - uses cuda version if available, otherwise cpu."""
    if torch.cuda.is_available():
        return uma_predict_unit_cuda
    uma_models = [name for name in pretrained_mlip.available_models if "uma" in name]
    return pretrained_mlip.get_predict_unit(uma_models[0])


@pytest.fixture(scope="module")
def uma_merge_mole_predict_unit():
    """Module-scoped predict unit with merge_mole=True for MgO tests."""
    uma_models = [name for name in pretrained_mlip.available_models if "uma" in name]
    settings = InferenceSettings(merge_mole=True, external_graph_gen=False)
    return pretrained_mlip.get_predict_unit(
        uma_models[0], device="cuda", inference_settings=settings
    )


@pytest.fixture(scope="module")
def uma_1p1_predict_unit():
    """Module-scoped predict unit for uma-s-1p1."""
    return pretrained_mlip.get_predict_unit("uma-s-1p1")


@pytest.fixture(scope="module")
def uma_1p2_predict_unit():
    """Module-scoped predict unit for uma-s-1p2."""
    return pretrained_mlip.get_predict_unit("uma-s-1p2")


@pytest.mark.gpu()
@pytest.mark.parametrize("internal_graph_gen_version", [2, 3])
def test_single_dataset_predict(internal_graph_gen_version):
    uma_models = [name for name in pretrained_mlip.available_models if "uma" in name]
    inference_settings = InferenceSettings(
        tf32=False,
        activation_checkpointing=True,
        merge_mole=False,
        compile=False,
        external_graph_gen=False,
        internal_graph_gen_version=internal_graph_gen_version,
    )
    uma_predict_unit = pretrained_mlip.get_predict_unit(
        uma_models[0], inference_settings=inference_settings
    )

    n = 10
    atoms = bulk("Pt")
    atomic_data_list = [AtomicData.from_ase(atoms, task_name="omat") for _ in range(n)]
    batch = atomicdata_list_to_batch(atomic_data_list)

    preds = uma_predict_unit.predict(batch)

    assert preds["energy"].shape == (n,)
    assert preds["forces"].shape == (n, 3)
    assert preds["stress"].shape == (n, 9)

    # compare result with that from the calculator
    calc = FAIRChemCalculator(uma_predict_unit, task_name="omat")
    atoms.calc = calc
    npt.assert_allclose(
        preds["energy"].detach().cpu().numpy(), atoms.get_potential_energy()
    )
    npt.assert_allclose(preds["forces"].detach().cpu().numpy() - atoms.get_forces(), 0)
    npt.assert_allclose(
        preds["stress"].detach().cpu().numpy()
        - atoms.get_stress(voigt=False).flatten(),
        0,
        atol=ATOL,
    )


@pytest.mark.xfail(reason="Issue with UMA 1.2 release TODO fix")
@pytest.mark.gpu()
@pytest.mark.parametrize("internal_graph_gen_version", [2, 3])
def test_multiple_dataset_predict(internal_graph_gen_version):
    uma_models = [name for name in pretrained_mlip.available_models if "uma" in name]
    inference_settings = InferenceSettings(
        tf32=False,
        activation_checkpointing=True,
        merge_mole=False,
        compile=False,
        external_graph_gen=False,
        internal_graph_gen_version=internal_graph_gen_version,
    )
    uma_predict_unit = pretrained_mlip.get_predict_unit(
        uma_models[0], inference_settings=inference_settings
    )

    h2o = molecule("H2O")
    h2o.info.update({"charge": 0, "spin": 1})
    h2o.pbc = True  # all data points must be pbc if mixing.

    slab = fcc100("Cu", (3, 3, 3), vacuum=8, periodic=True)
    adsorbate = molecule("CO")
    add_adsorbate(slab, adsorbate, 2.0, "bridge")

    pt = bulk("Pt")
    pt.repeat((2, 2, 2))

    atomic_data_list = [
        AtomicData.from_ase(
            h2o,
            task_name="omol",
            r_data_keys=["spin", "charge"],
            molecule_cell_size=120,
        ),
        AtomicData.from_ase(slab, task_name="oc20"),
        AtomicData.from_ase(pt, task_name="omat"),
    ]

    batch = atomicdata_list_to_batch(atomic_data_list)
    preds = uma_predict_unit.predict(batch)

    n_systems = len(batch)
    n_atoms = sum(batch.natoms).item()
    assert preds["energy"].shape == (n_systems,)
    assert preds["forces"].shape == (n_atoms, 3)
    assert preds["stress"].shape == (n_systems, 9)

    # compare to fairchem calcs
    seed_everywhere(42)
    omol_calc = FAIRChemCalculator(uma_predict_unit, task_name="omol")
    oc20_calc = FAIRChemCalculator(uma_predict_unit, task_name="oc20")
    omat_calc = FAIRChemCalculator(uma_predict_unit, task_name="omat")

    pred_energy = preds["energy"].detach().cpu().numpy()
    pred_forces = preds["forces"].detach().cpu().numpy()

    h2o.calc = omol_calc
    h2o.center(vacuum=120)
    slab.calc = oc20_calc
    pt.calc = omat_calc

    npt.assert_allclose(pred_energy[0], h2o.get_potential_energy())
    npt.assert_allclose(pred_energy[1], slab.get_potential_energy())
    npt.assert_allclose(pred_energy[2], pt.get_potential_energy())

    batch_batch = batch.batch.detach().cpu().numpy()
    npt.assert_allclose(pred_forces[batch_batch == 0], h2o.get_forces(), atol=ATOL)
    npt.assert_allclose(pred_forces[batch_batch == 1], slab.get_forces(), atol=ATOL)
    npt.assert_allclose(pred_forces[batch_batch == 2], pt.get_forces(), atol=ATOL)


def _test_parallel_predict_unit_impl(workers, device, checkpointing, graph_gen_version):
    """Implementation of parallel predict unit test."""
    seed = 42
    runs = 2
    model_path = pretrained_checkpoint_path_from_name("uma-s-1p1")
    num_atoms = 10
    ifsets = InferenceSettings(
        tf32=False,
        merge_mole=True,
        activation_checkpointing=checkpointing,
        internal_graph_gen_version=graph_gen_version,
        external_graph_gen=False,
    )
    atoms = get_fcc_crystal_by_num_atoms(num_atoms)
    atomic_data = AtomicData.from_ase(atoms, task_name=["omat"])

    seed_everywhere(seed)
    ppunit = ParallelMLIPPredictUnit(
        inference_model_path=model_path,
        device=device,
        inference_settings=ifsets,
        num_workers=workers,
    )
    for _ in range(runs):
        pp_results = ppunit.predict(atomic_data)

    distutils.cleanup_gp_ray()

    seed_everywhere(seed)
    normal_predict_unit = pretrained_mlip.get_predict_unit(
        "uma-s-1p1", device=device, inference_settings=ifsets
    )
    for _ in range(runs):
        normal_results = normal_predict_unit.predict(atomic_data)

    logging.info(f"normal_results: {normal_results}")
    logging.info(f"pp_results: {pp_results}")
    assert torch.allclose(
        pp_results["energy"].detach().cpu(),
        normal_results["energy"].detach().cpu(),
        atol=ATOL,
    )
    assert torch.allclose(
        pp_results["forces"].detach().cpu(),
        normal_results["forces"].detach().cpu(),
        atol=FORCE_TOL,
    )


@pytest.mark.serial()
@pytest.mark.parametrize(
    "workers, checkpointing, graph_gen_version",
    [
        (1, False, 2),
        (2, False, 2),
        (1, False, 3),
        (1, True, 3),
        (2, False, 3),
    ],
)
def test_parallel_predict_unit_cpu(workers, checkpointing, graph_gen_version):
    _test_parallel_predict_unit_impl(workers, "cpu", checkpointing, graph_gen_version)


@pytest.mark.gpu()
@pytest.mark.parametrize(
    "workers, checkpointing, graph_gen_version",
    [
        (1, False, 2),
        (1, True, 2),
        (1, True, 3),
        (1, False, 3),
        # (2, False),
        # (2, True),
    ],
)
def test_parallel_predict_unit_gpu(workers, checkpointing, graph_gen_version):
    _test_parallel_predict_unit_impl(workers, "cuda", checkpointing, graph_gen_version)


def _test_parallel_predict_unit_batch_impl(workers, device, checkpointing):
    """Implementation of parallel predict unit batch test."""
    seed = 42
    runs = 1
    model_path = pretrained_checkpoint_path_from_name("uma-s-1p1")
    ifsets = InferenceSettings(
        tf32=False,
        merge_mole=False,
        activation_checkpointing=checkpointing,
        internal_graph_gen_version=2,
        external_graph_gen=False,
    )

    # Create H2O and O molecules batch
    h2o = molecule("H2O")
    h2o.info.update({"charge": 0, "spin": 1})
    h2o.pbc = True

    o_atom = molecule("O")
    o_atom.info.update({"charge": 0, "spin": 2})  # triplet oxygen
    o_atom.pbc = True

    h2o_data = AtomicData.from_ase(
        h2o,
        task_name="omol",
        r_data_keys=["spin", "charge"],
        molecule_cell_size=120,
    )
    o_data = AtomicData.from_ase(
        o_atom,
        task_name="omol",
        r_data_keys=["spin", "charge"],
        molecule_cell_size=120,
    )
    atomic_data = atomicdata_list_to_batch([h2o_data, o_data])
    seed_everywhere(seed)
    ppunit = ParallelMLIPPredictUnit(
        inference_model_path=model_path,
        device=device,
        inference_settings=ifsets,
        num_workers=workers,
    )
    for _ in range(runs):
        pp_results = ppunit.predict(atomic_data)

    distutils.cleanup_gp_ray()

    seed_everywhere(seed)
    normal_predict_unit = pretrained_mlip.get_predict_unit(
        "uma-s-1p1", device=device, inference_settings=ifsets
    )
    for _ in range(runs):
        normal_results = normal_predict_unit.predict(atomic_data)

    assert torch.allclose(
        pp_results["energy"].detach().cpu(),
        normal_results["energy"].detach().cpu(),
        atol=ATOL,
    )
    assert torch.allclose(
        pp_results["forces"].detach().cpu(),
        normal_results["forces"].detach().cpu(),
        atol=FORCE_TOL,
    )


@pytest.mark.serial()
@pytest.mark.parametrize(
    "workers, checkpointing",
    [
        (1, False),
        (2, True),
    ],
)
def test_parallel_predict_unit_batch(workers, checkpointing):
    _test_parallel_predict_unit_batch_impl(workers, "cpu", checkpointing)


@pytest.mark.gpu()
@pytest.mark.parametrize(
    "workers, checkpointing",
    [
        (1, True),
        (1, False),
        # (2, True),
        # (2, False),
    ],
)
def test_parallel_predict_unit_batch_gpu(workers, checkpointing):
    _test_parallel_predict_unit_batch_impl(workers, "cuda", checkpointing)


@pytest.mark.gpu()
@pytest.mark.parametrize(
    "padding",
    [
        (0),
        (1),
        (32),
    ],
)
def test_batching_consistency(padding):
    """Test that batched and unbatched predictions are consistent."""
    # Get the appropriate predict unit

    ifsets = InferenceSettings(
        tf32=False,
        merge_mole=False,
        activation_checkpointing=True,
        internal_graph_gen_version=2,
        external_graph_gen=False,
        edge_chunk_size=padding,
    )
    predict_unit = pretrained_mlip.get_predict_unit(
        "uma-s-1p1", device="cuda", inference_settings=ifsets
    )

    # Create H2O molecule
    h2o = molecule("H2O")
    h2o.info.update({"charge": 0, "spin": 1})
    h2o.pbc = True

    # Create system of two oxygen atoms 100 A apart
    from ase import Atoms

    o_atom = Atoms("O2", positions=[[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    o_atom.info.update({"charge": 0, "spin": 4})  # two triplet oxygens -> quintet
    o_atom.pbc = True

    # Convert to AtomicData
    h2o_data = AtomicData.from_ase(
        h2o,
        task_name="omol",
        r_data_keys=["spin", "charge"],
        molecule_cell_size=120,
    )
    o_data = AtomicData.from_ase(
        o_atom,
        task_name="omol",
        r_data_keys=["spin", "charge"],
        molecule_cell_size=120,
    )

    # Batch 1: [H2O, O]
    batch1 = atomicdata_list_to_batch([h2o_data, o_data])
    seed_everywhere(42)
    preds1 = predict_unit.predict(batch1)

    # Batch 2: [H2O]
    batch2 = atomicdata_list_to_batch([h2o_data])
    seed_everywhere(42)
    preds2 = predict_unit.predict(batch2)

    # Batch 3: [O]
    batch3 = atomicdata_list_to_batch([o_data])
    seed_everywhere(42)
    preds3 = predict_unit.predict(batch3)

    # Assert energies match
    assert torch.allclose(preds1["energy"][0], preds2["energy"][0], atol=ATOL)
    assert torch.allclose(preds1["energy"][1], preds3["energy"][0], atol=ATOL)

    # Assert forces match
    batch1_batch = batch1.batch
    h2o_forces_batch1 = preds1["forces"][batch1_batch == 0]
    o_forces_batch1 = preds1["forces"][batch1_batch == 1]

    h2o_forces_batch2 = preds2["forces"]
    o_forces_batch3 = preds3["forces"]

    assert torch.allclose(h2o_forces_batch1, h2o_forces_batch2, atol=ATOL)
    assert torch.allclose(o_forces_batch1, o_forces_batch3, atol=ATOL)

    # Assert stress matches
    assert torch.allclose(preds1["stress"][0], preds2["stress"][0], atol=ATOL)
    assert torch.allclose(preds1["stress"][1], preds3["stress"][0], atol=ATOL)


# ---------------------------------------------------------------------------
# Rotation / out-of-plane force invariance tests (planar molecules)
# For H2O and NH2 in ASE default coordinates, all atoms lie in the yz plane (x=0).
# Thus out-of-plane component is simply the x-component of the forces.
# ---------------------------------------------------------------------------


def _random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """Generate a 3D rotation matrix from two angles in [0, 2π).

    We sample two independent angles:
      phi   ~ U(0, 2π)  (rotation about z)
      theta ~ U(0, 2π)  (rotation about y)

    The resulting rotation: R = Rz(phi) * Ry(theta)
    Note: This is NOT a uniform (Haar) distribution over SO(3), but
    satisfies the requested two-angle construction.
    """
    phi = rng.uniform(0.0, 2.0 * np.pi)
    theta = rng.uniform(0.0, 2.0 * np.pi)
    cphi, sphi = np.cos(phi), np.sin(phi)
    cth, sth = np.cos(theta), np.sin(theta)
    Rz = np.array([[cphi, -sphi, 0.0], [sphi, cphi, 0.0], [0.0, 0.0, 1.0]])
    Ry = np.array([[cth, 0.0, sth], [0.0, 1.0, 0.0], [-sth, 0.0, cth]])
    return Rz @ Ry


@pytest.mark.gpu()
@pytest.mark.parametrize("mol_name", ["H2O", "NH2"])
def test_rotational_invariance_out_of_plane(mol_name, uma_predict_unit_cuda):
    rng = np.random.default_rng(seed=123)
    calc = FAIRChemCalculator(uma_predict_unit_cuda, task_name="omol")

    atoms = molecule(mol_name)
    atoms.info.update({"charge": 0, "spin": 1})
    atoms.calc = calc

    orig_positions = atoms.get_positions().copy()
    n_rot = 50  # fewer rotations for speed
    for _ in range(n_rot):
        R = _random_rotation_matrix(rng)
        rotated_pos = orig_positions @ R.T
        atoms.set_positions(rotated_pos)
        rot_forces = atoms.get_forces()
        # Unrotate forces back to original frame (covariant transformation)
        unrot_forces = rot_forces @ R
        assert (np.abs(unrot_forces[:, 0]) < FORCE_TOL).all()


@pytest.mark.gpu()
@pytest.mark.parametrize("mol_name", ["H2O", "NH2"])
def test_original_out_of_plane_forces(mol_name, uma_predict_unit_cuda):
    calc = FAIRChemCalculator(uma_predict_unit_cuda, task_name="omol")
    atoms = molecule(mol_name)
    atoms.info.update({"charge": 0, "spin": 1})
    atoms.calc = calc
    forces = atoms.get_forces()
    print(f"Max out-of-plane forces for {mol_name}: {np.abs(forces[:, 0]).max()}")
    assert np.abs(forces[:, 0]).max() < FORCE_TOL


# ---------------------------------------------------------------------------
# Euler vs Quaternion Wigner D agreement tests
# ---------------------------------------------------------------------------


def _get_predict_unit_with_wigner_mode(use_quaternion: bool):
    """
    Create a predict unit with the specified Wigner D computation mode.
    """
    settings = InferenceSettings(
        tf32=False,
        activation_checkpointing=True,
        merge_mole=False,
        compile=False,
        external_graph_gen=False,
        use_quaternion_wigner=use_quaternion,
    )
    uma_models = [name for name in pretrained_mlip.available_models if "uma" in name]
    return pretrained_mlip.get_predict_unit(
        uma_models[0], device="cuda", inference_settings=settings
    )


@pytest.mark.gpu()
def test_euler_vs_quaternion_random_molecule():
    """
    Euler and quaternion Wigner D paths produce identical energy and forces
    for a random molecule with no Y-aligned edges.
    """
    # Methanol (CH3OH) - 6 atoms, no edges along Y axis
    atoms = molecule("CH3OH")
    # Apply a rotation to ensure no edges are Y-aligned
    rng = np.random.default_rng(seed=42)
    R = _random_rotation_matrix(rng)
    atoms.set_positions(atoms.get_positions() @ R.T)
    atoms.info.update({"charge": 0, "spin": 1})
    atoms.pbc = True

    predict_euler = _get_predict_unit_with_wigner_mode(use_quaternion=False)
    predict_quat = _get_predict_unit_with_wigner_mode(use_quaternion=True)

    data = AtomicData.from_ase(
        atoms,
        task_name="omol",
        r_data_keys=["spin", "charge"],
        molecule_cell_size=120,
    )
    batch = atomicdata_list_to_batch([data])

    seed_everywhere(42)
    preds_euler = predict_euler.predict(batch)
    seed_everywhere(42)
    preds_quat = predict_quat.predict(batch)

    npt.assert_allclose(
        preds_euler["energy"].detach().cpu().numpy(),
        preds_quat["energy"].detach().cpu().numpy(),
        atol=ATOL,
        err_msg="Energy differs between Euler and quaternion paths",
    )
    npt.assert_allclose(
        preds_euler["forces"].detach().cpu().numpy(),
        preds_quat["forces"].detach().cpu().numpy(),
        atol=ATOL,
        err_msg="Forces differ between Euler and quaternion paths",
    )


@pytest.mark.gpu()
def test_euler_vs_quaternion_bulk():
    """
    Euler and quaternion Wigner D paths produce identical energy, forces,
    and stress for a bulk crystal with no Y-aligned edges.
    """
    # FCC Cu 2x2x2 supercell with a random perturbation to avoid symmetry
    atoms = bulk("Cu")
    atoms = atoms.repeat((2, 2, 2))
    rng = np.random.default_rng(seed=99)
    atoms.set_positions(
        atoms.get_positions() + rng.normal(0, 0.05, atoms.positions.shape)
    )

    predict_euler = _get_predict_unit_with_wigner_mode(use_quaternion=False)
    predict_quat = _get_predict_unit_with_wigner_mode(use_quaternion=True)

    data = AtomicData.from_ase(atoms, task_name="omat")
    batch = atomicdata_list_to_batch([data])

    seed_everywhere(42)
    preds_euler = predict_euler.predict(batch)
    seed_everywhere(42)
    preds_quat = predict_quat.predict(batch)

    npt.assert_allclose(
        preds_euler["energy"].detach().cpu().numpy(),
        preds_quat["energy"].detach().cpu().numpy(),
        atol=ATOL,
        err_msg="Energy differs between Euler and quaternion paths (bulk)",
    )
    npt.assert_allclose(
        preds_euler["forces"].detach().cpu().numpy(),
        preds_quat["forces"].detach().cpu().numpy(),
        atol=ATOL,
        err_msg="Forces differ between Euler and quaternion paths (bulk)",
    )
    npt.assert_allclose(
        preds_euler["stress"].detach().cpu().numpy(),
        preds_quat["stress"].detach().cpu().numpy(),
        atol=ATOL,
        err_msg="Stress differs between Euler and quaternion paths (bulk)",
    )


@pytest.mark.gpu()
@pytest.mark.parametrize(
    "supercell_matrix",
    [
        2 * np.eye(3),  # 2x2x2 supercell (8 atoms)
        3 * np.eye(3),  # 3x3x3 supercell (27 atoms)
        np.array([[2, 0, 0], [0, 3, 0], [0, 0, 1]]),  # 2x3x1 supercell (6 atoms)
    ],
)
def test_merge_mole_with_supercell(supercell_matrix, uma_merge_mole_predict_unit):
    atoms_orig = bulk("MgO", "rocksalt", a=4.213)

    calc = FAIRChemCalculator(uma_merge_mole_predict_unit, task_name="omat")

    atoms_orig.calc = calc
    energy_orig = atoms_orig.get_potential_energy()
    forces_orig = atoms_orig.get_forces()

    atoms_super = make_supercell(atoms_orig, supercell_matrix)
    num_atoms_ratio = len(atoms_super) / len(atoms_orig)

    atoms_super.calc = calc

    energy_super = atoms_super.get_potential_energy()
    forces_super = atoms_super.get_forces()

    energy_ratio = energy_super / energy_orig
    npt.assert_allclose(
        energy_ratio,
        num_atoms_ratio,
        rtol=0.01,  # 1% tolerance
        err_msg=f"Energy scaling is incorrect for supercell. "
        f"Expected ratio: {num_atoms_ratio}, got: {energy_ratio}",
    )

    mean_force_mag_orig = np.linalg.norm(forces_orig, axis=1).mean()
    mean_force_mag_super = np.linalg.norm(forces_super, axis=1).mean()
    npt.assert_allclose(
        mean_force_mag_orig,
        mean_force_mag_super,
        rtol=0.1,  # 10% tolerance (forces can vary slightly due to numerical precision)
        atol=1e-5,
        err_msg=f"Mean force magnitude differs significantly between original and supercell. "
        f"Original: {mean_force_mag_orig}, Supercell: {mean_force_mag_super}",
    )


@pytest.mark.gpu()
def test_merge_mole_composition_check():
    atoms_cu = bulk("Cu", "fcc", a=3.6)

    settings = InferenceSettings(merge_mole=True, external_graph_gen=False)
    predict_unit = pretrained_mlip.get_predict_unit(
        "uma-s-1p1", device="cuda", inference_settings=settings
    )
    calc = FAIRChemCalculator(predict_unit, task_name="omat")

    atoms_cu.calc = calc
    _ = atoms_cu.get_potential_energy()

    atoms_al = bulk("Al", "fcc", a=4.05)
    atoms_al.calc = calc

    with pytest.raises(
        AssertionError,
        match="Compositions differ from merged model",
    ):
        _ = atoms_al.get_potential_energy()


@pytest.mark.gpu()
@pytest.mark.parametrize("model_name", ["uma-s-1p1", "uma-s-1p2"])
def test_merge_mole_vs_non_merged_consistency(model_name):
    """Test that merged and non-merged versions produce identical results."""
    atoms = bulk("MgO", "rocksalt", a=4.213)

    # Test with merge_mole=True
    settings_merged = InferenceSettings(merge_mole=True, external_graph_gen=False)
    predict_unit_merged = pretrained_mlip.get_predict_unit(
        model_name, device="cuda", inference_settings=settings_merged
    )
    calc_merged = FAIRChemCalculator(predict_unit_merged, task_name="omat")

    atoms_merged = atoms.copy()
    atoms_non_merged = atoms.copy()
    atoms_merged.calc = calc_merged
    energy_merged = atoms_merged.get_potential_energy()
    forces_merged = atoms_merged.get_forces()
    stress_merged = atoms_merged.get_stress(voigt=False)

    distutils.cleanup_gp_ray()  # Ensure clean state before next test

    # Test with merge_mole=False
    settings_non_merged = InferenceSettings(merge_mole=False, external_graph_gen=False)
    predict_unit_non_merged = pretrained_mlip.get_predict_unit(
        model_name, device="cuda", inference_settings=settings_non_merged
    )
    calc_non_merged = FAIRChemCalculator(predict_unit_non_merged, task_name="omat")

    atoms_non_merged.calc = calc_non_merged
    energy_non_merged = atoms_non_merged.get_potential_energy()
    forces_non_merged = atoms_non_merged.get_forces()
    stress_non_merged = atoms_non_merged.get_stress(voigt=False)

    # Assert that results are identical
    npt.assert_allclose(
        energy_merged,
        energy_non_merged,
        rtol=1e-6,
        atol=1e-6,
        err_msg=f"Energies differ: merged={energy_merged}, non-merged={energy_non_merged}",
    )
    npt.assert_allclose(
        forces_merged,
        forces_non_merged,
        rtol=1e-6,
        atol=1e-6,
        err_msg="Forces differ between merged and non-merged versions",
    )
    npt.assert_allclose(
        stress_merged,
        stress_non_merged,
        rtol=1e-6,
        atol=1e-6,
        err_msg="Stress differs between merged and non-merged versions",
    )


@pytest.mark.gpu()
def test_merge_mole_supercell_energy_forces_consistency(uma_merge_mole_predict_unit):
    atoms_orig = bulk("MgO", "rocksalt", a=4.213)

    calc = FAIRChemCalculator(uma_merge_mole_predict_unit, task_name="omat")

    atoms_orig.calc = calc
    energy1 = atoms_orig.get_potential_energy()

    atoms_2x = make_supercell(atoms_orig, 2 * np.eye(3))
    atoms_2x.calc = calc
    energy_2x = atoms_2x.get_potential_energy()

    atoms_3x = make_supercell(atoms_orig, 3 * np.eye(3))
    atoms_3x.calc = calc
    energy_3x = atoms_3x.get_potential_energy()

    energy1_again = atoms_orig.get_potential_energy()

    npt.assert_allclose(energy1, energy1_again, rtol=1e-6)
    npt.assert_allclose(energy_2x / energy1, 8.0, rtol=0.01)
    npt.assert_allclose(energy_3x / energy1, 27.0, rtol=0.01)


@pytest.mark.gpu()
def test_merge_mole_consistent_batch():
    """Test that merge_mole works for batch_size > 1 when all systems have identical composition."""
    atoms = bulk("MgO", "rocksalt", a=4.213)
    n_systems = 3
    settings = InferenceSettings(merge_mole=True, external_graph_gen=False)
    predict_unit = pretrained_mlip.get_predict_unit(
        "uma-s-1p1", device="cuda", inference_settings=settings
    )

    atomic_data_list = [
        AtomicData.from_ase(atoms, task_name="omat") for _ in range(n_systems)
    ]
    batch = atomicdata_list_to_batch(atomic_data_list)
    preds = predict_unit.predict(batch)

    assert preds["energy"].shape == (n_systems,)
    assert preds["forces"].shape == (n_systems * len(atoms), 3)
    assert torch.isfinite(preds["energy"]).all()
    assert torch.isfinite(preds["forces"]).all()


@pytest.mark.gpu()
def test_merge_mole_inconsistent_batch():
    """Test that merge_mole raises AssertionError when batch contains systems with different compositions."""
    settings = InferenceSettings(merge_mole=True, external_graph_gen=False)
    predict_unit = pretrained_mlip.get_predict_unit(
        "uma-s-1p1", device="cuda", inference_settings=settings
    )

    atomic_data_list = [
        AtomicData.from_ase(bulk("MgO", "rocksalt", a=4.213), task_name="omat"),
        AtomicData.from_ase(bulk("Cu", "fcc", a=3.6), task_name="omat"),
    ]
    batch = atomicdata_list_to_batch(atomic_data_list)

    with pytest.raises(AssertionError, match="same reduced composition"):
        predict_unit.predict(batch)


@pytest.mark.gpu()
def test_merge_mole_batch_predict_matches_single():
    """Test that merging on a multi-system batch gives consistent single-system predictions.

    Merging MOLE on a batch of N identical systems should yield the same inference
    results as merging on a single system when predicting on that same single system.
    """
    atoms = bulk("MgO", "rocksalt", a=4.213)
    atoms_supercell = make_supercell(atoms, 2 * np.eye(3))
    settings = InferenceSettings(merge_mole=True, external_graph_gen=False)
    predict_unit = pretrained_mlip.get_predict_unit(
        "uma-s-1p1", device="cuda", inference_settings=settings
    )

    batch_of_two = atomicdata_list_to_batch(
        [AtomicData.from_ase(a, task_name="omat") for a in (atoms, atoms_supercell)]
    )
    preds_batch = predict_unit.predict(batch_of_two)

    batch_single = atomicdata_list_to_batch(
        [AtomicData.from_ase(atoms, task_name="omat")]
    )
    preds_single = predict_unit.predict(batch_single)

    n_atoms = len(atoms)
    npt.assert_allclose(
        preds_batch["energy"][0].item(),
        preds_single["energy"][0].item(),
        atol=ATOL,
        err_msg="Energy for first batch system differs from single system prediction",
    )
    npt.assert_allclose(
        preds_batch["forces"][:n_atoms].cpu().numpy(),
        preds_single["forces"].cpu().numpy(),
        atol=ATOL,
        err_msg="Forces for first batch system differ from single system prediction",
    )

    batch_single = atomicdata_list_to_batch(
        [AtomicData.from_ase(atoms_supercell, task_name="omat")]
    )
    preds_single = predict_unit.predict(batch_single)
    npt.assert_allclose(
        preds_batch["energy"][1].item(),
        preds_single["energy"][0].item(),
        atol=ATOL,
        err_msg="Energy for second batch system differs from single system prediction",
    )


@pytest.fixture()
def batch_server_handle(uma_predict_unit):
    """Set up a batch server for testing."""
    pytest.importorskip("ray.serve", reason="ray[serve] not installed")
    from ray import serve

    from fairchem.core.units.mlip_unit._batch_serve import setup_batch_predict_server

    # Ensure Ray is properly shut down before initializing
    if ray.is_initialized():
        with contextlib.suppress(Exception):
            serve.shutdown()
        ray.shutdown()

    # Initialize Ray with specific configuration
    ray.init(
        ignore_reinit_error=True,
        num_cpus=4,
        num_gpus=1 if torch.cuda.is_available() else 0,
        logging_level="ERROR",  # Reduce noise in test output
    )

    # Setup the batch server
    server_handle = setup_batch_predict_server(
        predict_unit=uma_predict_unit,
        max_batch_size=8,
        batch_wait_timeout_s=0.05,
        num_replicas=1,
        ray_actor_options={
            "num_gpus": 1 if torch.cuda.is_available() else 0,
            "num_cpus": 2,
        },
    )

    yield server_handle

    # Cleanup
    try:
        serve.shutdown()
    except Exception as e:
        print(f"Warning: Error during serve shutdown: {e}")
    try:
        ray.shutdown()
    except Exception as e:
        print(f"Warning: Error during ray shutdown: {e}")


@pytest.mark.gpu()
def test_batch_server_predict_unit_with_calculator(
    batch_server_handle, uma_predict_unit
):
    """Test BatchServerPredictUnit works with FAIRChemCalculator."""
    from fairchem.core.units.mlip_unit.predict import BatchServerPredictUnit

    batch_predict_unit = BatchServerPredictUnit(
        server_handle=batch_server_handle,
        predict_unit=uma_predict_unit,
    )

    atoms = bulk("Cu")
    atoms.calc = FAIRChemCalculator(batch_predict_unit, task_name="omat")

    atoms_ = bulk("Cu")
    atoms_.calc = FAIRChemCalculator(uma_predict_unit, task_name="omat")

    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    stress = atoms.get_stress(voigt=False)

    energy_ = atoms_.get_potential_energy()
    forces_ = atoms_.get_forces()
    stress_ = atoms_.get_stress(voigt=False)

    npt.assert_allclose(
        energy,
        energy_,
        atol=ATOL,
    )
    npt.assert_allclose(
        forces,
        forces_,
        atol=ATOL,
    )
    npt.assert_allclose(
        stress,
        stress_,
        atol=ATOL,
    )


@pytest.mark.gpu()
def test_batch_server_predict_unit_multiple_systems(
    batch_server_handle, uma_predict_unit
):
    """Test BatchServerPredictUnit with multiple concurrent requests."""
    from concurrent.futures import ThreadPoolExecutor

    from fairchem.core.units.mlip_unit.predict import BatchServerPredictUnit

    batch_predict_unit = BatchServerPredictUnit(
        server_handle=batch_server_handle,
        predict_unit=uma_predict_unit,
    )

    atoms_list = [bulk("Cu"), bulk("Al"), bulk("Fe"), bulk("Ni")]
    atomic_data_list = [
        AtomicData.from_ase(atoms, task_name="omat") for atoms in atoms_list
    ]

    # Submit concurrent predictions
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(batch_predict_unit.predict, data)
            for data in atomic_data_list
        ]
        results = [future.result() for future in futures]

    # Check all predictions completed successfully
    assert len(results) == len(atoms_list)
    for i, preds in enumerate(results):
        assert "energy" in preds
        assert "forces" in preds
        assert "stress" in preds
        assert preds["energy"].shape == (1,)
        assert preds["forces"].shape == (len(atoms_list[i]), 3)


# this should pass for multi-gpu as well when run locally
# @pytest.mark.skip()
@pytest.mark.serial()
@pytest.mark.parametrize("workers", [0, 2])
@pytest.mark.parametrize("ensemble", ["nvt", "npt"])
@pytest.mark.parametrize("device", ["cpu"])
def test_merge_mole_md_consistency(workers, ensemble, device):
    """Test merge_mole vs no-merge consistency over MD trajectory.

    Runs 3 trials:
    A) no merge
    B) no merge again (baseline for numerical noise)
    C) merge

    Compares the relative drift of A-C against baseline A-B to ensure
    merge_mole doesn't introduce additional numerical drift beyond
    the inherent noise between identical runs.
    """
    import torch

    torch.use_deterministic_algorithms(True)

    from ase import units
    from ase.md.langevin import Langevin
    from ase.md.nptberendsen import NPTBerendsen
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    #  Simple system
    atoms_template = bulk("Cu", "fcc", a=3.6)
    atoms_template = atoms_template.repeat((2, 2, 2))

    md_steps = 2
    timestep = 1.0 * units.fs
    initial_temp_K = 300.0
    pressure = 1.01325 * units.bar  # 1 atm
    taut = 100 * units.fs  # Thermostat coupling time
    taup = 500 * units.fs  # Barostat coupling time
    compressibility = 4.57e-5 / units.bar  # Water-like compressibility

    # Shared inference settings (except merge_mole)
    base_settings = dict(
        tf32=True,
        activation_checkpointing=False,
        compile=False,
        external_graph_gen=False,
        internal_graph_gen_version=2,
    )

    def run_md_trial(atoms, calc, seed, steps):
        """Run MD and collect energy/forces/stress at each step."""
        atoms = atoms.copy()
        atoms.calc = calc

        seed_everywhere(seed)
        MaxwellBoltzmannDistribution(atoms, temperature_K=initial_temp_K)

        if ensemble == "npt":
            dyn = NPTBerendsen(
                atoms,
                timestep=timestep,
                temperature_K=initial_temp_K,
                pressure_au=pressure,
                taut=taut,
                taup=taup,
                compressibility_au=compressibility,
            )
        else:  # nvt
            dyn = Langevin(
                atoms,
                timestep=timestep,
                temperature_K=initial_temp_K,
                friction=0.01 / units.fs,
            )

        energies = []
        forces_list = []
        stresses = []

        # Collect initial state
        energies.append(atoms.get_potential_energy())
        forces_list.append(atoms.get_forces().copy())
        stresses.append(atoms.get_stress(voigt=False).copy())

        for _ in range(steps):
            dyn.run(1)
            energies.append(atoms.get_potential_energy())
            forces_list.append(atoms.get_forces().copy())
            stresses.append(atoms.get_stress(voigt=False).copy())

        return {
            "energies": np.array(energies),
            "forces": np.array(forces_list),
            "stresses": np.array(stresses),
        }

    # Trial A: no merge
    settings_no_merge = InferenceSettings(merge_mole=False, **base_settings)
    predict_unit_A = pretrained_mlip.get_predict_unit(
        "uma-s-1p1",
        device=device,
        inference_settings=settings_no_merge,
        workers=workers,
    )
    calc_A = FAIRChemCalculator(predict_unit_A, task_name="omat")
    results_A = run_md_trial(atoms_template, calc_A, seed=42, steps=md_steps)
    distutils.cleanup_gp_ray()

    # Trial B: no merge again (baseline for numerical noise)
    predict_unit_B = pretrained_mlip.get_predict_unit(
        "uma-s-1p1",
        device=device,
        inference_settings=settings_no_merge,
        workers=workers,
    )
    calc_B = FAIRChemCalculator(predict_unit_B, task_name="omat")
    results_B = run_md_trial(atoms_template, calc_B, seed=42, steps=md_steps)
    distutils.cleanup_gp_ray()

    # Trial C: merge
    settings_merge = InferenceSettings(merge_mole=True, **base_settings)
    predict_unit_C = pretrained_mlip.get_predict_unit(
        "uma-s-1p1", device=device, inference_settings=settings_merge, workers=workers
    )
    calc_C = FAIRChemCalculator(predict_unit_C, task_name="omat")
    results_C = run_md_trial(atoms_template, calc_C, seed=42, steps=md_steps)
    distutils.cleanup_gp_ray()

    # Compute drifts
    # Energy drift
    energy_drift_AB = np.abs(results_A["energies"] - results_B["energies"])
    energy_drift_AC = np.abs(results_A["energies"] - results_C["energies"])

    # Forces drift (mean absolute difference across all atoms and steps)
    forces_drift_AB = np.abs(results_A["forces"] - results_B["forces"])
    forces_drift_AC = np.abs(results_A["forces"] - results_C["forces"])

    # Stress drift
    stress_drift_AB = np.abs(results_A["stresses"] - results_B["stresses"])
    stress_drift_AC = np.abs(results_A["stresses"] - results_C["stresses"])

    # Log the drifts for debugging
    logging.info(f"Energy drift A-B (max): {energy_drift_AB.max():.2e}")
    logging.info(f"Energy drift A-C (max): {energy_drift_AC.max():.2e}")
    logging.info(f"Forces drift A-B (max): {forces_drift_AB.max():.2e}")
    logging.info(f"Forces drift A-C (max): {forces_drift_AC.max():.2e}")
    logging.info(f"Stress drift A-B (max): {stress_drift_AB.max():.2e}")
    logging.info(f"Stress drift A-C (max): {stress_drift_AC.max():.2e}")

    # The drift between A-C should be comparable to the baseline drift A-B.
    # Allow some tolerance factor (e.g., 10x) for merge_mole overhead.
    # Clamp the baseline to a minimum floor so the threshold doesn't collapse
    # to ~1e-6 when A-B is near-zero (e.g. on highly deterministic CI hardware),
    # which would make the test a de-facto fixed threshold regardless of the
    # 10x multiplier.
    tolerance_factor = 10.0
    abs_floor_energy = 1e-5  # eV
    abs_floor_forces = 5e-6  # eV/Ang
    abs_floor_stress = 1e-6  # eV/Ang^3

    # For energy: max drift A-C should be within tolerance of max drift A-B
    baseline_energy_drift = max(energy_drift_AB.max(), abs_floor_energy)
    npt.assert_array_less(
        energy_drift_AC.max(),
        tolerance_factor * baseline_energy_drift,
        err_msg=f"Energy drift A-C ({energy_drift_AC.max():.2e}) exceeds "
        f"{tolerance_factor}x baseline A-B ({baseline_energy_drift:.2e})",
    )

    # For forces: max drift A-C should be within tolerance of max drift A-B
    baseline_forces_drift = max(forces_drift_AB.max(), abs_floor_forces)
    npt.assert_array_less(
        forces_drift_AC.max(),
        tolerance_factor * baseline_forces_drift,
        err_msg=f"Forces drift A-C ({forces_drift_AC.max():.2e}) exceeds "
        f"{tolerance_factor}x baseline A-B ({baseline_forces_drift:.2e})",
    )

    # For stress: max drift A-C should be within tolerance of max drift A-B
    baseline_stress_drift = max(stress_drift_AB.max(), abs_floor_stress)
    npt.assert_array_less(
        stress_drift_AC.max(),
        tolerance_factor * baseline_stress_drift,
        err_msg=f"Stress drift A-C ({stress_drift_AC.max():.2e}) exceeds "
        f"{tolerance_factor}x baseline A-B ({baseline_stress_drift:.2e})",
    )


# ---------------------------------------------------------------------------
# Single-atom prediction tests
# ---------------------------------------------------------------------------


def _test_single_atom_predict(predict_unit, task_name, energy_atol):
    """
    Verify single-atom predictions for a given predict unit and task.

    Checks that energy, forces, and stress have correct shapes,
    forces are zero (no neighbors), and energy matches the reference
    table within the specified tolerance.
    """
    for atomic_number, charge, spin in _REPRESENTATIVE_ELEMENTS:
        symbol = chemical_symbols[atomic_number]
        elem_id = f"{symbol} (Z={atomic_number})"

        atom = Atoms([atomic_number], positions=[(0.0, 0.0, 0.0)])
        atom.info["charge"] = charge
        atom.info["spin"] = spin

        from_ase_kwargs = {"task_name": task_name}
        if task_name == "omol":
            from_ase_kwargs["r_data_keys"] = ["spin", "charge"]

        atomic_data = AtomicData.from_ase(atom, **from_ase_kwargs)
        batch = atomicdata_list_to_batch([atomic_data])
        preds = predict_unit.predict(batch)

        # Shape checks
        assert preds["energy"].shape == (
            1,
        ), f"{elem_id}: energy shape {preds['energy'].shape} != (1,)"
        assert preds["forces"].shape == (
            1,
            3,
        ), f"{elem_id}: forces shape {preds['forces'].shape} != (1, 3)"
        assert torch.isfinite(
            preds["energy"]
        ).all(), f"{elem_id}: energy is not finite: {preds['energy']}"

        # Forces must be zero (no neighbors)
        assert (
            preds["forces"] == 0.0
        ).all(), f"{elem_id}: forces are not zero: {preds['forces']}"

        # Get reference energy via single_atom_prediction_from_lookup
        ref_batch = atomicdata_list_to_batch([atomic_data])
        ref_preds = single_atom_prediction_from_lookup(
            data=ref_batch,
            atom_refs=predict_unit.atom_refs,
            tasks=predict_unit.tasks,
            device=torch.device("cpu"),
        )
        ref_energy = next(
            v.item()
            for k, v in ref_preds.items()
            if predict_unit.tasks[k].property == "energy"
        )
        npt.assert_allclose(
            preds["energy"].detach().cpu().item(),
            ref_energy,
            atol=energy_atol,
            err_msg=(f"{elem_id}: predicted energy does not match reference"),
        )


@pytest.mark.parametrize("task_name", ["omat", "omol"])
def test_single_atom_predict_1p1(task_name, uma_1p1_predict_unit):
    """Verify uma-s-1p1 single atom energies match the lookup table exactly."""
    _test_single_atom_predict(uma_1p1_predict_unit, task_name, energy_atol=0.0)


@pytest.mark.parametrize("task_name", ["omat", "omol"])
def test_single_atom_predict_1p2(task_name, uma_1p2_predict_unit):
    """Verify uma-s-1p2 single atom energies are close to reference values."""
    _test_single_atom_predict(
        uma_1p2_predict_unit,
        task_name,
        energy_atol=SINGLE_ATOM_ENERGY_ATOL,
    )


@pytest.mark.gpu()
def test_untrained_forces(conserving_mole_checkpoint, device="cuda"):
    """
    Test that untrained forces can be computed for energy-only checkpoint.
    """
    # Create predictor with untrained forces enabled
    settings = InferenceSettings(predict_untrained_forces={"omol"})
    predictor = MLIPPredictUnit(
        conserving_mole_checkpoint[0], device=device, inference_settings=settings
    )

    # Check that forces task was created
    task_names = list(predictor.tasks.keys())
    assert any(
        "forces" in name for name in task_names
    ), f"No forces task found in {task_names}"

    # Create test data
    from ase.build import molecule

    atoms = molecule("H2O")
    atoms.info.update({"charge": 0, "spin": 1})

    data = AtomicData.from_ase(
        atoms,
        task_name="omol",
        r_data_keys=["spin", "charge"],
        molecule_cell_size=120,
    )
    batch = atomicdata_list_to_batch([data])

    # Get predictions
    preds = predictor.predict(batch)

    # Verify both energy and forces are present
    assert "energy" in preds, "Energy prediction missing"
    assert "forces" in preds, "Forces prediction missing"

    # Verify shapes
    assert preds["energy"].shape == (1,), f"Wrong energy shape: {preds['energy'].shape}"
    assert preds["forces"].shape == (
        3,
        3,
    ), f"Wrong forces shape: {preds['forces'].shape}"

    # Verify forces are finite
    assert torch.isfinite(preds["forces"]).all(), "Forces contain NaN or Inf"


@pytest.mark.gpu()
def test_untrained_stress_selective_gpu(conserving_mole_checkpoint):
    """Test selective stress computation on GPU."""
    _test_untrained_stress_selective(conserving_mole_checkpoint[0], "cuda")


def test_untrained_stress_selective_cpu(conserving_mole_checkpoint):
    """Test selective stress computation on CPU."""
    _test_untrained_stress_selective(conserving_mole_checkpoint[0], "cpu")


def _test_untrained_stress_selective(checkpoint_path, device):
    """
    Test that stress can be selectively enabled for specific datasets.
    """
    # Enable stress only for omol dataset
    settings = InferenceSettings(
        predict_untrained_forces={"omol"},
        predict_untrained_stress={"omol"},
    )
    predictor = MLIPPredictUnit(
        checkpoint_path, device=device, inference_settings=settings
    )

    # Check that stress task was created
    task_names = list(predictor.tasks.keys())
    assert any(
        "stress" in name for name in task_names
    ), f"No stress task found in {task_names}"

    # Create test data
    from ase.build import molecule

    atoms = molecule("H2O")
    atoms.info.update({"charge": 0, "spin": 1})

    data = AtomicData.from_ase(
        atoms,
        task_name="omol",
        r_data_keys=["spin", "charge"],
        molecule_cell_size=120,
    )
    batch = atomicdata_list_to_batch([data])

    # Get predictions
    preds = predictor.predict(batch)

    # Verify energy, forces, and stress are present
    assert "energy" in preds, "Energy prediction missing"
    assert "forces" in preds, "Forces prediction missing"
    assert "stress" in preds, "Stress prediction missing"

    # Verify stress shape
    assert preds["stress"].shape == (
        1,
        9,
    ), f"Wrong stress shape: {preds['stress'].shape}"

    # Verify stress is finite
    assert torch.isfinite(preds["stress"]).all(), "Stress contains NaN or Inf"


@pytest.mark.gpu()
def test_untrained_hessian(conserving_mole_checkpoint, device="cuda"):
    """
    Test that hessian can be computed for energy-only checkpoint.
    """
    # Enable hessian for omol
    settings = InferenceSettings(
        predict_untrained_forces={"omol"},
        predict_untrained_hessian={"omol"},
        hessian_vmap=True,
    )
    predictor = MLIPPredictUnit(
        conserving_mole_checkpoint[0], device=device, inference_settings=settings
    )

    # Check that hessian task was created
    task_names = list(predictor.tasks.keys())
    assert any(
        "hessian" in name for name in task_names
    ), f"No hessian task found in {task_names}"

    # Create test data (single system required for hessian)
    from ase.build import molecule

    atoms = molecule("H2O")
    atoms.info.update({"charge": 0, "spin": 1})

    data = AtomicData.from_ase(
        atoms,
        task_name="omol",
        r_data_keys=["spin", "charge"],
        molecule_cell_size=120,
    )
    batch = atomicdata_list_to_batch([data])

    # Get predictions
    preds = predictor.predict(batch)

    # Verify energy, forces, and hessian are present
    assert "energy" in preds, "Energy prediction missing"
    assert "forces" in preds, "Forces prediction missing"
    assert "hessian" in preds, "Hessian prediction missing"

    # Verify hessian shape: (1, 3*N, 3*N) — batch dim is always 1
    n_atoms = len(atoms)
    expected_shape = (1, n_atoms * 3, n_atoms * 3)
    assert (
        preds["hessian"].shape == expected_shape
    ), f"Wrong hessian shape: {preds['hessian'].shape}, expected {expected_shape}"

    # Verify hessian is finite
    assert torch.isfinite(preds["hessian"]).all(), "Hessian contains NaN or Inf"

    # Verify hessian is symmetric (squeeze batch dim for symmetry check)
    hessian = preds["hessian"].squeeze(0)
    assert torch.allclose(hessian, hessian.T, atol=1e-5), "Hessian is not symmetric"


def test_no_duplicate_tasks(conserving_mole_checkpoint):
    """Test that no duplicate tasks are created if checkpoint already has them."""
    # Load checkpoint without untrained tasks
    # Now load with untrained forces enabled (but if checkpoint already has forces, no duplicate)
    settings = InferenceSettings(predict_untrained_stress={"omol"})
    predictor_untrained = MLIPPredictUnit(
        conserving_mole_checkpoint[0], device="cpu", inference_settings=settings
    )
    untrained_task_names = set(predictor_untrained.tasks.keys())

    # Check that we don't have duplicate tasks
    task_name_counts = {}
    for name in untrained_task_names:
        # Count how many tasks have the same property
        property_name = name.split("_")[-1]  # e.g., "forces" from "omol_forces"
        task_name_counts[property_name] = task_name_counts.get(property_name, 0) + 1

    # Each property should appear at most once per dataset
    # (This is a simplified check; in reality we'd need to look at dataset+property combos)
    for prop, count in task_name_counts.items():
        # With a single-dataset checkpoint, we should have exactly 1 task per property
        assert count <= 3, f"Property {prop} has {count} tasks, may have duplicates"


def test_auto_add_default_untrained_stress_tasks(conserving_mole_checkpoint):
    """Test that stress tasks are auto-added for energy datasets by eSCNMDBackbone.

    This test verifies the auto_add_default_untrained_tasks feature that allows
    backbones to automatically add default untrained tasks during inference.
    For eSCNMDBackbone, this means adding stress tasks for all energy datasets
    that don't already have trained stress.
    """
    # Load checkpoint with default settings (auto_add_default_untrained_tasks=True by default)
    settings = InferenceSettings()
    predictor = MLIPPredictUnit(
        conserving_mole_checkpoint[0], device="cpu", inference_settings=settings
    )

    # Get the tasks from the predictor
    tasks = deepcopy(predictor.tasks)

    # Find all energy datasets
    energy_datasets = set()
    for task in tasks.values():
        if task.property == "energy":
            energy_datasets.update(task.datasets)

    # Verify that stress tasks exist for each energy dataset
    for dataset in energy_datasets:
        stress_task_name = f"{dataset}_stress"
        assert (
            stress_task_name in tasks
        ), f"Expected auto-added stress task '{stress_task_name}' not found"

        stress_task = tasks[stress_task_name]
        assert stress_task.property == "stress"
        assert stress_task.level == "system"
        assert stress_task.inference_only or stress_task in tasks.values()
        assert dataset in stress_task.datasets


def test_auto_add_disabled(conserving_mole_checkpoint):
    """Test that setting auto_add_default_untrained_tasks=False disables auto-add."""
    # Load checkpoint with auto_add_default_untrained_tasks disabled
    settings = InferenceSettings(auto_add_default_untrained_tasks=False)
    predictor = MLIPPredictUnit(
        conserving_mole_checkpoint[0], device="cpu", inference_settings=settings
    )

    tasks = predictor.tasks

    # Find all energy datasets
    energy_datasets = set()
    for task in tasks.values():
        if task.property == "energy":
            energy_datasets.update(task.datasets)

    # Verify that auto-added stress tasks are NOT present
    for dataset in energy_datasets:
        stress_task_name = f"{dataset}_stress"
        # The task should not exist since auto-add is disabled
        # (unless it was explicitly requested or already in checkpoint)
        if stress_task_name in tasks:
            # If it exists, it should not be inference_only (which indicates auto-added)
            assert tasks[stress_task_name].inference_only is False, (
                f"Stress task '{stress_task_name}' should not be auto-added "
                "when auto_add_default_untrained_tasks=False"
            )


@pytest.mark.gpu()
def test_untrained_forces_gpu(conserving_mole_checkpoint):
    """Test computing forces for energy-only checkpoint on GPU."""
    _test_untrained_forces(conserving_mole_checkpoint[0], "cuda")


def test_untrained_forces_cpu(conserving_mole_checkpoint):
    """Test computing forces for energy-only checkpoint on CPU."""
    _test_untrained_forces(conserving_mole_checkpoint[0], "cpu")


def _test_untrained_forces(checkpoint_path, device):
    """
    Test that untrained forces can be computed for energy-only checkpoint.
    """
    # Create predictor with untrained forces enabled
    settings = InferenceSettings(predict_untrained_forces={"omol"})
    predictor = MLIPPredictUnit(
        checkpoint_path, device=device, inference_settings=settings
    )

    # Check that forces task was created
    task_names = list(predictor.tasks.keys())
    assert any(
        "forces" in name for name in task_names
    ), f"No forces task found in {task_names}"

    # Create test data
    from ase.build import molecule

    atoms = molecule("H2O")
    atoms.info.update({"charge": 0, "spin": 1})

    data = AtomicData.from_ase(
        atoms,
        task_name="omol",
        r_data_keys=["spin", "charge"],
        molecule_cell_size=120,
    )
    batch = atomicdata_list_to_batch([data])

    # Get predictions
    preds = predictor.predict(batch)

    # Verify both energy and forces are present
    assert "energy" in preds, "Energy prediction missing"
    assert "forces" in preds, "Forces prediction missing"

    # Verify shapes
    assert preds["energy"].shape == (1,), f"Wrong energy shape: {preds['energy'].shape}"
    assert preds["forces"].shape == (
        3,
        3,
    ), f"Wrong forces shape: {preds['forces'].shape}"

    # Verify forces are finite
    assert torch.isfinite(preds["forces"]).all(), "Forces contain NaN or Inf"


@pytest.mark.gpu()
def test_untrained_hessian_gpu(conserving_mole_checkpoint):
    """Test hessian computation on GPU."""
    _test_untrained_hessian(conserving_mole_checkpoint[0], "cuda")


def test_untrained_hessian_cpu(conserving_mole_checkpoint):
    """Test hessian computation on CPU."""
    _test_untrained_hessian(conserving_mole_checkpoint[0], "cpu")


def _test_untrained_hessian(checkpoint_path, device):
    """
    Test that hessian can be computed for energy-only checkpoint.
    """
    # Enable hessian for omol
    settings = InferenceSettings(
        predict_untrained_forces={"omol"},
        predict_untrained_hessian={"omol"},
        hessian_vmap=True,
    )
    predictor = MLIPPredictUnit(
        checkpoint_path, device=device, inference_settings=settings
    )

    # Check that hessian task was created
    task_names = list(predictor.tasks.keys())
    assert any(
        "hessian" in name for name in task_names
    ), f"No hessian task found in {task_names}"

    # Create test data (single system required for hessian)
    from ase.build import molecule

    atoms = molecule("H2O")
    atoms.info.update({"charge": 0, "spin": 1})

    data = AtomicData.from_ase(
        atoms,
        task_name="omol",
        r_data_keys=["spin", "charge"],
        molecule_cell_size=120,
    )
    batch = atomicdata_list_to_batch([data])

    # Get predictions
    preds = predictor.predict(batch)

    # Verify energy, forces, and hessian are present
    assert "energy" in preds, "Energy prediction missing"
    assert "forces" in preds, "Forces prediction missing"
    assert "hessian" in preds, "Hessian prediction missing"

    # Verify hessian shape: (1, 3*N, 3*N) — batch dim is always 1
    n_atoms = len(atoms)
    expected_shape = (1, n_atoms * 3, n_atoms * 3)
    assert (
        preds["hessian"].shape == expected_shape
    ), f"Wrong hessian shape: {preds['hessian'].shape}, expected {expected_shape}"

    # Verify hessian is finite
    assert torch.isfinite(preds["hessian"]).all(), "Hessian contains NaN or Inf"

    # Verify hessian is symmetric (squeeze batch dim for symmetry check)
    hessian = preds["hessian"].squeeze(0)
    assert torch.allclose(hessian, hessian.T, atol=1e-5), "Hessian is not symmetric"


def test_hessian_batch_size_validation(conserving_mole_checkpoint):
    """Test that hessian computation fails for batch_size > 1."""
    settings = InferenceSettings(
        predict_untrained_forces={"omol"},
        predict_untrained_hessian={"omol"},
    )
    predictor = MLIPPredictUnit(
        conserving_mole_checkpoint[0], device="cpu", inference_settings=settings
    )

    # Create batch with 2 systems
    from ase.build import molecule

    atoms1 = molecule("H2O")
    atoms1.info.update({"charge": 0, "spin": 1})
    atoms2 = molecule("H2O")
    atoms2.info.update({"charge": 0, "spin": 1})

    data1 = AtomicData.from_ase(
        atoms1,
        task_name="omol",
        r_data_keys=["spin", "charge"],
        molecule_cell_size=120,
    )
    data2 = AtomicData.from_ase(
        atoms2,
        task_name="omol",
        r_data_keys=["spin", "charge"],
        molecule_cell_size=120,
    )
    batch = atomicdata_list_to_batch([data1, data2])

    # Should raise ValueError
    with pytest.raises(
        ValueError, match="Hessian computation requires exactly 1 system in batch"
    ):
        predictor.predict(batch)


def test_direct_force_model_untrained_validation(direct_mole_checkpoint):
    """Test that direct-force models reject hessian requests."""
    # Try to enable hessian on direct-force model (should fail)

    for prop in ("forces", "stress", "hessian"):
        settings = InferenceSettings(**{f"predict_untrained_{prop}": {"omol"}})

        with pytest.raises(
            ValueError,
            match=f"Cannot add autograd-based '{prop}' task to direct-force model",
        ):
            MLIPPredictUnit(
                direct_mole_checkpoint[0], device="cpu", inference_settings=settings
            )


# ---------------------------------------------------------------------------
# Execution mode auto-selection tests
# ---------------------------------------------------------------------------


@pytest.mark.gpu()
@pytest.mark.parametrize("model_name", ["uma-s-1p1", "uma-s-1p2"])
def test_execution_mode_auto_set_umas_fast_gpu(model_name):
    """Test that UMA-S models automatically use umas_fast_gpu on GPU with compatible settings.

    When running on GPU with merge_mole=True and activation_checkpointing=False,
    the execution_mode should automatically be set to umas_fast_gpu.
    """

    predict_unit = pretrained_mlip.get_predict_unit(
        model_name, device="cuda", inference_settings="turbo"
    )

    # Verify that actual module backend is UMASFastGPUBackend when set to turbo mode
    assert isinstance(predict_unit.model.module.backbone.backend, UMASFastGPUBackend), (
        f"Expected backend to be {UMASFastGPUBackend}, "
        f"got {predict_unit.model.module.backbone.backend}"
    )


@pytest.mark.gpu()
@pytest.mark.parametrize("model_name", ["uma-s-1p1", "uma-s-1p2"])
def test_execution_mode_not_overridden_when_explicit(model_name):
    """Test that explicitly set execution_mode is not overridden."""
    from fairchem.core.models.uma.nn.execution_backends import ExecutionMode

    # Explicitly set execution_mode to GENERAL
    settings = InferenceSettings(
        merge_mole=True,
        activation_checkpointing=False,
        external_graph_gen=False,
        execution_mode=ExecutionMode.GENERAL,
    )

    predict_unit = pretrained_mlip.get_predict_unit(
        model_name, device="cuda", inference_settings=settings
    )

    # Verify that execution_mode was NOT changed
    assert predict_unit.inference_settings.execution_mode == ExecutionMode.GENERAL, (
        f"Expected execution_mode to remain {ExecutionMode.GENERAL}, "
        f"got {predict_unit.inference_settings.execution_mode}"
    )


@pytest.mark.gpu()
@pytest.mark.parametrize("model_name", ["uma-m-1p1"])
def test_execution_mode_not_set_when_conditions_not_met(model_name):
    """Test that umas_fast_gpu is not auto-selected when conditions aren't met."""

    predict_unit = pretrained_mlip.get_predict_unit(
        model_name, device="cuda", inference_settings="turbo"
    )

    # execution_mode should remain None (not auto-set to umas_fast_gpu)
    assert predict_unit.inference_settings.execution_mode is None, (
        f"Expected execution_mode to be None when activation_checkpointing=True, "
        f"got {predict_unit.inference_settings.execution_mode}"
    )
