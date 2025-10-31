from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest
import torch
from ase.build import add_adsorbate, bulk, fcc100, molecule

from fairchem.core import FAIRChemCalculator, pretrained_mlip
from fairchem.core.calculate.pretrained_mlip import pretrained_checkpoint_path_from_name
from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
from fairchem.core.units.mlip_unit.predict import ParallelMLIPPredictUnit
from tests.conftest import seed_everywhere

FORCE_TOL = 1e-4
ATOL = 1e-5


def get_fcc_carbon_xtal(
    num_atoms: int,
    lattice_constant: float = 3.8,
):
    # lattice_constant = 3.8, fcc generates a supercell with ~50 edges/atom
    atoms = bulk("C", "fcc", a=lattice_constant)
    n_cells = int(np.ceil(np.cbrt(num_atoms)))
    atoms = atoms.repeat((n_cells, n_cells, n_cells))
    indices = np.random.choice(len(atoms), num_atoms, replace=False)
    sampled_atoms = atoms[indices]
    return sampled_atoms


@pytest.fixture(scope="module")
def uma_predict_unit(request):
    uma_models = [name for name in pretrained_mlip.available_models if "uma" in name]
    return pretrained_mlip.get_predict_unit(uma_models[0])


@pytest.mark.gpu()
def test_single_dataset_predict(uma_predict_unit):
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


@pytest.mark.gpu()
def test_multiple_dataset_predict(uma_predict_unit):
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


@pytest.mark.gpu()
@pytest.mark.parametrize(
    "workers, device",
    [
        (1, "cpu"),
        (2, "cpu"),
        (1, "cuda"),
    ],
)
def test_parallel_predict_unit(workers, device):
    seed = 42
    runs = 2
    model_path = pretrained_checkpoint_path_from_name("uma-s-1p1")
    num_atoms = 10
    ifsets = InferenceSettings(
        tf32=False,
        merge_mole=True,
        activation_checkpointing=True,
        internal_graph_gen_version=2,
        external_graph_gen=False,
    )
    atoms = get_fcc_carbon_xtal(num_atoms)
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


# ---------------------------------------------------------------------------
# Rotation / out-of-plane force invariance tests (planar molecules)
# For H2O and NH2 in ASE default coordinates, all atoms lie in the y–z plane (x=0).
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
def test_rotational_invariance_out_of_plane(mol_name):
    rng = np.random.default_rng(seed=123)
    predict_unit = pretrained_mlip.get_predict_unit("uma-s-1", device="cuda")
    calc = FAIRChemCalculator(predict_unit, task_name="omol")

    atoms = molecule(mol_name)
    atoms.info.update({"charge": 0, "spin": 1})
    atoms.calc = calc

    orig_positions = atoms.get_positions().copy()\

    n_rot = 50  # fewer rotations for speed
    for _ in range(n_rot):
        R = _random_rotation_matrix(rng)
        rotated_pos = orig_positions @ R.T
        atoms.set_positions(rotated_pos)
        rot_forces = atoms.get_forces()
        # Unrotate forces back to original frame (covariant transformation)
        unrot_forces = rot_forces @ R
        assert (np.abs(unrot_forces[:,0])<FORCE_TOL).all()



@pytest.mark.gpu()
@pytest.mark.xfail(reason="Y-aligned edges cause problems in eSCN family", strict=False)
@pytest.mark.parametrize("mol_name", ["H2O", "NH2"])
def test_original_out_of_plane_forces(mol_name):
    predict_unit = pretrained_mlip.get_predict_unit("uma-s-1", device="cuda")
    calc = FAIRChemCalculator(predict_unit, task_name="omol")
    atoms = molecule(mol_name)
    atoms.info.update({"charge": 0, "spin": 1})
    atoms.calc = calc
    forces = atoms.get_forces()
    print(f"Max out-of-plane forces for {mol_name}: {np.abs(forces[:,0]).max()}")
    assert np.abs(forces[:,0]).max() < FORCE_TOL
