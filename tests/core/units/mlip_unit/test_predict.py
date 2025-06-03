from __future__ import annotations

import numpy.testing as npt
import pytest
from ase.build import add_adsorbate, bulk, fcc100, molecule

from fairchem.core import FAIRChemCalculator, pretrained_mlip
from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch

ATOL = 5e-6


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
