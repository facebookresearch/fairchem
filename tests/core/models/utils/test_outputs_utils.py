from __future__ import annotations

import numpy as np
import pytest
from ase.build import molecule

from fairchem.core import pretrained_mlip
from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.models.utils.outputs import get_numerical_hessian


@pytest.mark.gpu()
def test_numerical_hessian():
    """Test numerical Hessian calculation using get_numerical_hessian utility."""
    predict_unit = pretrained_mlip.get_predict_unit("uma-s-1", device="cuda")

    atoms = molecule("H2O")
    atoms.info.update({"charge": 0, "spin": 1})

    # Convert to AtomicData
    data = AtomicData.from_ase(
        atoms,
        task_name="omol",
        r_data_keys=["spin", "charge"],
        molecule_cell_size=120,
    )

    hessian = get_numerical_hessian(data, predict_unit, device="cuda")
    hessian = hessian.detach().cpu().numpy()

    # Check shape (3 atoms * 3 coords = 9x9 matrix)
    assert hessian.shape == (9, 9)
    assert np.isfinite(hessian).all()
