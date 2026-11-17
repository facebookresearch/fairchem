"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from ase import Atoms

from fairchem.core.datasets.atomic_data import AtomicData, atomicdata_list_to_batch
from fairchem.core.modules.normalization.element_references import (
    LinearReferences,
    create_element_references,
    fit_linear_references,
)


@pytest.fixture()
def reference_dataset():
    """
    Use a full-rank two-element dataset with exactly known reference energies.
    """
    data = []
    for formula in ("H2", "O2", "HO", "H2O", "HO2"):
        atoms = Atoms(formula)
        item = AtomicData.from_ase(atoms)
        item.energy = torch.tensor(
            [sum(-2.0 if z == 1 else -3.0 for z in atoms.numbers)]
        )
        data.append(item)
    return data


@pytest.mark.parametrize("use_numpy", [True, False])
def test_fit_and_apply_linear_references(reference_dataset, use_numpy):
    refs = fit_linear_references(
        ["energy"],
        reference_dataset,
        batch_size=2,
        max_num_elements=8,
        use_numpy=use_numpy,
        shuffle=False,
    )["energy"]
    assert refs.element_references[1].item() == pytest.approx(-2.0, abs=1e-5)
    assert refs.element_references[8].item() == pytest.approx(-3.0, abs=1e-5)
    assert torch.equal(refs.element_references[2:8], torch.zeros(6))
    batch = atomicdata_list_to_batch(reference_dataset)
    energy = batch.energy.reshape(-1, 1)
    residual = refs.dereference(energy, batch)
    assert torch.allclose(residual, torch.zeros_like(residual), atol=1e-5)
    assert torch.allclose(refs(residual, batch), energy)
    flat_residual = refs.dereference(batch.energy, batch, reshaped=False)
    assert torch.allclose(flat_residual, torch.zeros_like(flat_residual), atol=1e-5)


def test_fit_respects_batch_limit_and_clamps_excess_batches(reference_dataset, caplog):
    for num_batches in (1, 20):
        refs = fit_linear_references(
            ["energy"],
            reference_dataset,
            batch_size=2,
            num_batches=num_batches,
            max_num_elements=8,
            shuffle=False,
            log_metrics=False,
        )["energy"]
        assert refs.element_references[1].item() == pytest.approx(-2.0)
        assert refs.element_references[8].item() == pytest.approx(-3.0)
    assert "num_batches 20" in caplog.text


def test_create_element_references_supported_formats_and_precedence(tmp_path):
    values = torch.tensor([0.0, -2.0, -3.0])
    state = {"element_references": values}
    assert torch.equal(
        create_element_references(state_dict=state).element_references, values
    )
    pt = tmp_path / "refs.pt"
    torch.save(state, pt)
    files = [pt]
    for key in ("coeff", "element_references"):
        path = tmp_path / f"{key}.npz"
        np.savez(path, **{key: values.numpy()})
        files.append(path)
    for path in files:
        refs = create_element_references(
            file=path, state_dict={"element_references": torch.ones(3)}
        )
        assert torch.equal(refs.element_references, values)
    assert torch.equal(
        LinearReferences(max_num_elements=2).element_references, torch.zeros(3)
    )
    with pytest.raises(RuntimeError, match="Unable to load"):
        create_element_references(state_dict={})
    with pytest.raises(RuntimeError, match="not supported"):
        create_element_references(file=tmp_path / "refs.txt")
