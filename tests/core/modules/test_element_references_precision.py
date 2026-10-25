"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from types import SimpleNamespace

import torch

from fairchem.core.modules.normalization.element_references import (
    ElementReferences,
    LinearReferences,
)

# A Pt-sized atomic baseline, as in the r2SCAN fine-tune whose element
# references reach ~1e6 eV/atom.
PT_REF = -518302.4767
N_PT = 27
RESIDUAL = -42.7115


def _refs_tensor(dtype: torch.dtype) -> torch.Tensor:
    refs = torch.zeros(119, dtype=dtype)
    refs[78] = PT_REF
    return refs


RESIDUAL_F32 = torch.tensor([RESIDUAL], dtype=torch.float32).double().item()


def _expected_total(refs: torch.Tensor) -> float:
    # Exact float64 arithmetic on the float32-stored operand values.
    return RESIDUAL_F32 + N_PT * float(refs[78].double())


def test_undo_refs_adds_huge_baseline_in_float64():
    """
    Undoing float32 references must not quantize the total energy.
    """
    # References deliberately stored in float32, as a config that does not
    # request DoubleTensor would produce; precision must not depend on it.
    refs = _refs_tensor(torch.float32)
    module = ElementReferences(refs)
    batch = SimpleNamespace(
        natoms=torch.tensor([N_PT]),
        batch_full=torch.zeros(N_PT, dtype=torch.long),
        atomic_numbers_full=torch.full((N_PT,), 78, dtype=torch.long),
    )
    residual = torch.tensor([RESIDUAL], dtype=torch.float32)

    total = module.undo_refs(batch, residual)

    assert total.dtype == torch.float64
    expected = _expected_total(refs)
    assert abs(total.item() - expected) < 1e-6
    # The same addition in float32 loses ~1 eV at this magnitude.
    total_f32 = residual + refs[78] * N_PT
    assert abs(total_f32.item() - expected) > 0.5

    recovered = module.apply_refs(batch, total)
    assert abs(recovered.item() - RESIDUAL_F32) < 1e-9


def test_linear_references_add_huge_baseline_in_float64():
    """
    LinearReferences must not downcast the references to the target dtype.
    """
    refs = _refs_tensor(torch.float32)
    module = LinearReferences(refs)
    batch = SimpleNamespace(
        natoms=torch.tensor([N_PT]),
        batch=torch.zeros(N_PT, dtype=torch.long),
        atomic_numbers=torch.full((N_PT,), 78, dtype=torch.long),
    )
    residual = torch.tensor([[RESIDUAL]], dtype=torch.float32)

    total = module(residual, batch)

    assert total.dtype == torch.float64
    assert abs(total.item() - _expected_total(refs)) < 1e-6

    recovered = module.dereference(total, batch)
    assert abs(recovered.item() - RESIDUAL_F32) < 1e-9
