"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import re
from types import SimpleNamespace

import pytest
import torch

from fairchem.core.modules.normalization.element_references import (
    ElementReferences,
)


def _make_batch(atomic_numbers, natoms):
    atomic_numbers_full = torch.tensor(atomic_numbers, dtype=torch.long)
    batch_full = torch.repeat_interleave(
        torch.arange(len(natoms)), torch.tensor(natoms)
    )
    return SimpleNamespace(
        atomic_numbers_full=atomic_numbers_full,
        batch_full=batch_full,
        natoms=torch.tensor(natoms),
    )


class TestElementReferencesNaN:
    def test_undo_refs_raises_on_untrained_elements(self):
        refs_tensor = torch.tensor([float("nan"), 1.0, 2.0, float("nan")])
        elem_refs = ElementReferences(refs_tensor)
        batch = _make_batch(atomic_numbers=[1, 2, 3], natoms=[3])
        tensor = torch.tensor([0.5])
        with pytest.raises(ValueError, match="atomic numbers"):
            elem_refs.undo_refs(batch, tensor)

    def test_apply_refs_raises_on_untrained_elements(self):
        refs_tensor = torch.tensor([float("nan"), 1.0, float("nan")])
        elem_refs = ElementReferences(refs_tensor)
        batch = _make_batch(atomic_numbers=[1, 2], natoms=[2])
        tensor = torch.tensor([10.0])
        with pytest.raises(ValueError, match="atomic numbers"):
            elem_refs.apply_refs(batch, tensor)

    def test_multiple_untrained_elements_reported(self):
        refs_tensor = torch.tensor([float("nan"), 1.0, float("nan"), float("nan")])
        elem_refs = ElementReferences(refs_tensor)
        batch = _make_batch(atomic_numbers=[1, 2, 3], natoms=[3])
        tensor = torch.tensor([0.0])
        with pytest.raises(ValueError, match=re.escape("[2, 3]")):
            elem_refs.undo_refs(batch, tensor)

    def test_trained_elements_pass_without_error(self):
        refs_tensor = torch.tensor([float("nan"), 1.0, 2.0])
        elem_refs = ElementReferences(refs_tensor)
        batch = _make_batch(atomic_numbers=[1, 2, 1], natoms=[3])
        tensor = torch.tensor([5.0])
        result = elem_refs.undo_refs(batch, tensor)
        expected = tensor + (1.0 + 2.0 + 1.0)
        assert torch.allclose(result, expected)

    def test_multi_system_batch_with_untrained_element(self):
        refs_tensor = torch.tensor([float("nan"), 1.0, 2.0, float("nan")])
        elem_refs = ElementReferences(refs_tensor)
        batch = _make_batch(atomic_numbers=[1, 2, 3, 1], natoms=[2, 2])
        tensor = torch.tensor([0.0, 0.0])
        with pytest.raises(ValueError, match="atomic numbers"):
            elem_refs.undo_refs(batch, tensor)

    def test_error_message_content(self):
        refs_tensor = torch.tensor([float("nan"), float("nan"), 2.0, float("nan")])
        elem_refs = ElementReferences(refs_tensor)
        batch = _make_batch(atomic_numbers=[1, 3], natoms=[2])
        tensor = torch.tensor([0.0])
        with pytest.raises(ValueError, match="not trained on"):
            elem_refs.undo_refs(batch, tensor)
