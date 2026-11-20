"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch

from fairchem.core.modules.scaling.scale_factor import ScaleFactor


@pytest.mark.parametrize("with_reference", [False, True])
def test_fit_matches_reference_variance(with_reference):
    scale = ScaleFactor("test")
    x = torch.tensor([[-2.0, -4.0], [0.0, 0.0], [2.0, 4.0]])
    ref = x / 2 if with_reference else None
    with scale.fit_context_():
        assert torch.equal(scale(x, ref=ref), x)
        stats, ratio, fitted = scale.fit_()
    expected_variance = torch.var(ref, dim=0).mean() if ref is not None else 1.0
    assert ratio == pytest.approx(torch.var(x, dim=0).mean() / expected_variance)
    assert fitted == pytest.approx(ratio**-0.5)
    assert stats["n_samples"] == 3
    assert scale.stats is None
    assert scale.fitted
    assert torch.var(scale(x), dim=0).mean() == pytest.approx(expected_variance)


def test_scale_initialization_reset_and_state_roundtrip():
    scale = ScaleFactor()
    index_fn = Mock()
    scale.initialize_(index_fn=index_fn)
    scale.set_(2.0)
    scale.set_(torch.tensor(2.0))
    assert torch.equal(scale(torch.ones(2, 1)), torch.full((2, 1), 2.0))
    index_fn.assert_called_once()
    restored = ScaleFactor()
    restored.load_state_dict(scale.state_dict())
    assert restored.fitted
    restored.load_state_dict(scale.state_dict())
    with pytest.raises(ValueError, match="inconsistent"):
        restored.set_(3.0)
    with pytest.raises(ValueError, match="inconsistent"):
        restored.load_state_dict({"scale_factor": torch.tensor(3.0)})
    restored.reset_()
    assert not restored.fitted
    assert torch.equal(restored(torch.ones(2, 1)), torch.ones(2, 1))


def test_scale_can_explicitly_disable_checkpoint_consistency():
    scale = ScaleFactor(enforce_consistency=False)
    scale.set_(2.0)
    scale.load_state_dict({"scale_factor": torch.tensor(3.0)})
    assert scale.scale_factor == 3.0


def test_fit_requires_nonzero_observed_variance():
    scale = ScaleFactor()
    with pytest.raises(AssertionError, match="Stats not set"):
        scale.fit_()
    with scale.fit_context_():
        with pytest.raises(AssertionError, match="variance_in"):
            scale.fit_()
        scale(torch.ones(2, 1))
        with pytest.raises(AssertionError, match="variance_out"):
            scale.fit_()
