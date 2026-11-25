"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from fairchem.core.models.utils.basis import (
    SIREN,
    Basis,
    FourierSmearing,
    GaussianSmearing,
    SINESmearing,
)


def test_radial_smearing_limits_and_gradients():
    for smearing in (SINESmearing, FourierSmearing):
        for cosine in (False, True):
            layer = smearing(2, num_freqs=3, use_cosine=cosine).double()
            x = torch.zeros(2, 2, dtype=torch.float64, requires_grad=True)
            result = layer(x)
            assert result.shape == (2, 6)
            assert torch.equal(
                result, torch.ones_like(result) if cosine else torch.zeros_like(result)
            )
            assert torch.autograd.gradcheck(layer, (x,))
    gaussian = GaussianSmearing(2, num_freqs=3).double()
    x = torch.full((2, 2), 0.5, dtype=torch.float64, requires_grad=True)
    result = gaussian(x)
    assert torch.equal(result[:, 2:4], torch.ones(2, 2, dtype=torch.float64))
    assert torch.all(result > 0)
    assert torch.all(result <= 1)
    assert torch.autograd.gradcheck(gaussian, (x,))


def test_basis_output_dimensions_and_raw_identity():
    x = torch.zeros(2, 2)
    for kind in (
        "powersine",
        "powercosine",
        "fouriersine",
        "gauss",
        "linact",
        "raw",
        "rawcat",
    ):
        layer = Basis(2, num_freqs=3, basis_type=kind)
        result = layer(x)
        assert result.shape == (2, layer.out_dim)
        assert torch.isfinite(result).all()
        if kind.startswith("raw"):
            assert result is x
    with pytest.raises(RuntimeError, match="Undefined basis"):
        Basis(2, basis_type="invalid")


def test_spherical_basis_composition_uses_only_distance_columns():
    x = torch.tensor([[1.0, 0, 0, 0.1], [0, 1.0, 0, 0.2]])
    sph = torch.ones(2, 4)
    changed_direction = x.clone()
    changed_direction[:, :3] *= -1
    for kind in ("sphcat", "sphmul", "sphm40", "sphnosine"):
        layer = Basis(4, num_freqs=2, basis_type=kind, sph=SimpleNamespace(out_dim=4))
        result = layer(x, sph)
        assert result.shape == (2, layer.out_dim)
        assert torch.equal(result, layer(changed_direction, sph))
        if kind == "sphcat":
            assert torch.equal(result[:, :4], sph)
        elif kind == "sphnosine":
            assert torch.allclose(result, x[:, 3:].expand(2, 4))
    with pytest.raises(AssertionError):
        Basis(4, basis_type="sphcat")
    with pytest.raises(ValueError, match="cat or mul not specified"):
        Basis(4, basis_type="sph", sph=SimpleNamespace(out_dim=4))


def test_siren_input_derivatives_are_finite_and_match_finite_differences(seed_fixture):
    for initializer in (None, "siren"):
        layer = SIREN(
            [3, 3], num_in_features=2, out_features=1, initializer=initializer
        ).double()
        x = torch.tensor([[0.1, -0.1]], dtype=torch.float64, requires_grad=True)
        assert layer(x).shape == (1, 1)
        assert torch.autograd.gradcheck(layer, (x,))
