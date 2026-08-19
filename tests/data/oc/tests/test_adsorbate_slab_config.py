"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import random

import numpy as np
import pytest
from fairchem.data.oc.core import Adsorbate, AdsorbateSlabConfig, Bulk, Slab
from fairchem.data.oc.core.adsorbate_slab_config import get_interstitial_distances


def assert_valid_adslab(adslab, slab, adsorbate, num_sites=100):
    """
    Check placement invariants independent of Pymatgen slab enumeration.
    """
    assert len(adslab.atoms_list) == num_sites
    assert len(adslab.sites) == num_sites
    assert len(np.unique(np.round(adslab.sites, decimals=4), axis=0)) == num_sites

    num_slab_atoms = len(slab.atoms)
    for atoms in adslab.atoms_list[:2]:
        assert len(atoms) == num_slab_atoms + len(adsorbate.atoms)
        np.testing.assert_allclose(
            atoms.positions[:num_slab_atoms], slab.atoms.positions
        )
        np.testing.assert_allclose(atoms.cell, slab.atoms.cell)
        assert atoms.get_chemical_symbols()[num_slab_atoms:] == (
            adsorbate.atoms.get_chemical_symbols()
        )
        assert np.all(atoms.get_tags()[num_slab_atoms:] == 2)
        assert np.array_equal(atoms.pbc, [True, True, False])


@pytest.fixture(scope="class")
def load_data(request):
    request.cls.bulk = Bulk(bulk_id_from_db=0)
    request.cls.adsorbate = Adsorbate(adsorbate_id_from_db=80)


@pytest.mark.usefixtures("load_data")
class TestAdslab:
    def test_adslab_init(self):
        random.seed(1)
        np.random.seed(1)

        slab = Slab.from_bulk_get_random_slab(self.bulk)
        adslab = AdsorbateSlabConfig(slab, self.adsorbate, num_sites=100)
        assert_valid_adslab(adslab, slab, self.adsorbate)

    def test_adslab_init_slab_only(self):
        random.seed(1)
        np.random.seed(1)

        _slab = Slab.from_bulk_get_random_slab(self.bulk)
        slab_atoms = _slab.atoms
        slab = Slab(slab_atoms=slab_atoms)
        adslab = AdsorbateSlabConfig(slab, self.adsorbate, num_sites=100)
        assert_valid_adslab(adslab, slab, self.adsorbate)

    def test_adslab_seeded_placement_is_deterministic(self):
        random.seed(1)
        np.random.seed(1)
        slab = Slab.from_bulk_get_random_slab(self.bulk)

        random.seed(2)
        np.random.seed(2)
        adslab1 = AdsorbateSlabConfig(slab, self.adsorbate, num_sites=2)

        random.seed(2)
        np.random.seed(2)
        adslab2 = AdsorbateSlabConfig(slab, self.adsorbate, num_sites=2)

        np.testing.assert_allclose(adslab1.sites, adslab2.sites)
        for atoms1, atoms2 in zip(adslab1.atoms_list, adslab2.atoms_list):
            np.testing.assert_allclose(atoms1.positions, atoms2.positions)

    def test_num_augmentations_per_site(self):
        random.seed(1)
        np.random.seed(1)

        slab = Slab.from_bulk_get_random_slab(self.bulk)
        adslab = AdsorbateSlabConfig(
            slab, self.adsorbate, num_sites=1, num_augmentations_per_site=100
        )
        assert len(adslab.atoms_list) == 100

        sites = [f"{i[0]:.4f}_{i[1]:.4f}_{i[2]:.4f}" for i in adslab.sites]
        assert len(set(sites)) == 1

    def test_placement_overlap(self):
        """
        Test that the adsorbate does not overlap with the slab.
        """
        random.seed(1)
        np.random.seed(1)

        slab = Slab.from_bulk_get_random_slab(self.bulk)
        adslab = AdsorbateSlabConfig(
            slab, self.adsorbate, num_sites=100, interstitial_gap=0.1
        )
        assert len(adslab.atoms_list) == 100

        min_distance_close = [
            np.isclose(min(get_interstitial_distances(atoms)), 0.1)
            for atoms in adslab.atoms_list
        ]
        assert all(min_distance_close)

        adslab = AdsorbateSlabConfig(
            slab, self.adsorbate, num_sites=100, interstitial_gap=0.5
        )
        min_distance_close = [
            np.isclose(min(get_interstitial_distances(atoms)), 0.5)
            for atoms in adslab.atoms_list
        ]
        assert all(min_distance_close)

    def test_is_adsorbate_com_on_normal(self):
        random.seed(1)
        np.random.seed(1)

        slab = Slab.from_bulk_get_random_slab(self.bulk)
        normal = np.cross(slab.atoms.cell[0], slab.atoms.cell[1])
        adslab = AdsorbateSlabConfig(slab, self.adsorbate, num_sites=100, mode="random")
        sample_ids = np.random.randint(0, len(adslab.atoms_list), 10)

        cp_test = []
        for idx in sample_ids:
            site, atoms = adslab.sites[idx], adslab.atoms_list[idx]
            mask = atoms.get_tags() == 2
            adsorbate_atoms = atoms[mask]
            adsorbate_com = adsorbate_atoms.get_center_of_mass()
            cp = np.cross(normal, adsorbate_com - site)
            cp_test.append(cp)
            assert np.isclose(cp_test, 0).all()

    def test_is_adsorbate_binding_atom_on_normal(self):
        random.seed(1)
        np.random.seed(1)

        slab = Slab.from_bulk_get_random_slab(self.bulk)
        normal = np.cross(slab.atoms.cell[0], slab.atoms.cell[1])
        adslab = AdsorbateSlabConfig(
            slab, self.adsorbate, num_sites=100, mode="heuristic"
        )
        binding_idx = self.adsorbate.binding_indices[0]
        sample_ids = np.random.randint(0, len(adslab.atoms_list), 10)

        cp_test = []
        for idx in sample_ids:
            site, atoms = adslab.sites[idx], adslab.atoms_list[idx]
            mask = atoms.get_tags() == 2
            adsorbate_atoms = atoms[mask]
            binding_atom = adsorbate_atoms[binding_idx].position
            cp = np.cross(normal, binding_atom - site)
            cp_test.append(cp)
            assert np.isclose(cp_test, 0).all()
