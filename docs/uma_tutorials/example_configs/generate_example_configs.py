#!/usr/bin/env python3
"""Regenerate the example input structures linked from the UMA catalysis tutorial.

These ``.xyz`` files are the **unrelaxed input geometries** the
``docs/uma_tutorials/uma_catalysis_tutorial.md`` "no UMA access?" admonitions
offer for download, so a reader can load them into the UMA web demo
(https://facebook-fairchem-uma-demo.hf.space/) without installing the gated
model. Building the geometry needs no model -- only relaxing/predicting does --
so each structure is produced with the exact construction code from the
tutorial, stopping *before* the UMA relaxation step. They are deliberately the
pre-relaxation inputs, not the optimized outputs.

Deterministic: a fixed seed and the experimental lattice constant make the
output byte-reproducible. Run from the repo root (or anywhere):

    python docs/uma_tutorials/example_configs/generate_example_configs.py

Requires ``fairchem-data-oc`` (and its ``fairchem-core`` dependency), ``ase``
and ``pymatgen`` -- the same packages the tutorial itself imports.

Outputs (written next to this script):
  ni_bulk.xyz        bulk Ni fcc (experimental a = 3.52 Å)
  ni111_slab.xyz     pymatgen SlabGenerator Ni(111), 4 layers, 10 Å vacuum
  h_on_ni111.xyz     AdsorbateSlabConfig *H, heuristic placement
  4h_on_ni111.xyz    MultipleAdsorbateSlabConfig 4x *H
  co_on_ni111.xyz    MultipleAdsorbateSlabConfig *CO
  c_o_on_ni111.xyz   MultipleAdsorbateSlabConfig *C + *O
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from ase import Atoms
from ase.build import bulk
from ase.io import read, write
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.ase import AseAtomsAdaptor

from fairchem.data.oc.core import (
    Adsorbate,
    AdsorbateSlabConfig,
    Bulk,
    MultipleAdsorbateSlabConfig,
    Slab,
)

OUT = Path(__file__).resolve().parent

SEED = 42
# Experimental Ni lattice constant; the tutorial starts here and relaxes with
# UMA. For a downloadable *input* geometry the experimental value is faithful.
A_OPT = 3.52
NUM_SITES = 5  # tutorial's FAST_DOCS value; one representative config is enough


def _write_clean(atoms: Atoms, name: str) -> None:
    """Write minimal extxyz (symbols + positions + cell), fully periodic.

    The fairchem.data.oc generators attach per-atom metadata (tags,
    bulk_wyckoff, bulk_equivalent, move_mask constraints). Those extra columns
    are empty for adsorbate atoms, which makes the extxyz non-round-trippable
    and risky for the demo's reader. We strip to a clean structure and set
    ``pbc=True`` on all axes to match the tutorial, which calls
    ``set_pbc([True, True, True])`` on every structure before prediction.
    """
    clean = Atoms(
        numbers=atoms.get_atomic_numbers(),
        positions=atoms.get_positions(),
        cell=atoms.get_cell(),
        pbc=True,
    )
    write(OUT / name, clean)


def gen_ni_bulk() -> Atoms:
    ni_bulk = bulk("Ni", "fcc", a=A_OPT, cubic=True)
    _write_clean(ni_bulk, "ni_bulk.xyz")
    return ni_bulk


def gen_ni111_slab(ni_bulk: Atoms) -> None:
    adaptor = AseAtomsAdaptor()
    ni_structure = adaptor.get_structure(ni_bulk)
    facet = (1, 1, 1)
    n_layers = 4
    slabgen = SlabGenerator(
        ni_structure,
        facet,
        min_slab_size=n_layers * A_OPT / np.sqrt(sum(h**2 for h in facet)),
        min_vacuum_size=10.0,
        center_slab=True,
    )
    pmg_slab = slabgen.get_slabs()[0]
    slab = adaptor.get_atoms(pmg_slab)
    slab.center(vacuum=10.0, axis=2)
    _write_clean(slab, "ni111_slab.xyz")


def _ni_slab_obj(ni_bulk_atoms: Atoms) -> Slab:
    ni_bulk_obj = Bulk(bulk_atoms=ni_bulk_atoms)
    return Slab.from_bulk_get_specific_millers(
        bulk=ni_bulk_obj, specific_millers=(1, 1, 1)
    )[0]


def gen_h_on_ni111(ni_bulk_atoms: Atoms) -> None:
    np.random.seed(SEED)
    slab = _ni_slab_obj(ni_bulk_atoms)
    adsorbate_h = Adsorbate(adsorbate_smiles_from_db="*H")
    cfg = AdsorbateSlabConfig(
        slab, adsorbate_h, mode="random_site_heuristic_placement", num_sites=NUM_SITES
    )
    _write_clean(cfg.atoms_list[0], "h_on_ni111.xyz")


def gen_4h_on_ni111(ni_bulk_atoms: Atoms) -> None:
    np.random.seed(SEED)
    slab = _ni_slab_obj(ni_bulk_atoms)
    adsorbates = [Adsorbate(adsorbate_smiles_from_db="*H") for _ in range(4)]
    cfg = MultipleAdsorbateSlabConfig(slab, adsorbates, num_configurations=NUM_SITES)
    _write_clean(cfg.atoms_list[0], "4h_on_ni111.xyz")


def gen_co_on_ni111(ni_bulk_atoms: Atoms) -> None:
    np.random.seed(SEED)
    slab = _ni_slab_obj(ni_bulk_atoms)
    adsorbate_co = Adsorbate(adsorbate_smiles_from_db="*CO")
    cfg = MultipleAdsorbateSlabConfig(slab, [adsorbate_co], num_configurations=NUM_SITES)
    _write_clean(cfg.atoms_list[0], "co_on_ni111.xyz")


def gen_c_o_on_ni111(ni_bulk_atoms: Atoms) -> None:
    np.random.seed(SEED)
    slab = _ni_slab_obj(ni_bulk_atoms)
    adsorbate_c = Adsorbate(adsorbate_smiles_from_db="*C")
    adsorbate_o = Adsorbate(adsorbate_smiles_from_db="*O")
    cfg = MultipleAdsorbateSlabConfig(
        slab, [adsorbate_c, adsorbate_o], num_configurations=NUM_SITES
    )
    _write_clean(cfg.atoms_list[0], "c_o_on_ni111.xyz")


# Expected composition of each written file -- a structural assertion that the
# generator produced what the tutorial describes. (symbol -> count)
EXPECTED = {
    "ni_bulk.xyz": {"Ni": 4},
    "ni111_slab.xyz": {"Ni": 5},
    "h_on_ni111.xyz": {"Ni": 96, "H": 1},
    "4h_on_ni111.xyz": {"Ni": 96, "H": 4},
    "co_on_ni111.xyz": {"Ni": 96, "C": 1, "O": 1},
    "c_o_on_ni111.xyz": {"Ni": 96, "C": 1, "O": 1},
}


def verify() -> None:
    for name, expected in EXPECTED.items():
        atoms = read(OUT / name)
        counts: dict[str, int] = {}
        for sym in atoms.get_chemical_symbols():
            counts[sym] = counts.get(sym, 0) + 1
        assert counts == expected, f"{name}: got {counts}, expected {expected}"
        assert all(atoms.get_pbc()), f"{name}: expected pbc True on all axes"
        assert atoms.get_volume() > 0, f"{name}: non-periodic / zero-volume cell"
        print(f"  {name:20s} {len(atoms):3d} atoms  {atoms.get_chemical_formula()}  pbc=TTT  OK")


if __name__ == "__main__":
    ni_bulk = gen_ni_bulk()
    gen_ni111_slab(ni_bulk)
    ni_bulk_atoms = bulk("Ni", "fcc", a=A_OPT, cubic=True)
    gen_h_on_ni111(ni_bulk_atoms)
    gen_4h_on_ni111(ni_bulk_atoms)
    gen_co_on_ni111(ni_bulk_atoms)
    gen_c_o_on_ni111(ni_bulk_atoms)
    print("Generated and verified:")
    verify()
