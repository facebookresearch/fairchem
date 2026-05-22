"""Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

End-to-end conformer generation for FastCSP
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from ase import Atoms, units
from fairchem.applications.fastcsp.core.utils.slurm import (
    get_slurm_config,
    submit_slurm_jobs,
    wait_for_jobs,
)
from fairchem.applications.fastcsp.core.workflow.relax import (
    create_calculator,
    relax_atoms,
)
from rdkit import Chem
from rdkit.Chem import (
    AllChem,
    rdDetermineBonds,
    rdMolAlign,
    rdMolTransforms,
)
from rdkit.ML.Cluster import Butina
from scipy.spatial import KDTree
from tqdm import tqdm

ENERGY_LABEL = "Energy (kJ/mol)"
EV_TO_KJMOL = units.mol / units.kJ


# ---------------------------------------------------------------------------
# Config loading + defaults
# ---------------------------------------------------------------------------

# Single source of truth for stage defaults. Anything not provided by the
# user falls back here. Printed verbatim at the start of every worker task
# so it's clear what was used.
CONF_GEN_DEFAULTS: dict = {
    # Target size of the seed pool RDKit embeds before any filtering.
    # The actual ETKDGv3 call uses ceil(initial_pool_size / 2.5) because
    # ``generate_conformers`` blends 4 seeding strategies internally.
    "initial_pool_size": 50,
    "seed": 42,
    "rmsd_thresh": 0.25,  # Angstrom
    "cluster_energy_thresh": 1.5,  # kJ/mol
    "include_hydrogens": True,  # if False, best-RMSD on heavy atoms only
    "output_format": "xyz",  # xyz | mol | sdf
}

RELAX_DEFAULTS: dict = {
    "calculator": "uma_sm_1p1_omol",
    "optimizer": "BFGS",
    "fmax": 0.05,  # eV/A
    "max_steps": 100,
    "fix_symmetry": False,  # no-op for isolated conformers
    "relax_cell": False,  # no-op for isolated conformers
    "write_traj": False,
    "traj_interval": 1,
    "energy_window": 40.0,  # kJ/mol cap on final relaxed pool
}


def _split_row_overrides(row: dict) -> tuple[dict, dict]:
    """Pick per-molecule overrides out of a CSV row.

    Columns whose names match a key in ``CONF_GEN_DEFAULTS`` / ``RELAX_DEFAULTS``
    are treated as per-molecule overrides. Empty / NaN cells fall through.
    Values are coerced to the same Python type as the default.
    """
    _TRUTHY = {"true", "yes", "y", "1"}

    def _coerce(ref, v):
        if isinstance(ref, bool):
            return v if isinstance(v, bool) else str(v).strip().lower() in _TRUTHY
        if isinstance(ref, int):
            return int(float(v))  # tolerate "100.0"
        return type(ref)(v)

    def _pick(defaults):
        return {
            k: _coerce(defaults[k], row[k])
            for k in defaults
            if k in row and not pd.isna(row[k])
        }

    return _pick(CONF_GEN_DEFAULTS), _pick(RELAX_DEFAULTS)


def _resolve_config(
    tag: str,
    rdkit_user: dict,
    relax_user: dict,
    rdkit_row: dict | None = None,
    relax_row: dict | None = None,
) -> tuple[dict, dict]:
    """Merge defaults < YAML overrides < per-row CSV overrides; print all."""
    rdkit_row = rdkit_row or {}
    relax_row = relax_row or {}
    rdkit_cfg = {**CONF_GEN_DEFAULTS, **rdkit_user, **rdkit_row}
    relax_cfg = {**RELAX_DEFAULTS, **relax_user, **relax_row}
    if rdkit_user or relax_user:
        print(f"[{tag}] yaml overrides: rdkit={rdkit_user!r} relax={relax_user!r}")
    if rdkit_row or relax_row:
        print(f"[{tag}] csv overrides:  rdkit={rdkit_row!r} relax={relax_row!r}")
    return rdkit_cfg, relax_cfg


def load_conformer_generation_config(
    path: str | Path,
) -> tuple[dict, dict, dict, Path | None, Path | None]:
    """Load a YAML conformer-generation config.

    Returns ``(rdkit, relax, raw_config, root, molecules_path)``.
    """
    cfg_path = Path(path).resolve()
    with open(cfg_path) as fh:
        data = yaml.safe_load(fh) or {}
    rdkit = dict(data.get("rdkit") or {})
    relax = dict(data.get("relax") or {})

    root = Path(data["root"]).expanduser().resolve() if data.get("root") else None
    molecules_path = None
    if data.get("molecules"):
        m = Path(data["molecules"]).expanduser()
        if m.is_absolute() and m.exists():
            molecules_path = m
        else:
            candidates = [cfg_path.parent / m]
            if root is not None:
                candidates.append(root / m)
            molecules_path = next(
                (c.resolve() for c in candidates if c.exists()),
                (cfg_path.parent / m).resolve(),
            )
        if not molecules_path.name.lower().endswith(".csv"):
            raise ValueError(f"molecules must be .csv: {molecules_path}")
        if not molecules_path.exists():
            raise FileNotFoundError(f"molecules CSV not found: {molecules_path}")
    return rdkit, relax, data, root, molecules_path


# ---------------------------------------------------------------------------
# 1. RDKit conformer generation
# ---------------------------------------------------------------------------


def _rotatable_bonds(mol: Chem.Mol) -> list[tuple[int, int, int, int]]:
    """
    Get the atom indices of the rotatable bonds in the molecule.

    :param mol: Molecule to find torsions in
    :return: atom indices defining the torsion, with preference given to
             highest atomic weight atoms for the first and last atom
    """
    tors_smarts = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")
    bonds: list[tuple[int, int, int, int]] = []
    for at1, at2 in mol.GetSubstructMatches(tors_smarts):
        at0 = max(
            (a for a in mol.GetAtomWithIdx(at1).GetNeighbors() if a.GetIdx() != at2),
            key=lambda x: x.GetMass(),
        )
        at3 = max(
            (a for a in mol.GetAtomWithIdx(at2).GetNeighbors() if a.GetIdx() != at1),
            key=lambda x: x.GetMass(),
        )
        bond = (at0.GetIdx(), at1, at2, at3.GetIdx())
        if at3.IsInRing():
            bond = tuple(reversed(bond))
        bonds.append(bond)
    return bonds


def _generate_rdkit_confs(
    mol: Chem.Mol,
    num_confs: int,
    seed: int,
    ff_opt: bool,
    random_coords: bool = False,
) -> Chem.Mol:
    """
    Generate conformers with RDKit (ETKDGv3, optional MMFF relax).
    """
    conf_mol = Chem.Mol(mol)
    params = AllChem.ETKDGv3()
    params.clearConfs = True
    params.enforceChirality = True
    params.randomSeed = seed
    params.useRandomCoords = random_coords
    params.useSmallRingTorsions = True
    AllChem.EmbedMultipleConfs(conf_mol, numConfs=num_confs, params=params)
    if ff_opt:
        AllChem.MMFFOptimizeMoleculeConfs(conf_mol)
    return conf_mol


def _generate_random_torsion_confs(
    mol: Chem.Mol, num_confs: int, seed: int
) -> Chem.Mol:
    """
    Generate conformers by randomizing the rotatable torsion angles.
    """
    rng = np.random.default_rng(seed)
    conf_mol = Chem.Mol(mol)
    if AllChem.EmbedMolecule(conf_mol, maxAttempts=50) == -1:
        AllChem.EmbedMolecule(conf_mol, maxAttempts=20, useBasicKnowledge=False)
    bonds = _rotatable_bonds(conf_mol)
    for _ in range(num_confs):
        new_conf = Chem.Conformer(conf_mol.GetConformer(0))
        values = 2 * np.pi * rng.random(len(bonds))
        for dihedral, value in zip(bonds, values):
            rdMolTransforms.SetDihedralRad(new_conf, *dihedral, value)
        conf_mol.AddConformer(new_conf, assignId=True)
    conf_mol.RemoveConformer(0)
    return conf_mol


def remove_collisions(mol: Chem.Mol, thresh: float = 0.6) -> None:
    """
    Remove conformers in which any pair of atoms is closer than ``thresh``
    Angstroms.
    """
    bad = [
        c.GetId()
        for c in mol.GetConformers()
        if KDTree(c.GetPositions()).query_pairs(thresh)
    ]
    for cid in bad:
        mol.RemoveConformer(cid)


def _adjacency(mol: Chem.Mol) -> set[frozenset[int]]:
    """Sparse bond set including hydrogens."""
    adj: set[frozenset[int]] = set()
    for bond in mol.GetBonds():
        adj.add(frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
    return adj


def _conformer_adjacency(mol: Chem.Mol, conf: Chem.Conformer) -> set[frozenset[int]]:
    mol_copy = Chem.Mol(mol)
    mol_copy.RemoveAllConformers()
    mol_copy.AddConformer(Chem.Conformer(conf))
    rdDetermineBonds.DetermineConnectivity(mol_copy)
    return _adjacency(mol_copy)


def remove_graph_changed(mol: Chem.Mol) -> int:
    """Drop conformers whose inferred connectivity differs from the reference.

    Connectivity is always compared over **all atoms** (hydrogens included).
    Tautomers and zwitterionic forms will fail.
    """
    ref = _adjacency(mol)
    bad = [
        c.GetId() for c in mol.GetConformers() if _conformer_adjacency(mol, c) != ref
    ]
    for cid in bad:
        mol.RemoveConformer(cid)
    return len(bad)


def generate_conformers(mol: Chem.Mol, n_confs: int = 10, seed: int = 42) -> Chem.Mol:
    """Generate a diverse pool of conformers using four complementary strategies.

    Inspired by https://doi.org/10.48550/arXiv.2302.07061:

    1. ETKDGv3 + MMFF
    2. ETKDGv3 + random-coords + MMFF
    3. ETKDGv3 without MMFF                       (n/4)
    4. Uniform random-torsion sampling            (n/4)

    Then prunes atom collisions and connectivity-changed conformers.
    """
    out = Chem.Mol(mol)
    out.RemoveAllConformers()
    sources = [
        _generate_rdkit_confs(mol, n_confs, seed, ff_opt=True),
        _generate_rdkit_confs(mol, n_confs, seed, ff_opt=True, random_coords=True),
        _generate_rdkit_confs(mol, max(1, n_confs // 4), seed, ff_opt=False),
        _generate_random_torsion_confs(mol, max(1, n_confs // 4), seed),
    ]
    for src in sources:
        for conf in src.GetConformers():
            out.AddConformer(conf, assignId=True)
    remove_collisions(out)
    remove_graph_changed(out)
    return out


# ---------------------------------------------------------------------------
# 2. ASE / RDKit interop
# ---------------------------------------------------------------------------


def mol_to_atoms(mol: Chem.Mol) -> list[Atoms]:
    numbers = [a.GetAtomicNum() for a in mol.GetAtoms()]
    return [
        Atoms(numbers=numbers, positions=c.GetPositions()) for c in mol.GetConformers()
    ]


def _apply_results(mol: Chem.Mol, results: list[dict | None]) -> None:
    """
    Write per-conformer relaxation results (positions + energy) back into
    ``mol`` in place. Conformers whose result is ``None`` (failed relax /
    single-point) are dropped from ``mol``.
    """
    bad = []
    for conf, res in zip(list(mol.GetConformers()), results):
        if res is None:
            bad.append(conf.GetId())
            continue
        conf.SetPositions(np.asarray(res["positions"]))
        conf.SetDoubleProp(ENERGY_LABEL, float(res["energy"]))
    for cid in bad:
        mol.RemoveConformer(cid)


# ---------------------------------------------------------------------------
# 3. MLIP relaxation (single GPU, sequential; delegated to fastcsp core)
# ---------------------------------------------------------------------------


def relax_conformers(
    mol: Chem.Mol,
    optimize: bool,
    relax_cfg: dict,
    calc,
    traj_dir: Path | None = None,
) -> None:
    """Sequentially relax (or single-point) every conformer in-place.

    Uses ``relax_atoms`` from ``fairchem.applications.fastcsp.core.workflow.relax``
    so optimizer dispatch, trajectory writing, ``fix_symmetry`` /
    ``relax_cell`` handling all match the rest of fastcsp. Failed conformers
    are removed from ``mol``.
    """
    atoms_list = mol_to_atoms(mol)
    if not atoms_list:
        return
    charge = Chem.GetFormalCharge(mol)

    if traj_dir is not None:
        traj_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict | None] = []
    desc = "relax" if optimize else "single-point"
    for i, atoms in enumerate(tqdm(atoms_list, desc=desc)):
        atoms.info = {"spin": 1, "charge": charge}
        if optimize and traj_dir is not None:
            atoms.info["traj_path"] = str(traj_dir / f"conf_{i:04d}.traj")
        try:
            if optimize:
                atoms = relax_atoms(atoms, relax_cfg, calc)  # noqa: PLW2901
                energy_ev = float(atoms.info["energy"])
            else:
                atoms.calc = calc
                energy_ev = float(atoms.get_potential_energy())
            results.append(
                {
                    "positions": atoms.get_positions().tolist(),
                    "energy": energy_ev * EV_TO_KJMOL,
                }
            )
        except Exception as exc:
            print(f"  conf {i}: {desc} failed: {exc}", file=sys.stderr)
            results.append(None)

    _apply_results(mol, results)


# ---------------------------------------------------------------------------
# 4. Clustering / filtering
# ---------------------------------------------------------------------------


def get_energy(mol: Chem.Mol, conf_id: int) -> float:
    try:
        return mol.GetConformer(conf_id).GetDoubleProp(ENERGY_LABEL)
    except (SystemError, KeyError):
        return 0.0


def cluster_conformers(
    mol: Chem.Mol,
    rmsd_thresh: float = 0.25,
    energy_thresh: float = 1.5,
    include_hydrogens: bool = True,
    ref_stereo: dict | None = None,
) -> Chem.Mol:
    """Butina-cluster on best-RMSD, gated by an energy window.

    Pairs with an energy gap >= ``energy_thresh`` are assigned a sentinel
    large distance (and so are not merged). All conformers must already
    have a recorded energy. Set ``include_hydrogens=False`` to compute
    RMSD over heavy atoms only.

    If ``ref_stereo`` is provided, the per-cluster representative is the
    first **stereo-correct** member (CIP signature matches
    ``ref_stereo`` at all SMILES-assigned centers). If every member of a
    cluster is stereo-flipped, the original Butina centroid (``clust[0]``)
    is kept.
    """
    n = mol.GetNumConformers()
    if n <= 1:
        return mol
    ids = [c.GetId() for c in mol.GetConformers()]
    rmsd_mol = mol if include_hydrogens else Chem.RemoveHs(mol)
    dists: list[float] = []
    for i, c1 in enumerate(ids):
        for c2 in ids[:i]:
            if abs(get_energy(mol, c1) - get_energy(mol, c2)) < energy_thresh:
                dists.append(rdMolAlign.GetBestRMS(rmsd_mol, rmsd_mol, c1, c2))
            else:
                dists.append(1000.0)
    clusters = Butina.ClusterData(
        dists, n, rmsd_thresh, isDistData=True, reordering=True
    )

    # Prefer the stereo-correct member of each cluster.
    # If a cluster has no stereo-correct member, keep Butina's centroid.
    stereo_ok: dict[int, bool] = {}
    if ref_stereo:
        labels = _atom_labels(mol)
        for cid in ids:
            try:
                stereo_ok[cid] = not _stereo_diff(
                    ref_stereo,
                    _stereo_signature_from_3d(mol, cid),
                    labels,
                )
            except (ValueError, RuntimeError, KeyError) as exc:
                print(
                    f"  conf {cid}: stereo signature failed: {exc}",
                    file=sys.stderr,
                )
                stereo_ok[cid] = True  # signature failed: don't penalize

    def _pick_rep(clust: tuple[int, ...]) -> int:
        for i in clust:  # Butina order: centroid is clust[0]
            if stereo_ok.get(ids[i], True):
                return i
        # If no stereo-correct member exists, fall back:
        return clust[0]

    out = Chem.Mol(mol)
    out.RemoveAllConformers()
    for clust in clusters:
        out.AddConformer(mol.GetConformer(ids[_pick_rep(clust)]), assignId=True)
    return out


def remove_high_energy(mol: Chem.Mol, window: float) -> int:
    if mol.GetNumConformers() == 0:
        return 0
    emin = min(get_energy(mol, c.GetId()) for c in mol.GetConformers())
    bad = [
        c.GetId()
        for c in mol.GetConformers()
        if get_energy(mol, c.GetId()) - emin > window
    ]
    for cid in bad:
        mol.RemoveConformer(cid)
    return len(bad)


# ---------------------------------------------------------------------------
# 4b. Stereochemistry tagging
# ---------------------------------------------------------------------------


def _bond_stereo_codes(mol: Chem.Mol) -> dict[tuple[str, int, int], str]:
    """Bond-level CIP codes (E/Z, M/P) keyed by sorted atom pair."""
    return {
        ("bond", *sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx()))): code
        for b in mol.GetBonds()
        if (code := b.GetPropsAsDict().get("_CIPCode")) is not None
    }


def _stereo_signature_from_2d(mol: Chem.Mol) -> dict:
    """Modern CIP signature from atom/bond stereo tags (SMILES-derived).

    Atom keys are ``int`` (atom index); bond keys are ``("bond", i, j)``
    with ``i < j``. Only assigned centers/bonds are returned. Uses
    ``FindMolChiralCenters(useLegacyImplementation=False, force=True)`` so
    non-tetrahedral centers and atropisomers (``M`` / ``P`` on bonds) are
    handled.
    """
    m = Chem.Mol(mol)
    Chem.AssignStereochemistry(m, cleanIt=True, force=True)
    atom_codes = dict(
        Chem.FindMolChiralCenters(
            m,
            useLegacyImplementation=False,
            force=True,
            includeUnassigned=False,
        )
    )
    return {**atom_codes, **_bond_stereo_codes(m)}


def _stereo_signature_from_3d(mol: Chem.Mol, conf_id: int) -> dict:
    """Per-conformer CIP signature inferred from 3D coords.

    Includes **unassigned** centers (``includeUnassigned=True``) so the
    CSV captures chirality emerging from the embedding for atoms the
    SMILES did not constrain. Compare via :func:`_stereo_diff`, which
    only flags changes on keys assigned in the SMILES.
    """
    m = Chem.Mol(mol, False, confId=conf_id)
    Chem.AssignStereochemistryFrom3D(m)
    atom_codes = dict(
        Chem.FindMolChiralCenters(
            m,
            useLegacyImplementation=False,
            force=True,
            includeUnassigned=True,
        )
    )
    return {**atom_codes, **_bond_stereo_codes(m)}


def _atom_labels(mol: Chem.Mol) -> dict[int, str]:
    """``{atom_idx: "C3"}`` map for human-readable stereo keys."""
    return {a.GetIdx(): f"{a.GetSymbol()}{a.GetIdx()}" for a in mol.GetAtoms()}


def _atom_indexed_smiles(mol: Chem.Mol) -> str:
    """SMILES with each atom's index as its atom map number.

    Pasting this into any RDKit/Marvin/ChemDraw viewer shows the indices
    used by the ``stereo_*`` columns (e.g. ``C3``, ``N7``).
    """
    m = Chem.Mol(mol)
    for a in m.GetAtoms():
        a.SetAtomMapNum(a.GetIdx())
    return Chem.MolToSmiles(m)


def _label(key, labels: dict[int, str]) -> str:
    if isinstance(key, int):
        return labels.get(key, f"atom{key}")
    _, i, j = key
    return f"{labels.get(i, f'atom{i}')}-{labels.get(j, f'atom{j}')}"


def _format_signature(sig: dict, labels: dict[int, str]) -> str:
    """Compact ``label:code`` rendering for one signature dict."""
    return ";".join(f"{_label(k, labels)}:{sig[k]}" for k in sorted(sig, key=str))


def _stereo_diff(ref: dict, cur: dict, labels: dict[int, str]) -> list[str]:
    """Changes vs. ``ref``, scoped to keys present in ``ref``.

    Centers/bonds that the SMILES left unassigned (and so are absent from
    ``ref``) are NOT counted as changes — their 3D-resolved labels still
    show up in the ``stereo_signature`` column for the user to inspect.
    """
    return [
        f"{_label(k, labels)}:{ref[k] or '-'}->{cur.get(k) or '-'}"
        for k in sorted(ref, key=str)
        if ref[k] != cur.get(k)
    ]


# ---------------------------------------------------------------------------
# 5. Artifacts
# ---------------------------------------------------------------------------


def _write_conformer(
    mol: Chem.Mol,
    idx: int,
    conf_id: int,
    out_dir: Path,
    prefix: str,
    output_format: str,
    pad: int,
) -> None:
    """Write one conformer as xyz / mol / sdf.

    mol/sdf preserve bonding; xyz embeds the energy in the comment line.
    """
    name = f"{prefix}conf_{idx:0{pad}d}.{output_format}"
    fp = out_dir / name
    if output_format == "xyz":
        block = Chem.MolToXYZBlock(mol, confId=conf_id)
        if mol.GetConformer(conf_id).HasProp(ENERGY_LABEL):
            e = get_energy(mol, conf_id)
            block = block.replace("\n\n", f"\nenergy = {e} kJ/mol\n", 1)
        fp.write_text(block)
    elif output_format == "mol":
        Chem.MolToMolFile(mol, str(fp), confId=conf_id)
    elif output_format == "sdf":
        with Chem.SDWriter(str(fp)) as w:
            w.write(mol, confId=conf_id)
    else:
        raise ValueError(f"Unsupported output_format: {output_format!r}")


def write_artifacts(
    mol: Chem.Mol,
    out_dir: Path,
    prefix: str,
    output_format: str = "xyz",
    ref_stereo: dict | None = None,
) -> None:
    """Write energy-sorted conformer files and a ``<prefix>confs.csv`` summary.

    Lowest-energy conformer is ``conf_0``. Filename index is zero-padded to
    fit the conformer count (e.g. ``conf_00``, ``conf_000`` for >=10 / >=100).
    The CSV includes both absolute (``energy``) and ``relative_energy``
    values vs. the lowest-energy conformer (in kJ/mol throughout).

    If ``ref_stereo`` is given (a signature from
    :func:`_stereo_signature_from_2d`), every conformer is tagged with
    ``stereo_changed`` (bool) and ``stereo_diff`` (semicolon-joined list of
    atom/bond CIP changes vs. the reference). Flipped conformers are
    **not** dropped — just tagged.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    conf_order = [
        c.GetId()
        for c in sorted(
            mol.GetConformers(),
            key=lambda x: x.GetDoubleProp(ENERGY_LABEL)
            if x.HasProp(ENERGY_LABEL)
            else 0.0,
        )
    ]
    n = len(conf_order)
    pad = max(2, len(str(max(n - 1, 0))))

    energies = [get_energy(mol, cid) for cid in conf_order]
    e_ref = min(energies) if energies else 0.0
    labels = _atom_labels(mol) if ref_stereo is not None else {}

    rows = []
    for idx, cid in enumerate(conf_order):
        e = energies[idx]
        row = {
            "idx": idx,
            "prefix": prefix.rstrip("_"),
            "conf_id": cid,
            "energy": e,
            "relative_energy": e - e_ref,
        }
        if ref_stereo is not None:
            try:
                cur = _stereo_signature_from_3d(mol, cid)
                diffs = _stereo_diff(ref_stereo, cur, labels)
                row["stereo_signature"] = _format_signature(cur, labels)
                row["stereo_changed"] = bool(diffs)
                row["stereo_diff"] = ";".join(diffs)
            except (ValueError, RuntimeError, KeyError) as exc:
                print(
                    f"  conf {cid}: stereo signature failed: {exc}",
                    file=sys.stderr,
                )
                row["stereo_signature"] = ""
                row["stereo_changed"] = None
                row["stereo_diff"] = ""
        rows.append(row)
        _write_conformer(
            mol,
            idx,
            cid,
            out_dir,
            prefix,
            output_format=output_format,
            pad=pad,
        )
    pd.DataFrame(rows).to_csv(out_dir / f"{prefix}confs.csv", index=False)


# ---------------------------------------------------------------------------
# 6. Mol construction + end-to-end per-molecule
# ---------------------------------------------------------------------------


def smiles_to_mol(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Cannot parse SMILES: {smiles!r}")
    return Chem.AddHs(mol)


def _select_for_fastcsp(
    name: str,
    n_target: int,
    gen_dir: Path,
    rel_dir: Path,
    sel_dir: Path,
    output_format: str,
) -> tuple[int, int]:
    """Populate ``conformers_fastcsp/<name>/`` with up to ``n_target`` files.

    Relaxed conformers (lower index = lower energy) are preferred. Generated
    conformers are used **only as a complete fallback** when no relaxed
    conformers exist (e.g. tiny rigid molecules where every relaxation
    failed connectivity/energy filters). We do not top up a partial relaxed
    set with generated conformers, since mixing relaxed and unrelaxed
    geometries produces redundancy.

    Each copied file keeps its original stem and gains a ``_relaxed`` /
    ``_generated`` suffix before the extension, so origin is unambiguous
    downstream. Returns ``(n_relaxed, n_generated)``.
    """
    relaxed_files = sorted(rel_dir.glob(f"*_conf_*.{output_format}"))[:n_target]
    if relaxed_files:
        generated_files: list[Path] = []
    else:
        generated_files = sorted(gen_dir.glob(f"*_conf_*.{output_format}"))[:n_target]
    chosen = [(f, "relaxed") for f in relaxed_files] + [
        (f, "generated") for f in generated_files
    ]
    if chosen:
        sel_dir.mkdir(parents=True, exist_ok=True)
        for src, origin in chosen:
            dst = sel_dir / f"{src.stem}_{origin}{src.suffix}"
            shutil.copy2(src, dst)
    return len(relaxed_files), len(generated_files)


def process_molecule(
    mol: Chem.Mol,
    name: str,
    output_dir: str | Path,
    rdkit_cfg: dict,
    relax_cfg: dict,
    select_for_fastcsp: int | None = None,
) -> dict:
    """Run the full workflow for one molecule.

    ``rdkit_cfg`` and ``relax_cfg`` should already be merged with
    ``CONF_GEN_DEFAULTS`` / ``RELAX_DEFAULTS`` (see ``_process_one``).

    Two output directories are always written under ``<output_dir>/<name>/``:

    * ``conformers_generated/`` — RDKit-generated geometries with UMA
      single-point energies.
    * ``conformers_relaxed/``   — UMA-relaxed geometries (post-connectivity,
      post-cluster, post-energy-window).

    If ``select_for_fastcsp`` is given (>0), an additional folder
    ``conformers_fastcsp/<name>/`` is populated with up to that many
    conformers, drawn from the relaxed pool. Generated conformers are
    used only as a complete fallback when the relaxed pool is empty
    (no mixing of relaxed and unrelaxed geometries). Filenames carry
    a ``_relaxed`` or ``_generated`` suffix.

    Returns a JSON-serialisable summary.
    """
    output_dir = Path(output_dir)
    gen_dir = output_dir / "conformers_generated" / name
    rel_dir = output_dir / "conformers_relaxed" / name
    prefix = f"{name}_"
    traj_dir = (
        output_dir / "trajectories" / name if relax_cfg.get("write_traj") else None
    )
    summary: dict = {"name": name, "start": time.time()}

    # Skip if either output folder already exists and is non-empty — lets
    # users resume an interrupted run without recomputing finished molecules.
    if (gen_dir.is_dir() and any(gen_dir.iterdir())) or (
        rel_dir.is_dir() and any(rel_dir.iterdir())
    ):
        summary["status"] = "skipped_existing"
        print(f"[{name}] already has outputs, skipping")
        return summary

    # Create per-molecule output dirs eagerly so failures (e.g., all conformers
    # drop on connectivity check) still leave a discoverable directory.
    gen_dir.mkdir(parents=True, exist_ok=True)
    rel_dir.mkdir(parents=True, exist_ok=True)

    # Reference stereo signature from SMILES (computed once; passed to
    # write_artifacts so per-conformer flips show up in the CSVs).
    ref_stereo = _stereo_signature_from_2d(mol)
    summary["indexed_smiles"] = _atom_indexed_smiles(mol)
    labels = _atom_labels(mol)
    summary["reference_stereo"] = _format_signature(ref_stereo, labels)
    print(f"[{name}] indexed SMILES: {summary['indexed_smiles']}")
    if ref_stereo:
        print(f"[{name}] reference stereo: {summary['reference_stereo']}")

    # --- 1. Generate ----------------------------------------------------
    adj_n = math.ceil(rdkit_cfg["initial_pool_size"] / 2.5)
    conf_mol = generate_conformers(mol, n_confs=adj_n, seed=rdkit_cfg["seed"])
    summary["generated"] = conf_mol.GetNumConformers()
    print(f"[{name}] generated {summary['generated']} conformers")

    # Pre-relax RMSD cluster (geometry-only; no energy gate yet — all pairs
    # have 0 "energy" so the gate is satisfied and falls through to RMSD).
    # Cuts the UMA single-point + relax cost for near-duplicate seeds.
    conf_mol = cluster_conformers(
        conf_mol,
        rmsd_thresh=rdkit_cfg["rmsd_thresh"],
        energy_thresh=rdkit_cfg["cluster_energy_thresh"],
        include_hydrogens=rdkit_cfg["include_hydrogens"],
    )
    summary["after_precluster"] = conf_mol.GetNumConformers()
    print(f"[{name}] {summary['after_precluster']} distinct after pre-relax cluster")
    if conf_mol.GetNumConformers() == 0:
        summary["status"] = "no_conformers"
        return summary

    # --- 2. Build calculator once, then single-point + relax -----------
    calc = create_calculator(relax_cfg)

    generated_mol = Chem.Mol(conf_mol)
    relax_conformers(generated_mol, optimize=False, relax_cfg=relax_cfg, calc=calc)
    summary["after_sp"] = generated_mol.GetNumConformers()
    write_artifacts(
        generated_mol,
        gen_dir,
        prefix,
        output_format=rdkit_cfg["output_format"],
        ref_stereo=ref_stereo,
    )
    print(
        f"[{name}] wrote {summary['after_sp']} generated conformers "
        f"(single-point) to conformers_generated/{name}/"
    )

    relaxed_mol = Chem.Mol(conf_mol)
    relax_conformers(
        relaxed_mol,
        optimize=True,
        relax_cfg=relax_cfg,
        calc=calc,
        traj_dir=traj_dir,
    )
    # Connectivity check (all atoms incl. H, even for zwitterions)
    n_changed = remove_graph_changed(relaxed_mol)
    if n_changed:
        print(
            f"[{name}] dropped {n_changed} relaxed conformers with changed connectivity"
        )
    summary["after_relax"] = relaxed_mol.GetNumConformers()
    if relaxed_mol.GetNumConformers() == 0:
        summary["status"] = "all_failed"
        return summary

    # --- 3. Cluster + energy window on relaxed -------------------------
    relaxed_mol = cluster_conformers(
        relaxed_mol,
        rmsd_thresh=rdkit_cfg["rmsd_thresh"],
        energy_thresh=rdkit_cfg["cluster_energy_thresh"],
        include_hydrogens=rdkit_cfg["include_hydrogens"],
        ref_stereo=ref_stereo,
    )
    remove_high_energy(relaxed_mol, relax_cfg["energy_window"])
    summary["final"] = relaxed_mol.GetNumConformers()
    write_artifacts(
        relaxed_mol,
        rel_dir,
        prefix,
        output_format=rdkit_cfg["output_format"],
        ref_stereo=ref_stereo,
    )
    print(
        f"[{name}] wrote {summary['final']} relaxed conformers "
        f"to conformers_relaxed/{name}/"
    )

    # --- 4. Optional: copy a small subset for the downstream fastcsp run ---
    if select_for_fastcsp and select_for_fastcsp > 0:
        sel_dir = output_dir / "conformers_fastcsp" / name
        n_rel, n_gen = _select_for_fastcsp(
            name,
            int(select_for_fastcsp),
            gen_dir,
            rel_dir,
            sel_dir,
            rdkit_cfg["output_format"],
        )
        summary["selected_relaxed"] = n_rel
        summary["selected_generated"] = n_gen
        print(
            f"[{name}] selected {n_rel + n_gen}/{select_for_fastcsp} for fastcsp "
            f"({n_rel} relaxed + {n_gen} generated)"
        )

    summary["status"] = "ok"
    summary["elapsed_s"] = time.time() - summary["start"]
    return summary


# ---------------------------------------------------------------------------
# 7. SLURM submission — one task per molecule
# ---------------------------------------------------------------------------


def _process_one(
    row: dict,
    output_dir: str,
    rdkit_user: dict,
    relax_user: dict,
) -> dict:
    """Pickleable per-task worker: process one molecule.

    Defaults are merged and printed at the start of the task.
    """
    out_dir = Path(output_dir)
    name = str(row["name"])
    rdkit_row, relax_row = _split_row_overrides(row)
    rdkit_cfg, relax_cfg = _resolve_config(
        name,
        rdkit_user,
        relax_user,
        rdkit_row,
        relax_row,
    )
    sel_raw = row.get("select_for_fastcsp")
    select_n = None if pd.isna(sel_raw) else int(float(sel_raw))
    try:
        mol = smiles_to_mol(row["smiles"])
        summary = process_molecule(
            mol,
            name,
            out_dir,
            rdkit_cfg=rdkit_cfg,
            relax_cfg=relax_cfg,
            select_for_fastcsp=select_n,
        )
    except Exception as exc:
        summary = {
            "name": name,
            "status": "error",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
        print(f"[{name}] FAILED: {summary['error']}", file=sys.stderr)

    sum_dir = out_dir / "summaries"
    sum_dir.mkdir(parents=True, exist_ok=True)
    (sum_dir / f"{name}.json").write_text(json.dumps(summary, indent=2))
    return summary


def submit_confgen_jobs(
    input_csv: str | Path,
    output_dir: str | Path,
    rdkit_overrides: dict | None = None,
    relax_overrides: dict | None = None,
    config_path: str | Path | None = None,
    wait: bool = True,
) -> tuple[list, list[dict] | None]:
    """Submit a SLURM array — one task per molecule.

    SLURM parameters are read from the YAML at ``config_path`` via
    ``fastcsp.core.utils.slurm.get_slurm_config``. ``rdkit_overrides`` and
    ``relax_overrides`` are passed as-is and merged onto
    ``CONF_GEN_DEFAULTS`` / ``RELAX_DEFAULTS`` inside each worker (and
    printed there).

    If ``config_path`` is given, the YAML file is copied verbatim into
    ``<output_dir>/`` alongside the input CSV, job ids, and final summary
    — so the whole run is reproducible from one folder.

    Returns ``(jobs, results)``. ``results`` is ``None`` if ``wait=False``,
    else a list (one per task) of per-molecule summaries.
    """
    rdkit_overrides = dict(rdkit_overrides or {})
    relax_overrides = dict(relax_overrides or {})
    raw_config: dict = {}
    if config_path is not None:
        with open(config_path) as fh:
            raw_config = yaml.safe_load(fh) or {}
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir = output_dir / "slurm"  # submitit stdout/stderr only
    log_dir.mkdir(parents=True, exist_ok=True)

    input_csv = Path(input_csv).resolve()
    molecules_df = pd.read_csv(input_csv)
    missing = {"name", "smiles"} - set(molecules_df.columns)
    if missing:
        raise ValueError(
            f"Input CSV {input_csv} is missing required columns: {sorted(missing)}."
        )
    # Verbatim copy of the molecules CSV (preserves original filename + any
    # extra columns/comments the user had). Rewritten on every run.
    csv_dst = (output_dir / input_csv.name).resolve()
    if input_csv != csv_dst:
        if csv_dst.exists():
            print(f"[submit] overwriting existing {csv_dst}")
        shutil.copy2(input_csv, csv_dst)
    rows = molecules_df.to_dict("records")

    # Snapshot the user's config (verbatim) so the run is reproducible
    # from <output_dir>/.
    if config_path is not None:
        cfg_src = Path(config_path).resolve()
        cfg_dst = (output_dir / cfg_src.name).resolve()
        if cfg_src != cfg_dst:
            if cfg_dst.exists():
                print(f"[submit] overwriting existing {cfg_dst}")
            shutil.copy2(cfg_src, cfg_dst)

    executor_params = get_slurm_config(raw_config, "relax", "submitit_executor")
    out_dir_str = str(output_dir)
    job_args = [
        (_process_one, (row, out_dir_str, rdkit_overrides, relax_overrides), {})
        for row in rows
    ]
    jobs = submit_slurm_jobs(
        job_args,
        log_dir,
        job_name="fastcsp_confgen",
        **executor_params,
    )

    if not wait:
        return jobs, None

    wait_for_jobs(jobs)
    results: list[dict] = []
    failed = 0
    for j in jobs:
        try:
            results.append(j.result())
        except Exception as exc:
            failed += 1
            print(f"[{j.job_id}] FAILED at SLURM level: {exc}", file=sys.stderr)
            results.append(
                {"job_id": j.job_id, "status": "slurm_error", "error": str(exc)}
            )

    n_ok = sum(1 for s in results if s.get("status") == "ok")
    print(f"Done: {n_ok}/{len(results)} molecules ok, {failed} task-level failures.")
    (output_dir / "summary.json").write_text(json.dumps(results, indent=2))

    # Per-molecule selection counts (one row per molecule).
    selection_cols = (
        "name",
        "status",
        "generated",
        "after_precluster",
        "after_sp",
        "after_relax",
        "final",
        "selected_relaxed",
        "selected_generated",
        "elapsed_s",
    )
    pd.DataFrame([{k: s.get(k) for k in selection_cols} for s in results]).to_csv(
        output_dir / "selection_summary.csv", index=False
    )

    # Aggregate per-molecule CSVs into top-level combined files.
    for stage in ("conformers_generated", "conformers_relaxed"):
        per_mol = sorted((output_dir / stage).glob("*/*_confs.csv"))
        if per_mol:
            pd.concat([pd.read_csv(p) for p in per_mol], ignore_index=True).to_csv(
                output_dir / f"{stage}.csv",
                index=False,
            )

    return jobs, results


# ---------------------------------------------------------------------------
# 8. CLI (``fastcsp-confgen``)
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="fastcsp-confgen",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-c",
        "--config",
        required=True,
        help="YAML config with top-level 'root:', 'molecules:', 'rdkit:', "
        "'relax:', and 'slurm:' blocks. The run is submitted as a SLURM "
        "array — one task per molecule.",
    )
    args = parser.parse_args(argv)

    rdkit, relax, raw_config, root, molecules = load_conformer_generation_config(
        args.config
    )
    if root is None or molecules is None:
        raise SystemExit(
            f"--config {args.config} must define both 'root:' and 'molecules:'."
        )

    out = (root / "conformers").resolve()
    _, summaries = submit_confgen_jobs(
        molecules,
        out,
        rdkit_overrides=rdkit,
        relax_overrides=relax,
        config_path=args.config,
        wait=True,
    )
    print(json.dumps(summaries or [], indent=2))


if __name__ == "__main__":
    main()
