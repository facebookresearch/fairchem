# Conformer generation (`fastcsp-confgen`)

End-to-end conformer generation for FastCSP. Per molecule:

1. **Generate** a diverse pool with RDKit (ETKDGv3 + MMFF, random-coords,
   no-MMFF, random-torsion). Atom-collision and connectivity-changed
   conformers are dropped.
2. **Pre-cluster** on best-RMSD (Butina) — geometry only, before paying
   for any MLIP cost. All energies are 0 at this point, so the energy
   gate is inactive and clustering is purely geometric. RMSD basis is
   all atoms by default (`include_hydrogens: true`); set
   `include_hydrogens: false` to use heavy atoms only.
3. **Single-point + relax** every survivor with a FAIR-Chem UMA MLIP via
   `fastcsp.core.workflow.relax` (calculator, optimizer, trajectory
   writing all delegated there).
4. **Filter**: drop conformers whose covalent graph changed during
   relaxation; tag (do not drop) conformers whose stereochemistry flipped.
5. **Post-cluster + energy window**: Butina on relaxed geometries with an
   energy gate (see [Clustering details](#clustering-details) below),
   then prune above `energy_window` kJ/mol.
6. **Write** per-conformer geometry files + a `confs.csv` summary.

Every run is submitted as a **SLURM array** — one task per molecule.

## CLI

```bash
fastcsp-confgen -c configs/example_config.yaml
```

A ready-to-edit pair of files is in [`configs/`](configs/):

- [`configs/example_config.yaml`](configs/example_config.yaml) — fully
  annotated YAML with all `rdkit:` / `relax:` / `slurm:` blocks.
- [`configs/example_molecules.csv`](configs/example_molecules.csv) — a
  small heterogeneous molecule set demonstrating every supported
  per-row override column.

The YAML carries:

- `root:` — output base. Outputs go to `<root>/conformers/`; the config
  and molecules CSV are copied there for reproducibility.
- `molecules:` — CSV with **required** `name` and `smiles` columns, plus
  any of the **optional per-molecule override columns** listed in
  [Per-molecule CSV overrides](#per-molecule-csv-overrides) below. `name`
  is used as the per-molecule subfolder. Path is resolved as absolute,
  relative to the config file, or relative to `root`.
- `rdkit:` — conformer generation + clustering + output options (used as
  *defaults* for any per-molecule column left blank).
- `relax:` — MLIP relaxation + energy window (same default-only role).
- `slurm:` — submitit/SLURM settings for the per-molecule array job
  (resolved via `fastcsp.core.utils.slurm.get_slurm_config`).

Precedence for any setting: `CONF_GEN_DEFAULTS` / `RELAX_DEFAULTS` (lowest)
< YAML `rdkit:` / `relax:` block < per-row CSV column (highest).
The worker prints both the YAML-level and the CSV-level overrides at
the start of its task.

## Per-molecule CSV overrides

In addition to the required `name` and `smiles` columns, the molecules
CSV may carry any of the following **optional** columns. Each one
overrides the corresponding `rdkit:` / `relax:` key for that single
molecule. Empty / NaN cells fall back to the YAML default.

| column | block | default | meaning |
| --- | --- | --- | --- |
| `initial_pool_size` | rdkit | 50 | target size of the RDKit seed pool before any filtering (scaled internally by 1/2.5 before embedding) |
| `seed` | rdkit | 42 | ETKDG random seed |
| `rmsd_thresh` | rdkit | 0.25 | Butina cluster radius (Å) |
| `cluster_energy_thresh` | rdkit | 1.5 | energy gate inside Butina (kJ/mol) |
| `include_hydrogens` | rdkit | true | include hydrogens in best-RMSD (bool; `true/false/1/0/yes/no`) |
| `output_format` | rdkit | `xyz` | `xyz`, `mol`, or `sdf` |
| `calculator` | relax | `uma_sm_1p1_omol` | FAIR-Chem MLIP |
| `optimizer` | relax | `BFGS` | ASE optimizer name |
| `fmax` | relax | 0.05 | relaxation force threshold (eV/Å) |
| `max_steps` | relax | 100 | optimizer step cap |
| `write_traj` | relax | false | dump ASE trajectory per conformer |
| `traj_interval` | relax | 1 | trajectory write interval |
| `energy_window` | relax | 40.0 | post-relax energy cap (kJ/mol above min) |
| `select_for_fastcsp` | — | *(absent)* | if set, copy this many conformers into `conformers_fastcsp/<name>/`, preferring relaxed and topping up from generated. If the column is absent from the CSV or the cell is blank for a molecule, no selection is done for that row. |

Example mixed CSV (rigid molecules left at defaults, flexible ones
get a wider pool and tighter RMSD; `select_for_fastcsp` controls how
many conformers are mirrored into `conformers_fastcsp/<name>/`) — see
[`configs/example_molecules.csv`](configs/example_molecules.csv) for
the full file:

```csv
name,smiles,initial_pool_size,rmsd_thresh,cluster_energy_thresh,max_steps,energy_window,select_for_fastcsp
aspirin,CC(=O)Oc1ccccc1C(=O)O,500,0.20,2.0,150,40.0,6
ibuprofen,CC(C)Cc1ccc(cc1)C(C)C(=O)O,1200,0.20,1.5,200,50.0,15
caffeine,Cn1cnc2c1c(=O)n(C)c(=O)n2C,50,0.25,1.5,100,40.0,1
```

This lets a single YAML drive a heterogeneous molecule set without
per-class config files. The matching YAML in this case carries only the
calculator and slurm blocks; rdkit/relax can be empty.

## Library

```python
from fairchem.applications.fastcsp.confgen.main import (
    process_molecule,
    generate_conformers,
    relax_conformers,
    submit_confgen_jobs,
)

rdkit = {"initial_pool_size": 250, "include_hydrogens": False}
relax = {"calculator": "uma_sm_1p1_omol"}
full_cfg = {
    "rdkit": rdkit,
    "relax": relax,
    "slurm": {
        "partition": "learnfair",
        "array_parallelism": 64,
        "timeout_min": 240,
        "gpus_per_node": 1,
        "cpus_per_task": 8,
        "mem_gb": 32,
    },
}
jobs, results = submit_confgen_jobs(
    "molecules.csv",
    "runs/2026_conformers",
    rdkit=rdkit,
    relax=relax,
    config=full_cfg,
    wait=True,
)
```

## Output layout

```
<root>/conformers/
  <molecules>.csv               # copy of input CSV (reproducibility)
  <config>.yaml                 # copy of the resolved config
  summary.json                  # aggregated per-molecule summary
  conformers_generated.csv      # aggregated per-conformer CSV (single-point)
  conformers_relaxed.csv        # aggregated per-conformer CSV (relaxed)
  conformers_generated/
    <name>/
      <name>_confs.csv          # one row per conformer, sorted by energy
      <name>_conf_00.xyz        # (or .mol / .sdf — see `rdkit.output_format`)
      ...
  conformers_relaxed/
    <name>/
      <name>_confs.csv
      <name>_conf_00.xyz
      ...
  conformers_fastcsp/            # only if `select_for_fastcsp` is set
    <name>/                      #   for at least one molecule; one file
      <name>_conf_00_relaxed.xyz #   per selected conformer, suffix records
      <name>_conf_01_relaxed.xyz #   the source pool. Relaxed conformers
      <name>_conf_02_generated.xyz # are picked first; the remainder is
      ...                        #   topped up from generated.
  trajectories/                 # only if `relax.write_traj: true`
    <name>/
      conf_0000.traj            # ASE trajectory per relaxed conformer
      ...
  summaries/
    <name>.json                 # per-molecule summary (counts, stereo ref, ...)
  slurm/                        # submitit stdout/stderr + pickles
```

Per-molecule `summaries/<name>.json` records `indexed_smiles` (SMILES
with atom-map indices matching the `stereo_*` column keys) and
`reference_stereo` (the SMILES-derived CIP signature used to tag
post-relax stereo flips), alongside conformer counts at each stage.

Per-molecule directories under `conformers_generated/` and
`conformers_relaxed/` are created eagerly, so molecules that fail (e.g.
all relaxed conformers dropped on connectivity check) still leave a
discoverable empty folder.

Resumability: a molecule with a **non-empty** `conformers_generated/<name>/`
or `conformers_relaxed/<name>/` is skipped, so reruns over an existing
output directory pick up where they stopped.

## Per-conformer CSV columns

Each `<name>_confs.csv` has:

| column | meaning |
| --- | --- |
| `idx` | rank by energy (0 = lowest) |
| `prefix` | molecule name (matches the per-molecule subfolder) |
| `conf_id` | RDKit conformer id |
| `energy` | absolute energy (kJ/mol) |
| `relative_energy` | vs. lowest in this set |
| `stereo_signature` | atom/bond CIP signature inferred from 3D coords |
| `stereo_changed` | bool vs. reference (from input SMILES) |
| `stereo_diff` | semicolon-joined list of flipped atoms/bonds |

The top-level `conformers_generated.csv` / `conformers_relaxed.csv`
are the per-molecule `<name>_confs.csv` files concatenated across all
molecules (the `prefix` column lets you split them back per molecule).

## Clustering details

`cluster_conformers` (in [`main.py`](main.py)) runs Butina on a pairwise
best-RMSD distance matrix with an **energy gate**:

- For every pair of conformers, if `|ΔE| < cluster_energy_thresh`
  (default **1.5 kJ/mol**) the distance is the real best-fit RMSD
  (heavy-atom only if `include_hydrogens: false`). Otherwise the
  distance is set to a sentinel `1000.0` so Butina will never merge
  the pair, no matter how geometrically similar.
- Butina (`reordering=True`) then groups conformers within
  `rmsd_thresh` (default **0.25 Å**). Inside any cluster, all members
  are within `rmsd_thresh` *and* within `cluster_energy_thresh` of each
  other.

**Cluster representative tie-break** (`_pick_rep`):

1. Walk the cluster in Butina order (centroid = most-connected member
   first, then second-most-connected, ...).
2. Return the **first stereo-correct** member — i.e. its CIP signature
   inferred from 3D coordinates matches the SMILES-derived reference at
   every SMILES-assigned center. Atoms left unassigned in the SMILES
   never count as flipped (their 3D-resolved labels are still written
   to `stereo_signature` for inspection).
3. If **no** member is stereo-correct, fall back to the centroid
   (`clust[0]`); the conformer is flagged `stereo_changed=True` and the
   flip is recorded in `stereo_diff`.

Energy is used **only in the distance matrix gate**, never as a
representative-selection tie-break. Since every member of a cluster is
within `cluster_energy_thresh` of every other member, the energy
spread inside a cluster is bounded (< 1.5 kJ/mol by default) — picking
the centroid vs. the lowest-energy member changes downstream results
negligibly.

The pre-relax call passes `ref_stereo=None`, so step 2 is dormant and
the centroid is always chosen — stereo-flipped seeds may survive
pre-clustering, but they're tagged (and possibly demoted) at the
post-relax pass.
