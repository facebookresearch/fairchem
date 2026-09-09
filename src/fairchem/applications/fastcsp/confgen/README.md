# `fastcsp-confgen` — SMILES → gas-phase conformers

Upstream helper for the FastCSP workflow. Reads SMILES from a
`molecules.csv`, submits **one SLURM task per molecule**, and writes
per-molecule directories of geometry files + a summary CSV.

Skip this stage entirely if you already have conformers from another
source — `fastcsp` only cares about the `conformers_path` column in its
own `molecules.csv`.

## Workflow (per molecule, in one SLURM task)

1. **Seed pool** — `generate_conformers` embeds ~`initial_pool_size`
   conformers using four complementary strategies (ETKDGv3 + MMFF,
   ETKDGv3 random-coords + MMFF, ETKDGv3 without MMFF, uniform
   random-torsion). Rejects atom-clash and connectivity-changed geometries.
2. **Pre-relax RMSD cluster** — Butina on best-RMSD only (energies are
   still zero at this point, so the energy gate falls through). Cheaply
   drops near-duplicate seeds before spending UMA compute on them.
3. **UMA single-point** on the pre-clustered pool → `conformers_generated/<name>/`.
4. **UMA relaxation** with the fastcsp `relax_atoms` driver (isolated
   molecule, `fix_symmetry=false`, `relax_cell=false`) → drops failed
   relaxes, then runs a connectivity check and drops relaxed conformers
   whose bond graph changed.
5. **Post-relax cluster + energy window** — Butina on best-RMSD gated by
   `cluster_energy_thresh`, then `energy_window` cap → `conformers_relaxed/<name>/`.
6. **Fastcsp subset** — if `select_for_fastcsp > 0` (see CSV column
   below), that many lowest-energy relaxed conformers are copied into
   `conformers_fastcsp/<name>/`. Falls back to `conformers_generated/`
   only if the relaxed pool is empty (no mixing).

Stereo is not a separate stage: a CIP signature is computed once from
the input SMILES and used to (a) tag every written conformer in
`*_confs.csv` (`stereo_changed`, `stereo_diff`) and (b) prefer
stereo-correct cluster representatives in step 5.

## CLI

```bash
fastcsp-confgen -c configs/example_config.yaml
```

See [`configs/example_config.yaml`](configs/example_config.yaml) for the
canonical, commented schema. YAML keys are exhaustively documented there;
this README does not repeat them.

## Per-molecule CSV overrides

Any column whose name matches a key in `CONF_GEN_DEFAULTS` or
`RELAX_DEFAULTS` (in [`main.py`](main.py)) becomes a per-row override:

| Column | Type | Default | Notes |
|---|---|---|---|
| `initial_pool_size` | int | 50 | Seed pool target before pruning |
| `seed` | int | 42 | RNG seed |
| `rmsd_thresh` | float | 0.25 | Butina cluster cutoff (Å) |
| `cluster_energy_thresh` | float | 1.5 | Skip RMSD if ΔE ≥ this (kJ/mol) |
| `include_hydrogens` | bool | true | RMSD over all atoms (else heavy only) |
| `output_format` | str | `xyz` | `xyz` \| `mol` \| `sdf` |
| `calculator` | str | `uma_sm_1p1_omol` | UMA task variant |
| `optimizer` | str | `BFGS` | `BFGS` \| `FIRE` \| `LBFGS` |
| `fmax` | float | 0.05 | eV/Å |
| `max_steps` | int | 100 | ASE relax step cap |
| `energy_window` | float | 40.0 | kJ/mol cap on relaxed pool |
| `select_for_fastcsp` | int | 0 | # conformers copied to `conformers_fastcsp/` |

Precedence: `defaults < YAML < CSV`. Every worker prints its final
resolved config at task start.

## Output layout

```
<root>/conformers/
├── conformers_generated/<name>/  conf_00.xyz, conf_01.xyz, ..., generated_confs.csv
├── conformers_relaxed/<name>/    conf_00.xyz, conf_01.xyz, ..., relaxed_confs.csv
├── conformers_fastcsp/<name>/    curated subset (only if select_for_fastcsp>0)
├── <config-name>.yaml            copy of the config used
└── <molecules-name>.csv          copy of the input CSV
```

Reruns are resumable — molecules with a non-empty `conformers_generated/`
or `conformers_relaxed/` subdir are skipped.

## Per-conformer CSV columns (`*_confs.csv`)

| Column | Notes |
|---|---|
| `idx` | 0-based rank by relaxed energy (0 = lowest) |
| `prefix` | `generated` or `relaxed` |
| `conf_id` | RDKit internal conformer id |
| `energy` | kJ/mol, absolute (UMA total energy for isolated conformer) |
| `relative_energy` | kJ/mol vs. lowest-energy conformer |
| `stereo_signature` | Per-atom/bond CIP signature from 3D |
| `stereo_changed` | `True` if signature differs from SMILES reference |
| `stereo_diff` | Semicolon-joined list of flipped centers |
