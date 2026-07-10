# FastCSP end-to-end example

Complete SMILES → predicted crystal-structure landscape workflow for two
molecular crystals from the FastCSP JACS test set — **XULDUD**
(semi-rigid bicyclic oxaindene, 2 experimental polymorphs) and **WIDBAO**
(flexible thiazoline-thione, 1 polymorph) — suitable as a smoke test or
a copy-paste starting point.

## Files

| File | Purpose |
|---|---|
| [`molecules.csv`](molecules.csv) | Shared input: `name,smiles,conformers_path,refcode,z,spg`. `fastcsp-confgen` reads `name` + `smiles` (extras ignored); `fastcsp` reads `name,conformers_path,refcode,z,spg` (extras ignored). |
| [`confgen_config.yaml`](confgen_config.yaml) | Conformer generation (SMILES → per-molecule XYZs) |
| [`fastcsp_config.yaml`](fastcsp_config.yaml) | 7-stage CSP workflow |
| [`genarris_base.conf`](genarris_base.conf) | Genarris template referenced from `fastcsp_config.yaml` |

## Placeholders you must edit

Both YAML configs have angle-bracket placeholders — none of the paths are
portable as-is.

- `<PROJECT_ROOT>` — absolute path where all outputs land. Set the same
  value in `confgen_config.yaml` and `fastcsp_config.yaml` so the workflow
  can find `<PROJECT_ROOT>/conformers/conformers_fastcsp/<name>/*.xyz`
  (that path matches the `conformers_path` column in `molecules.csv`).
- `<PATH_TO_GENARRIS>` — env with [Genarris 3.0](https://github.com/Yi5817/Genarris)
  installed (used only in the `generate` stage).
- `<PATH_TO_MPIRUN>` — MPI launcher visible to the Genarris env.
- `<PATH_TO_EXPERIMENTAL_CIFS>` — directory of `<refcode>.cif` files
  (needed for the `evaluate` stage; skip if you don't have experimental
  references).
- `<YOUR_PARTITION>` — SLURM partition(s) for each stage's `slurm:` block.
- (Optional) `<PATH_TO_CSD_PYTHON>` — env with the CCDC Python API, only
  if you switch `evaluate.method` from `pymatgen` to `csd`.

## Workflow

### 1. Generate starting conformers from SMILES

```bash
fastcsp-confgen -c example/confgen_config.yaml
```

Reads [`molecules.csv`](molecules.csv) (only the `name` + `smiles`
columns are needed at this stage; the optional `select_for_fastcsp`
column controls how many conformers per molecule are mirrored into the
``conformers_fastcsp/`` subdir), submits one SLURM task per molecule,
and writes:

```
<PROJECT_ROOT>/conformers/
├── conformers_fastcsp/                  # curated subset per select_for_fastcsp
│   ├── XULDUD/
│   │   └── XULDUD_conf_00_relaxed.xyz
│   └── WIDBAO/
│       ├── WIDBAO_conf_00_relaxed.xyz
│       └── WIDBAO_conf_01_relaxed.xyz
├── conformers_generated/                # all RDKit-generated (pre-relax)
├── conformers_relaxed/                  # all UMA-relaxed survivors
├── summaries/                           # per-molecule .json summaries
└── slurm/                               # submitit logs
```

These are relaxed (BFGS on UMA-omol), deduplicated (Butina + energy gate),
and pruned by `energy_window` — see [`../confgen/README.md`](../confgen/README.md)
for the full workflow.

### 2. Run the crystal-structure prediction workflow

The same [`molecules.csv`](molecules.csv) drives Stage 1: its
`conformers_path` column points at
`<PROJECT_ROOT>/conformers/conformers_fastcsp/<name>/`. Run all 7 stages
in order:

```bash
fastcsp -c example/fastcsp_config.yaml \
        -s generate process_generated relax \
           compute_conformer_corrections filter evaluate compute_free_energy
```

Stages 4, 6, and 7 are optional — drop any of them and the workflow
still runs end-to-end. Restart is automatic: rerunning the same command
after a failure resumes from the first incomplete stage.

### 3. Inspect results

For reference, on our scavenge-queue run this took **~2.5 hours**
end-to-end:

| Stage | Wall time (scavenge queue) | Output |
|---|---|---|
| generate | ~2 min | 7 Genarris array tasks |
| process_generated | ~1 min | 3 dedup jobs |
| relax | **~50 min** | 10 GPU ranks over 12 parquets |
| compute_conformer_corrections | ~1 min | 4 GPU ranks (re-scoring only) |
| filter | ~2 min | 111 filtered structures across the 2 molecules |
| evaluate | ~4 s | 2 pymatgen matcher jobs |
| compute_free_energy | **~1h 35 min** | 56 GPU jobs (2 structures per job) |

```
<PROJECT_ROOT>/
├── FastCSP.log                                # workflow log
├── config.yaml                                # copy of fastcsp_config.yaml
├── molecules.csv                              # copy of the input CSV
│
├── generated_structures/                      # Stage 1 (Genarris raw output)
│   ├── XULDUD/<conf>/Z<z>/structures.json     #   one JSON per (mol, conf, Z, SG) task
│   └── WIDBAO/<conf>/Z<z>/structures.json
│
├── raw_structures/                            # Stage 2 (processed + pre-dedup)
│   └── <mol>/<conf>/partition_id=*/*.parquet
│
├── slurm/                                     # workflow-orchestrator submitit logs
│
└── relaxed/
    └── uma_sm_1p1_omc_bfgs_0.01_1000_relaxcell/   # <calculator>_<optimizer>_<fmax>_<steps>_...
        ├── raw_structures/                    # Stage 3+4 (relaxed + corrected)
        │   └── <mol>/<conf>/partition_id=*/*.parquet
        ├── slurm/                             # Stage 3 per-rank submitit logs
        ├── slurm_conformer_corrections/       # Stage 4 per-rank submitit logs
        ├── filtered_structures/               # Stage 5 (one parquet per mol)
        │   ├── XULDUD.parquet
        │   └── WIDBAO.parquet
        ├── matched_structures_pmg_l0.2_s0.3_a5/   # Stage 6 (evaluate; suffix = tolerances)
        │   ├── XULDUD.parquet
        │   └── WIDBAO.parquet
        └── free_energy/                       # Stage 7 (one parquet per mol)
            ├── XULDUD.parquet
            └── WIDBAO.parquet
```

The final per-molecule parquets (`filtered_structures/<mol>.parquet` and
downstream) carry both the CIF (`cif_generated`, `cif_relaxed`) and the
per-structure metrics. Every row has a unique `structure_id` of the form
``mol=<name>::conf=<conf_id>::z=<Z>::spg=<SG>::hash=<12hex>`` so it can be
joined back to the trajectory / SLURM log / experimental match.

Key parquet columns to look at:

| Column | Stage | Meaning |
|---|---|---|
| `energy_relaxed_per_molecule` | 3 | UMA-omc relaxed energy (kJ/mol/molecule) |
| `density_relaxed`, `volume_relaxed` | 3 | post-relaxation cell (g/cm³, Å³) |
| `optimizer_converged`, `optimizer_steps` | 3 | did BFGS hit `fmax` and in how many steps |
| `validity.crystal_relaxed.*` | 3 | 3 booleans: correct_z, molecule_matches_reference, connectivity_unchanged |
| `correction.e_fragments_{original,corrector}_per_molecule` | 4 | per-molecule fragment single-points (kJ/mol) with the relax + corrector calculators |
| `energy_corrected_per_molecule` | 4 | `energy_relaxed - sum_z E_original + sum_z E_corrector`, per-molecule |
| `validity.conformer_corrections.applied` | 4 | `True` iff both single-point sweeps succeeded for that row |
| `group_index` | 5 | dedup cluster id (`-1` = kept singleton or filtered out) |
| `pymatgen_match`, `pymatgen_rmsd` | 6 | non-null only when the row matched a `refcode` |
| `temperatures`, `free_energy`, `gibbs_free_energies`, `entropy`, `heat_capacity`, `heat_capacity_P` | 7 | arrays on the `t_min..t_max..t_step` grid (Helmholtz / Gibbs / S / C_v / C_p) |
| `bulk_modulus_P`, `thermal_expansion_coefficients`, `gruneisen_parameters` | 7 | quasi-harmonic extras (arrays on the same grid); `phonon_dos` when `compute_dos=true` |

Quick inspection:

```python
import pandas as pd

df = pd.read_parquet(
    "<PROJECT_ROOT>/relaxed/<run_name>/filtered_structures/XULDUD.parquet"
)
top = df.nsmallest(5, "energy_corrected_per_molecule")
print(
    top[
        [
            "structure_id",
            "z",
            "spg_generated",
            "energy_corrected_per_molecule",
            "density_relaxed",
        ]
    ]
)

# Predicted structures that matched an experimental refcode
mdf = pd.read_parquet(
    "<PROJECT_ROOT>/relaxed/<run_name>/matched_structures_pmg_l0.2_s0.3_a5/XULDUD.parquet"
)
print(
    mdf[mdf["pymatgen_match"].notna()][
        ["structure_id", "pymatgen_match", "pymatgen_rmsd"]
    ]
)
```

## Scaling to production

This example is sized as a smoke test — expect no experimental matches
at `num_structures_per_spg: 25`. Production runs are ~10-100x wider:

| Knob | Example | Production |
|---|---|---|
| `genarris.vars.num_structures_per_spg` | 25 | 500–2000 |
| `relax.slurm.num_ranks` | 10 | 1000–3000 |
| `pre_relaxation_filter.npartitions` | 4 | 500–5000 |
| `free_energy.structures_per_job` | 2 | 5–20 |

Every `slurm:` block also accepts `partition`, `mem`, `cpus_per_task`,
`gpus_per_node`, `array_parallelism`, `time`. See
[`../core/configs/example_config.yaml`](../core/configs/example_config.yaml)
for the exhaustive commented reference of every knob the workflow honours.
