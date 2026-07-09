# FastCSP: Accelerated Molecular Crystal Structure Prediction with Universal Model for Atoms

FastCSP is a complete computational workflow for predicting molecular crystal structures from molecular SMILES strings by combining conformer generation, random structure generation, and machine learning-based optimization without requiring any final DFT reranking.

## Overview

<div align="center">
<img src="fastcsp.svg" alt="FastCSP Workflow Overview" width="800"/>
</div>

### Workflow Stages

FastCSP splits into an **upstream conformer helper** (`fastcsp-confgen`,
SMILES → 3D geometries) and the **main 7-stage prediction workflow**
(`fastcsp`). The two are separate CLIs: `fastcsp-confgen` writes per-molecule XYZ conformers, and `fastcsp` consumes them via the `conformers_path` column of `molecules.csv`. You can skip `fastcsp-confgen` entirely if you already have 3D
conformers from another source.

0. **Conformer Generation** (Optional, upstream — [`fastcsp-confgen`](confgen/README.md)):
   Per molecule, generate a diverse RDKit ETKDG pool, MLIP-relax with a
   FAIR-Chem UMA calculator, dedup on best-RMSD (Butina) + energy gate,
   and write per-conformer geometry files that feed into Stage 1 below.
1. **Structure Generation**: [`Genarris 3.0`](https://github.com/Yi5817/Genarris) generates putative crystal structures.
2. **Process Generated Structures**: Pymatgen's StructureMatcher deduplicates generated structures.
3. **MLIP Relaxation**: Structures are fully relaxed using the Universal Model for Atoms (UMA) from [`fairchem`](https://fair-chem.github.io/).
4. **Conformer Energy Corrections** (Optional): Each relaxed crystal is split into its ``z``molecules and re-scored with a second ("corrector") UMA calculator.
5. **Filtering**: Property filtering and structure deduplication using pymatgen's StructureMatcher. This generates the energy landscape at 0 K.
6. **Experimental Validation** (Optional): Evaluation through comparison against experimental crystal structures using PackingSimilarity from CSD Python API [requires [CCDC license](https://downloads.ccdc.cam.ac.uk/documentation/API/installation_notes.html)] or pymatgen's StructureMatcher.
7. **Free Energy Calculations** (Optional): Quasi-harmonic vibrational thermodynamics is computed per structure with the same UMA calculator (Gibbs free energies, entropies, and optional phonon DOS), producing a temperature-dependent free-energy landscape.

### Key Features

- Native SLURM support for parallel processing across compute clusters
- Scalable from single molecules to large datasets
- Control knobs for each stage runtime/stringency tradeoff
- Modular stage-based execution - run complete pipeline or individual steps
- Resume capability - skip already completed stages

## Output Directory Structure

FastCSP creates a well-organized directory structure to manage all data and results:

```
your_project_root/
├── FastCSP.log                     # Main workflow log file
├── molecules.csv                   # Input: Molecule definitions and conformer paths
├── config.yaml                     # Workflow configuration file
│
├── generated_structures/           # Stage 1: Raw Genarris structure generation
│   ├── MOLECULE1/
│   │   ├── CONFORMER1/
│   │   │   ├── Z1/
│   │   │   │    ├── ui.conf
│   │   │   │    ├── slurm.sh
│   │   │   │    ├── Genarris.out
│   │   │   │    └── structures.json
│   │   │   └── Z2/
│   │   └── CONFORMER2/
│   └── MOLECULE2/
│
├── raw_structures/                 # Stage 2: Processed and deduplicated structures
│   ├── MOLECULE1/
│   │   ├── CONFORMER1/
│   │   │   └── partition_id=*/
│   │   │       └── *.parquet      # Processed structures in Parquet format
│   │   └── CONFORMER2/
│   └── MOLECULE2/
│
└── relaxed/                        # Stage 3+: ML relaxation and analysis results
    └── uma_sm_1p1_omc_bfgs_0.01_1000_relaxcell/  # Named by ML model + optimizer settings
        ├── raw_structures/         # Stage 3: ML-relaxed crystal structures
        │   ├── MOLECULE1/          #   Stage 4 (compute_conformer_corrections) rewrites
        │   │   └── CONFORMER1/     #   these parquets in place by default, adding an
        │   │       └── partition_id=*/  # energy_corrected column (see raw_conformer_
        │   │           └── *.parquet    # corrected_structures/ below for separate-output mode).
        │   └── MOLECULE2/
        │
        ├── raw_conformer_corrected_structures/   # Stage 4 (optional): only when
        │   ├── MOLECULE1/                        # conformer_corrections.separate_output=true
        │   │   └── CONFORMER1/                   # (default is to rewrite raw_structures/ in place)
        │   │       └── partition_id=*/
        │   │           └── *.parquet
        │   └── MOLECULE2/
        │
        ├── filtered_structures/    # Stage 5: Energy-filtered and deduplicated structures
        │   ├── MOLECULE1.parquet   # One parquet per molecule
        │   └── MOLECULE2.parquet
        │
        ├── matched_structures/         # Stage 6 (eval): name depends on method
        │   │                               #   csd      → matched_structures_csd/
        │   │                               #   pymatgen → matched_structures_pmg_l<ltol>_s<stol>_a<angle_tol>/
        │   ├── MOLECULE1.parquet           # Per-molecule structures with experimental similarity scores
        │   └── MOLECULE2.parquet
        │
        └── free_energy/             # Stage 7 (optional, compute_free_energy):
            ├── MOLECULE1.parquet    # per-structure vibrational thermo (F(T), S(T),
            └── MOLECULE2.parquet    # optional DOS) joined with the input columns
```

### Key Data Files

- **Parquet Files**: Compressed columnar storage containing structure data, energies, lattice parameters, and metadata. Beyond the base ``energy_relaxed`` and ``density_relaxed`` columns, downstream stages add:
  - ``energy_corrected``, ``energy_corrected_per_molecule``, ``correction.*``, ``validity.conformer_corrections.applied`` (Stage 4 `compute_conformer_corrections`).
  - vibrational thermo columns (Helmholtz/Gibbs free energies, entropies, and optional phonon DOS) on a temperature grid (Stage 7 `compute_free_energy`).
- **CIF Strings**: Stored within Parquet files for easy structure visualization and analysis
- **JSON Files**: Raw Genarris outputs with structure information
- **Log Files**: Comprehensive workflow logs with timestamps, stage progress, and error tracking

### Input File: molecules.csv

The `molecules.csv` file defines the target molecules for crystal structure prediction.

**Required Columns:**
| Column | Type | Description |
|--------|------|-------------|
| `name` | str | Unique identifier for the molecule (used as directory names) |
| `conformers_path` | str | Path to molecular geometry file (.xyz, .extxyz, .mol) or directory containing multiple conformers |

**Optional Columns:**
| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `z` | str | List of Z-values (molecules per unit cell) | `"[1, 2, 4]"` |
| `spg` | str | Space group specification per Z-value | `"[[14, 19], [2, 4]]"` or `"standard"` |
| `refcode` | str | CSD refcode(s) for evaluation, comma-separated for polymorphs | `"ACSALA01,ACSALA02"` |
| `cif_path` | str | Path to experimental CIF file or directory for evaluation (alternative to global `evaluate.target_xtals_dir`) | `/data/experimental/aspirin.cif` |

**Example molecules.csv** (matches `core/configs/example_systems.csv`):
```csv
name,conformers_path,refcode
ACETAC,ACETAC03_mol.xyz,ACETAC
GLYCIN,GLYCIN20_mol.xyz,"GLYCIN20,GLYCIN32,GLYCIN68,GLYCIN16,GLYCIN67"
IHEPUG,IHEPUG_mol.xyz,"IHEPUG02,IHEPUG"
```

**Space Group (`spg`) Behavior:**
| `spg` value | `z` value | Result |
|-------------|-----------|--------|
| `"standard"` | `[1, 2, 4]` | All compatible space groups used for each Z |
| `[14, 19]` | `[1, 2, 4]` | Space groups 14 and 19 used for **all** Z values |
| `[[14, 19], [2, 4], [14]]` | `[1, 2, 4]` | SG 14,19 for Z=1; SG 2,4 for Z=2; SG 14 for Z=4 |

**Notes:**
- Enable `read_z_from_file: true` and/or `read_spg_from_file: true` under `genarris.vars` in the config to use per-molecule `z`/`spg` values from the CSV
- `conformers_path` can point to a single geometry file (.xyz, .extxyz, .mol) or a directory containing multiple conformer files
- For evaluation, supply experimental crystals via either `evaluate.target_xtals_dir` (one shared directory of `.cif` files keyed by refcode) or a per-molecule `cif_path` column
- `refcode` can be comma-separated for polymorphs

## Getting Started

### Prerequisites
- SLURM cluster environment for parallel processing
- GPU resources for efficient ML relaxations

### Installation
1. Clone the [fairchem repo](https://github.com/facebookresearch/fairchem/tree/main)
2. Install FastCSP: `pip install -e packages/fairchem-applications-fastcsp`

### External Dependencies
- **(Required)** [`Genarris 3.0`](https://github.com/Yi5817/Genarris): Crystal structure generation engine
- **(Optional)** [`CSD Python API`](https://downloads.ccdc.cam.ac.uk/documentation/API/installation_notes.html): For experimental structure comparison (requires license)

### End-to-end example

See [`example/`](example/) for a copy-paste-ready SMILES → predicted
crystal-structure landscape pipeline for aspirin and glycine. It uses
`fastcsp-confgen` to generate starting conformers, then all 7 workflow
stages (generate → process_generated → relax → compute_conformer_corrections
→ filter → evaluate → compute_free_energy). Edit the placeholders
(`<PROJECT_ROOT>`, `<PATH_TO_GENARRIS>`, `<YOUR_PARTITION>`, …) and run.

### Basic Usage

**Complete Workflow:**
```bash
# Run full crystal structure prediction pipeline
fastcsp --config config.yaml --stages generate process_generated relax filter
```

**Stage-by-Stage Execution:**
```bash
# Generate structures only
fastcsp --config config.yaml --stages generate

# Run relaxation and filtering
fastcsp --config config.yaml --stages relax filter

# Evaluate against experimental data
fastcsp --config config.yaml --stages evaluate
```

**Restart Capability:**
```bash
# FastCSP automatically detects completed stages and resumes from the last incomplete stage
fastcsp --config config.yaml --stages generate process_generated relax filter
```

### Available Workflow Stages

The `fastcsp` CLI orchestrates the 7 stages below. `fastcsp-confgen` is a
separate upstream CLI (see [`confgen/README.md`](confgen/README.md)) whose
outputs feed Stage 1 via the `conformers_path` column of `molecules.csv`.

| CLI | Stage | Description | Output |
|-----|-------|-------------|--------|
| `fastcsp-confgen` | *upstream (optional)* | SMILES → RDKit ETKDG pool → UMA relax → Butina dedup → per-conformer `.xyz` files | `<PROJECT_ROOT>/conformers/conformers_fastcsp/<name>/*.xyz` |
| `fastcsp` | `generate` | Generate crystal structures using Genarris | `generated_structures/` |
| `fastcsp` | `process_generated` | Process and deduplicate Genarris outputs | `raw_structures/` |
| `fastcsp` | `relax` | Perform UMA-based structure relaxation | `relaxed/<run_name>/raw_structures/` |
| `fastcsp` | `compute_conformer_corrections` *(optional)* | Per-molecule fragment energy corrections on relaxed parquets | `relaxed/<run_name>/raw_structures/` (in-place, or `raw_conformer_corrected_structures/`) |
| `fastcsp` | `filter` | Property filtering and duplicate removal | `relaxed/<run_name>/filtered_structures/` |
| `fastcsp` | `evaluate` | Compare against experimental data | `relaxed/<run_name>/matched_structures_{csd,pmg_*}/` |
| `fastcsp` | `compute_free_energy` *(optional)* | Quasi-harmonic vibrational free energies | `relaxed/<run_name>/free_energy/` |

### Configuration

FastCSP uses YAML configuration files to control all workflow parameters. Example configurations can be found in `core/configs/example_config.yaml`.

**Key Configuration Sections:**
- `root`: Base directory for all outputs
- `molecules`: Path to input molecule CSV file (required columns `name`,
  `conformers_path`; optional `z`, `spg`, `refcode`, `cif_path`)
- `genarris`: Structure generation parameters
  (`mpi_launcher`, `python_cmd`, `genarris_cli`, `genarris_base_config`,
  `vars.{Z, spg_distribution_type, num_structures_per_spg, read_z_from_file,
  read_spg_from_file}`) and SLURM block
- `pre_relaxation_filter`: Pre-ML deduplication
  (`assign_groups`, `remove_duplicates`, `remove_problematic`, `ltol`/`stol`/`angle_tol`,
  `bin_by_conf`/`bin_by_z`/`bin_by_spg`, `density_bin_size`, `density_tol`,
  `apply_niggli_filter`, `npartitions`). Set `remove_problematic: true` to drop structures whose
  generation-time validity flags (`correct_z`, `molecule_matches_reference`) are False before
  relaxation.
- `relax`: ML relaxation settings
  (`calculator`, `optimizer`, `fmax`, `max_steps`, `fix_symmetry`,
  `relax_cell`, `write_traj`, `traj_interval`) and SLURM block
- `conformer_corrections`: Per-conformer fragment energy corrections (run
  with `--stages compute_conformer_corrections`). Keys: `corrector_calculator`
  (required), `original_calculator` (defaults to `relax.calculator`),
  `separate_output` (default `false` → rewrite parquets in place), and a
  `slurm` block. Adds columns `energy_corrected`,
  `energy_corrected_per_molecule`, `correction.*`, and
  `validity.conformer_corrections.applied` to the relaxed parquets.
- `post_relaxation_filter`: Property cutoffs and deduplication
  (`remove_problematic`, `energy_cutoff`, `density_min_cutoff`,
  `density_max_cutoff`, `assign_groups`, `remove_duplicates`,
  `ltol`/`stol`/`angle_tol`,
  `bin_by_conf`/`bin_by_z`/`bin_by_spg`, `density_bin_size`/`energy_bin_size`,
  `density_tol`/`energy_tol`, `apply_niggli_filter`)
- `evaluate`: Experimental comparison
  (`method` = `csd` or `pymatgen`, `target_xtals_dir`,
  `csd.{num_cpus, python_cmd, target_rows_per_chunk, chunk_timeout}`,
  `pymatgen.{match_params, slurm}`)
- `free_energy`: Vibrational free energy corrections (run with
  `--stages compute_free_energy`). Keys: `calculator`, `input_directory`
  (default `filtered_structures`), `quasiharmonic`, `atom_disp`,
  `min_lengths`, `t_min`/`t_max`/`t_step`, `match_only` (default `false`;
  requires `input_directory: matched_structures` when `true`),
  `energy_cutoff`, `max_structures`, `structures_per_job`, `compute_dos`,
  and a `slurm` block.
- `logging`: Log file settings (`level`, `console`)

> See [`core/configs/example_config.yaml`](core/configs/example_config.yaml)
> for the exhaustive, commented reference. Note: enabling
> `apply_niggli_filter=true` outside a `(mol_id, Z, spg)` bucket emits a
> runtime warning - the prefilter is most reliable when both `bin_by_z` and
> `bin_by_spg` are also `true`.

### Monitoring Progress

FastCSP provides comprehensive logging and progress tracking:

```bash
# Monitor workflow progress
tail -f your_project_root/FastCSP.log

# Check SLURM job status
squeue -u $USER

# View stage completion in log
grep "STAGE COMPLETE" your_project_root/FastCSP.log
```

## Citation

If you use FastCSP in your research, please cite:

```bibtex
@misc{gharakhanyan2025fastcsp,
  title={FastCSP: Accelerated Molecular Crystal Structure Prediction with Universal Model for Atoms},
  author={Gharakhanyan, Vahe and Yang, Yi and Barroso-Luque, Luis and Shuaibi, Muhammed and Levine, Daniel S and Michel, Kyle and Bernat, Viachaslau and Dzamba, Misko and Fu, Xiang and Gao, Meng and others},
  year={2025},
  eprint={2508.02641},
  archivePrefix={arXiv},
  primaryClass={physics.chem-ph},
  url={https://arxiv.org/abs/2508.02641},
}
```

## Support & Contribution

- **Issues**: [GitHub Issues](https://github.com/facebookresearch/fairchem/issues)
- **Discussions**: [GitHub Discussions](https://github.com/facebookresearch/fairchem/discussions)
