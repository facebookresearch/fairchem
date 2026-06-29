# FastCSP Core Modules

This directory contains the core implementation modules for the FastCSP (Fast Crystal Structure Prediction) workflow.

## Architecture Overview

FastCSP follows a modular, workflow-based architecture where each stage can be run independently or as part of a complete workflow. The core implementation emphasizes:

- **SLURM Integration**: Native support for high-performance computing environments
- **Scalability**: Efficient parallel processing and memory management

## Directory Structure

```
fairchem/applications/fastcsp/core/
├── cli.py                   # Command-line interface entry point
│
├── workflow/                # Main workflow orchestration and processing
│   ├── main.py              # Primary workflow orchestrator with logging
│   ├── generate.py          # Genarris structure generation with SLURM
│   ├── process_generated.py # Genarris output processing and deduplication
│   ├── relax.py             # ML-based structure relaxation with UMA
│   ├── filter.py            # Multi-criteria filtering and ranking
│   ├── eval.py              # Experimental structure comparison
│   └── free_energy.py       # Free energy calculations (in development)
│
├── utils/                   # Core utility modules
│   ├── logging.py           # Logging utilities
│   ├── structure.py         # Structure conversion and validation utilities
│   ├── slurm.py             # SLURM job management and monitoring
│   ├── configuration.py     # Configuration validation and parsing
│   └── deduplicate.py       # Structure deduplication
│
└── configs/                 # Example configuration files
    └── example_config.yaml  # Complete workflow configuration template
```

## Data Flow Architecture

```
Input: molecules.csv + config.yaml
        ↓
[generate] → generated_structures/ (raw structure generation)
        ↓
[process_generated] → raw_structures/ (processed & deduplicated)
        ↓
[relax] → relaxed/<calculator_and_optimizer_info>/raw_structures/ (ML-optimized parquet per conformer)
        ↓
[filter] → relaxed/<calculator_and_optimizer_info>/filtered_structures/ (one parquet per molecule)
        ↓
[evaluate] → relaxed/<calculator_and_optimizer_info>/matched_structures_{csd,pmg_l*_s*_a*}/ (one parquet per molecule)
```

## Configuration Management

FastCSP uses a hierarchical YAML configuration system. The full set of supported flags
lives in [`core/configs/example_config.yaml`](configs/example_config.yaml) (which itself
points at [`core/configs/example_systems.csv`](configs/example_systems.csv)). The snippet
below is a faithful, abbreviated mirror of that file - every key here is honoured by the
workflow; defaults are noted in comments.

```yaml
# Required
root: "/path/to/project"
molecules: "configs/example_systems.csv"   # CSV: name, conformers_path, [z, spg, refcode, cif_path]

# Stage 1: Genarris generation
genarris:
  mpi_launcher: /path/to/mpirun
  python_cmd: /path/to/python/with/genarris/installed
  genarris_cli: /path/to/genarris_cli.py
  genarris_base_config: configs/gnrs_base.conf
  vars:
    Z: [1, 2, 3, 4, 6, 8]
    spg_distribution_type: standard   # "standard", list[int], or list[list[int]] (per Z)
    num_structures_per_spg: 500
    read_z_from_file: false
    read_spg_from_file: false
  slurm:
    job-name: genarris
    nodes: 1
    ntasks-per-node: 40
    time: 10800

# Stage 2: pre-ML deduplication on Genarris outputs
# Bin key = (mol_id) + any of conf_id / Z / spg / binned-density that you enable.
pre_relaxation_filter:
  assign_groups: true        # run dedup blocker
  remove_duplicates: true    # drop all-but-one per group (rep = closest-to-median density)
  ltol: 0.3                  # StructureMatcher tolerances (looser than post-relax)
  stol: 0.4
  angle_tol: 5
  bin_by_conf: false         # bin-key extensions (all default false)
  bin_by_z: false
  bin_by_spg: false
  density_bin_size: 0.02     # density blocker bin (g/cc); null disables
  density_tol: null          # cheap |Δρ| pairwise prefilter (g/cc); null disables
  apply_niggli_filter: false # most reliable when bin_by_z and bin_by_spg are both true
  npartitions: 1000          # number of parquet partitions / SLURM array size

# Stage 3: ML relaxation
relax:
  calculator: uma_sm_1p1_omc   # one of: uma_sm_1p1_omc, uma_sm_1p1_omol,
                               #         uma_sm_1p2_omc, uma_sm_1p2_omol
  optimizer: BFGS
  fmax: 0.01
  max_steps: 1000
  fix_symmetry: false
  relax_cell: true
  write_traj: true
  traj_interval: 1
  slurm:
    num_ranks: 1000

# Stage 4: post-ML filtering + deduplication
post_relaxation_filter:
  remove_problematic: true   # drop non-converged or connectivity-changed (else group=-1)
  energy_cutoff: 20          # kJ/mol above the global minimum; null disables
  density_min_cutoff: 0.5    # g/cm³ lower bound on relaxed density; null disables
  density_max_cutoff: 3.0    # g/cm³ upper bound on relaxed density; null disables
  assign_groups: true
  remove_duplicates: true    # representative = lowest energy_relaxed_per_molecule
  ltol: 0.2                  # tighter tolerances post-relax
  stol: 0.3
  angle_tol: 5
  bin_by_conf: false
  bin_by_z: false            # when true, dedup also disables sm.scale and
                             # sm.primitive_cell internally (small speedup)
  bin_by_spg: false
  density_bin_size: 0.1      # g/cc; coarser than pre-relax (basins are tight)
  energy_bin_size: 2         # kJ/mol; ~thermal scale
  density_tol: null          # cheap |Δρ| prefilter
  energy_tol: null           # cheap |ΔE| prefilter (kJ/mol)
  apply_niggli_filter: false # most reliable when bin_by_z and bin_by_spg are both true

# Stage 5 (optional): experimental evaluation
evaluate:
  method: csd                                     # "csd" or "pymatgen"
  target_xtals_dir: /path/to/experimental/cifs    # global directory of {refcode}.cif files
  csd:
    num_cpus: 60
    python_cmd: /path/to/python/with/csd/api/installed
    target_rows_per_chunk: 200                    # rows per CCDC subprocess chunk
    chunk_timeout: 3600                           # per-chunk subprocess timeout (sec)
  pymatgen:
    match_params: {ltol: 0.2, stol: 0.3, angle_tol: 5}
    slurm: {job-name: eval_pymatgen, cpus_per_task: 1, mem_gb: 10, time: 1000}

# Stage 6 (optional, in development): vibrational free energies
# Run via ``--stages compute_free_energy``. Reads matched_structures/ and
# writes per-structure thermo to free_energy/.
free_energy:
  calculator: uma_sm_1p1_omc
  quasiharmonic: true        # quasi-harmonic (volume sweep)
  atom_disp: 0.01            # finite-difference displacement (Å)
  min_lengths: 15.0          # min supercell side (Å)
  t_min: 0                   # temperature grid (K)
  t_max: 500
  t_step: 10
  match_only: true           # only structures matched to experiment
  energy_cutoff: null        # kJ/mol above the minimum; null = no cutoff
  max_structures: null       # cap per molecule; null = no cap
  structures_per_job: 10     # SLURM batching
  compute_dos: false

# (Optional) logging
logging:
  level: INFO          # DEBUG | INFO | WARNING | ERROR
  console: true        # mirror to stdout in addition to FastCSP.log
```

> **Tip**: every stage section also accepts a `slurm:` block. Omitted SLURM
> blocks fall back to module-specific defaults (see
> [`core/utils/slurm.py`](utils/slurm.py): `genarris`, `process_generated`,
> `relax`, `filter`, `evaluate`, `free_energy`).

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

### Programmatic Usage
```python
from fairchem.applications.fastcsp.core.workflow.main import main
from fairchem.applications.fastcsp.core.utils.logging import get_central_logger

# Get the central logger
logger = get_central_logger()

# Run individual functions
from fairchem.applications.fastcsp.core.workflow.relax import run_relax_jobs

jobs = run_relax_jobs(input_dir, output_dir, relax_config)
```
