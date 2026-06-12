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

FastCSP uses a hierarchical YAML configuration system. The snippet below mirrors `core/configs/example_config.yaml` (which itself points at `core/configs/example_systems.csv`):

```yaml
# Core workflow settings
root: "/path/to/project"

# Input molecules CSV (columns: name, conformers_path, [refcode, z, spg, cif_path])
molecules: "configs/example_systems.csv"

# generation stage
genarris:
  mpi_launcher: /path/to/mpirun
  python_cmd: /path/to/python/with/genarris/installed
  genarris_cli: /path/to/genarris_cli.py
  genarris_base_config: configs/gnrs_base.conf
  vars:
    Z: [1, 2, 3, 4, 6, 8]
    spg_distribution_type: standard   # "standard" or a list[int] of space-group numbers (per-Z lists also accepted)
    num_structures_per_spg: 500
    read_z_from_file: false   # set true to read z from molecules.csv
    read_spg_from_file: false # set true to read spg from molecules.csv
  slurm:
    job-name: genarris
    nodes: 1
    ntasks-per-node: 40
    time: 10800

# Pre-ML deduplication on Genarris outputs
pre_relaxation_filter:
  assign_groups: true       # assign group indices via deduplication pass
  remove_duplicates: true   # drop duplicates within each group
  ltol: 0.3                 # lattice tolerance
  stol: 0.4                 # site tolerance (Å)
  angle_tol: 5              # angle tolerance (degrees)
  density_bin_size: 0.02    # density blocker bin (g/cc) for hash grouping;
  npartitions: 1000         # number of output partitions / SLURM array size

# ML relaxation
relax:
  calculator: uma_sm_1p1_omc   # or uma_sm_1p1_omol
  optimizer: BFGS
  fmax: 0.01
  max_steps: 1000
  fix_symmetry: false
  relax_cell: true
  write_traj: true
  traj_interval: 1
  slurm:
    num_ranks: 1000

# Post-ML filtering + deduplication
post_relaxation_filter:
  remove_problematic: true   # drop structures that didn't converge or whose connectivity changed
  energy_cutoff: 20          # kJ/mol above the global minimum (null = no filter)
  density_min_cutoff: 0.5    # g/cm³ lower bound on relaxed density (null = no lower bound)
  density_max_cutoff: 3.0    # g/cm³ upper bound on relaxed density (null = no upper bound)
  assign_groups: true
  remove_duplicates: true
  ltol: 0.2
  stol: 0.3
  angle_tol: 5
  # Post-relax dedup blocker subdivides (mol_id, Z) buckets.
  density_bin_size: 0.1      # g/cc; relaxation tightens density into basins
  energy_bin_size: 2         # kJ/mol; ~thermal scale

# (Optional) Experimental evaluation
evaluate:
  method: csd                                     # "csd" or "pymatgen"
  target_xtals_dir: /path/to/experimental/cifs    # global directory of {refcode}.cif files
  csd:
    num_cpus: 60
    python_cmd: /path/to/python/with/csd/api/installed
    target_rows_per_chunk: 200   # rows per CCDC subprocess chunk (default 200)
    chunk_timeout: 3600          # per-chunk subprocess timeout in seconds (default 3600 = 60 min)
  pymatgen:
    match_params:
      ltol: 0.2
      stol: 0.3
      angle_tol: 5
    slurm:
      job-name: eval_pymatgen
      cpus_per_task: 1
      mem_gb: 10
      time: 1000
```

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
