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
[generate] → genarris/ (raw structure generation)
        ↓
[process_generated] → raw_structures/ (processed & deduplicated)
        ↓
[relax] → relaxed/<calculator_and_optimizer_info>/relaxed_structures/ (ML-optimized)
        ↓
[filter] → relaxed/<calculator_and_optimizer_info>/filtered_structures/ (ranked by energy)
        ↓
[evaluate] → relaxed/<calculator_and_optimizer_info>/matched_structures/ (experimental comparison)
```

## Configuration Management

FastCSP uses a hierarchical YAML configuration system:

```yaml
# Core workflow settings
root: "/path/to/project"

# Input molecules
molecules: "molecules.csv"

# generation stage
genarris:
  vars:
    Z: [1,2]
    num_structures_per_spg: 500
  slurm:
    nodes: 1

# Structure comparison tolerances for Pymatgen's StructureMatcher
# Deduplication parameters after structure generation with Genarris
pre_relaxation_filter:        # Before ML relaxation
  ltol: 0.2           # lattice tolerance
  stol: 0.3           # site tolerance
  angle_tol: 5        # angle tolerance (degrees)

# ml relaxation
relax:
  calculator: "uma_sm_1p1_omc"
  optimizer: "bfgs"
  fmax: 0.01
  max_steps: 1000
  slurm:
    gpus_per_node: 1

# post-ml deduplication and ranking
post_relaxation_filter:       # After ML relaxation
  energy_cutoff: 20.0  # kJ/mol above minimum
  density_cutoff: 0.1  # g/cm³ tolerance
  ltol: 0.2
  stol: 0.3
  angle_tol: 5
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
from fairchem.applications.fastcsp.core.utils.logging import get_fastcsp_logger

# Set up logging
logger = get_fastcsp_logger(config=config, root_dir="./results")

# Run individual functions
from fairchem.applications.fastcsp.core.workflow.relax import run_relax_jobs

jobs = run_relax_jobs(input_dir, output_dir, relax_config)
```
