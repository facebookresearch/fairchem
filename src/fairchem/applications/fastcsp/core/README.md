# FastCSP Core Modules

Developer / library reference for the FastCSP implementation under
[`fairchem/applications/fastcsp/core/`](.). For user-facing docs (getting
started, config reference, end-to-end example), see the
[top-level FastCSP README](../README.md), the
[commented example config](configs/example_config.yaml), and the
[end-to-end example](../example/README.md).

## Directory Structure

```
fairchem/applications/fastcsp/core/
├── cli.py                   # Command-line interface entry point
│
├── workflow/                    # Main workflow orchestration and processing
│   ├── main.py                     # Primary orchestrator with logging + restart
│   ├── generate.py                 # Genarris structure generation with SLURM
│   ├── process_generated.py        # Genarris output processing + dedup
│   ├── relax.py                    # ML-based structure relaxation with UMA
│   ├── conformer_correction.py     # (Optional) per-conformer fragment corrections
│   ├── filter.py                   # Multi-criteria filtering and ranking
│   ├── eval.py                     # Experimental structure comparison
│   └── free_energy.py              # (Optional) quasi-harmonic vibrational free energies
│
├── utils/                       # Core utility modules
│   ├── logging.py                  # Central logger
│   ├── structure.py                # Structure conversion, validation, extract_molecules
│   ├── slurm.py                    # SLURM job submission (submitit wrappers)
│   ├── configuration.py            # Stage-aware config validation + ordering
│   └── deduplicate.py              # StructureMatcher-based deduplication
│
├── dft/                         # Optional DFT (VASP) validation stage
│
└── configs/
    ├── example_config.yaml         # Exhaustive commented workflow config
    ├── example_systems.csv         # Reference molecule set (JACS paper)
    └── genarris_base.conf          # Genarris configuration template
```

## Data Flow

```
Input: molecules.csv + config.yaml
        ↓
[generate]                          → generated_structures/
        ↓
[process_generated]                 → raw_structures/
        ↓
[relax]                             → relaxed/<run_name>/raw_structures/
        ↓
[compute_conformer_corrections]     → same raw_structures/ (in place, +correction.*
   (optional)                          + energy_corrected columns), or
                                       raw_conformer_corrected_structures/ if
                                       separate_output=true
        ↓
[filter]                            → relaxed/<run_name>/filtered_structures/
        ↓
[evaluate] (optional)               → relaxed/<run_name>/matched_structures_{csd,pmg_l*_s*_a*}/
        ↓
[compute_free_energy] (optional)    → relaxed/<run_name>/free_energy/
```

`<run_name>` is auto-derived from the relax config, e.g.
`uma_sm_1p1_omc_bfgs_0.01_1000_relaxcell`. See the
[example README](../example/README.md) for the concrete on-disk layout
and per-parquet column reference.

## Programmatic (library) usage

Every stage exposes `get_<stage>_config()` and `run_<stage>_jobs()`; the
[orchestrator](workflow/main.py) shows how they compose. Example:

```python
import yaml
from fairchem.applications.fastcsp.core.workflow.relax import (
    get_relax_config_and_dir,
    run_relax_jobs,
)

config = yaml.safe_load(open("config.yaml"))
relax_cfg, relax_out = get_relax_config_and_dir(config, verbose=True)
jobs = run_relax_jobs(
    input_dir=relax_out.parent / "raw_structures",
    output_dir=relax_out / "raw_structures",
    relax_config=relax_cfg,
)
```
