# Numerical Regression Suite

The regression suite checks that a model configuration produces the same
numbers as the reference implementation. A *baseline* runs the main-branch FP64
eager model on a fixed set of structures, calibrates how much FP64, FP32, and
TF32 runs normally differ from it, and verifies those limits with independent
repeats. A *candidate* (an inference profile, a backend, or a code branch) is
then qualified against the same FP64 outputs and receives the tightest tier it
passes: `PASS`, `PASS@FP32`, `PASS@TF32`, or `FAIL`. No comparison uses DFT
targets; the suite measures numerical fidelity, not model accuracy.

This page has six parts:

- [Check a candidate](#check-a-candidate): install, run a quick, fast, or full
  check, pick a profile, and choose where jobs run.
- [Outputs](#outputs): the JSON, HTML, and W&B reports each check writes.
- [Baselines](#baselines): select, build, and publish the FP64 baseline.
- [How grading works](#how-grading-works): the suites, the qualification rule,
  the reports, and the artifact layout.
- [Extend the suite](#extend-the-suite): add a suite or a profile.
- [Operations](#operations): troubleshooting, CI coverage, and debugging one
  task.

## Check a candidate

### Install and log in

Run every command from the repository root. Install the package, and log in to
Hugging Face with an account that has access to the
[UMA model](https://huggingface.co/facebook/UMA):

```bash
pip install -e 'packages/fairchem-core[dev,extras]'
hf auth login
```

Artifacts go to `regression-artifacts/` in your checkout (ignored by git), so
the checkout must be on a filesystem that SLURM compute nodes can read.

Fast and full checks grade against the checkpoint's published baseline on
Hugging Face. See [Select a baseline](#select-a-baseline) to use another one.

There are three ways to check a candidate profile. Each is one command and
runs locally or on SLURM:

| Command | Baseline | Cases | Verdict |
| --- | --- | --- | --- |
| `quick PROFILE` | none | each suite's smoke cases | none; prints the gaps to a reference profile |
| `fast PROFILE` | sealed | each suite's smoke cases | `SMOKE_PASS@TIER` or `SMOKE_FAIL` per suite |
| `full PROFILE` | sealed | every case | `PASS`, `PASS@FP32`, `PASS@TF32`, or `FAIL` |

`--suite NAME` limits any check to one suite and can be repeated. NVE MD runs
in quick and fast checks only when selected with `--suite nve-md`; its smoke
cases take far longer than every other suite together.

### Quick check (no baseline)

A quick check runs the candidate and a reference profile (FP32 eager by
default) on a few representative cases of every standard suite and prints, per
suite and observable, the median and P95 of the candidate's difference to the
reference, both relative and absolute. It needs no baseline and grades nothing.

```bash
# On this machine, on its GPU if it has one:
python -m fairchem.core.regression_suite quick nitro --run-on local

# Compare with another profile, on one suite:
python -m fairchem.core.regression_suite quick nitro \
    --reference fp64 --suite inference --run-on local
```

A suite's status is `OK` unless a statistic is NaN or infinite (`NON-FINITE`)
or a declared observable is missing (`MISSING`); the command exits with code 1
if any suite is not `OK` or a run crashes.
`--checkpoint` and `--checkpoint-id` select another model (default: the
`checkpoint` setting), and `--device cpu` runs on the CPU (see
[Where jobs run](#where-jobs-run)). The output ends with an overview table, one
row per suite and observable with the suite's status, and the total wall time.
`--format summary` prints only that table and `--format json` only the JSON
report (see [Outputs](#outputs)).

### Fast check against the baseline

A fast check runs the same smoke cases and grades them against the baseline,
printing a verdict per suite. It takes about 6 to 8 minutes on one GPU, may run
on uncommitted code, and is meant as a quick check, not a qualification.

```bash
python -m fairchem.core.regression_suite fast nitro --run-on local

# From a login node, as one SLURM GPU job:
python -m fairchem.core.regression_suite fast nitro --run-on slurm
```

The output ends with the overview table, which gives each suite's status
(`SMOKE_PASS@TIER` or `SMOKE_FAIL`) and each observable's tier, and the total
wall time. The check runs the baseline's checkpoint on the baseline's
device. `--checkpoint` and `--checkpoint-id` run another file with the same
checkpoint identity, as for `full`; a baseline built from a checkpoint file
needs both, because it records only the identity. `--format` works as for
`quick`.

The first check downloads the checkpoint and the baseline into the Hugging
Face cache. After that, `HF_HUB_OFFLINE=1` uses the cached copies without
network requests, which also avoids proxies that block Hugging Face.

### Qualify a candidate on the full suite

A full qualification runs every case of every supported suite from a clean
commit. It returns immediately; when the jobs finish, a finalizer writes the
report and, when W&B publication is on, publishes it there. The report is
written even if tasks failed, and a suite without records fails in it.

```bash
python -m fairchem.core.regression_suite full nitro --run-on slurm
python -m fairchem.core.regression_suite status        # progress and W&B links
python -m fairchem.core.regression_suite report        # main table of the latest candidate
```

With enough GPUs a candidate takes about 35 to 40 minutes once its jobs start;
the compiled profiles spend most of that compiling one predictor per accuracy
structure. `--wait` blocks and prints the report, as does a local run; the
report ends with its overviews. `--format summary` prints only the overviews
and `--format json` only the JSON report. The report exits with zero only when
every supported suite passes, so it can gate CI.

### Candidate profiles

The `<profile>` argument selects an inference configuration. The preset
profiles take their settings from FAIRChem's named inference settings
(`preset:` in the profile YAML), so they follow the library:

| Profile | TF32 | Merged MOLE | Compiled | Model or backend | Expected tier |
| --- | --- | --- | --- | --- | --- |
| `default` (= `nitro`) | yes | yes | n/a | UMA Nitro | `PASS@TF32` |
| `nitro` | yes | yes | n/a | UMA Nitro | `PASS@TF32` |
| `high-precision` | no | yes | yes | `umas_fast_gpu` | `PASS@FP32` |
| `multigpu` | yes | yes | yes | `umas_fast_gpu` | `PASS@TF32` |
| `batch` | no | no | no | `general`, activation checkpointing | `PASS@FP32` |

A Nitro fallback changes the settings during a run, which fails the run, so a
passing `nitro` or `default` run used UMA Nitro throughout.

One-axis ablations help localize a failure: `fast-gpu-tf32-eager` and
`merged-fp32-compiled`. `fp64`, `fp32`, and `tf32` are the eager profiles used
for calibration; for example, `fast tf32` checks plain TF32 eager inference.
Compiled and `umas_fast_gpu` profiles still run `finite_difference_frequencies`,
but report `autograd_frequencies` as unsupported. The `umas_fast_gpu` backend's
Triton backward kernels are not differentiable, so its autograd Hessian runs
but is wrong, and AOTAutograd's donated-buffer backward cannot build
the double-backward graph for the compiled general backend. Profiles are YAML
files in `src/fairchem/core/regression_suite/configs/profile/`; see
[Add a profile](#add-a-profile).

### Where jobs run

`quick`, `fast`, and `full` accept `--run-on auto|slurm|local`; `baseline`
accepts `slurm` (the default) or `local`.

- `slurm` submits SLURM job arrays and returns; `quick` and `fast` submit one
  job and wait for it.
- `local` runs on this machine and blocks until done: one task per visible GPU,
  or for CPU runs one task per four CPU cores with `OMP_NUM_THREADS=4` unless
  it is set.
- `auto` chooses `local` inside a SLURM allocation or when `sbatch` is
  unavailable, and `slurm` otherwise.

Both produce identical artifacts. A local baseline takes many hours on one GPU,
so run it only deliberately.

`quick` and `baseline` also accept `--device auto|cpu|cuda`. `auto`, the
default, is `cuda` for SLURM runs and, for local runs, `cuda` when a GPU is
visible and `cpu` otherwise. A baseline records its device in `baseline.json`
(baselines without one ran on CUDA), and `fast` and `full` always run on their
baseline's device. A local run that needs CUDA on a host without a GPU stops
before anything starts. CPU runs on SLURM request no GPUs.

SLURM resources and W&B defaults come from the packaged
`configs/settings.yaml` (partition `g3`, account `ocp`, QoS
`g3_alignment_shared`, one GPU and 120 GB per task). Override them with a YAML
file named by `FAIRCHEM_REGRESSION_CONFIG`, which is merged over the defaults;
for example, a file containing `slurm: {qos: g3_ocp_high}` selects a
higher-priority QoS. Submitted jobs clear inherited proxy variables and set
`HF_HUB_OFFLINE=1`, so pre-cache the checkpoint before submitting.

## Outputs

Every check writes a JSON report and a local HTML report, with or without W&B.
Each command prints both report paths last.

| Command | JSON | HTML | W&B |
| --- | --- | --- | --- |
| `quick` | `reports/PROFILE/quick/TIMESTAMP/report.json` | `reports/PROFILE/quick/TIMESTAMP/index.html` | `--wandb` |
| `fast` | `reports/PROFILE/fast/TIMESTAMP/report.json` | `reports/PROFILE/fast/TIMESTAMP/index.html` | `--wandb` |
| `full` | `reports/PROFILE/full/COMMIT-TIMESTAMP/report.json` | `reports/PROFILE/full/COMMIT-TIMESTAMP/index.html` | `--wandb` |
| `report` | rewrites `report.json`; `--format json --output PATH` writes a copy | rewrites `index.html` | `--wandb` |

Paths are under `regression-artifacts/`. Every report of one candidate profile
sits under `reports/PROFILE/`, one directory per run; the raw run data stays
in `quick/`, `fast/`, and `candidates/` (see [Artifacts](#artifacts)).

`quick`, `fast`, `full`, and `report` take `--format table` (the default:
detailed tables ending with the overview), `summary` (only the overview), or
`json` (the report document, the same content as `report.json`). The files
are written whatever the format. With `--format json`, the command prints only
the JSON on standard output and its other messages on standard error. On GPU
hosts NVIDIA Warp can still print its banner to standard output while
`fairchem` is imported, so read the written `report.json`, or use
`report --output PATH`, when you need a file.

### Quick and fast reports

A quick or fast `report.json` has the keys `mode` (`quick` or `fast`),
`profile`, `reference` (the reference profile, or the baseline ID),
`checkpoint`, `created_at` (UTC), `baseline` (the baseline's provenance for a
fast check, null for a quick one), `suites` (each suite's result, with
non-finite statistics as null), and `exit_code`. Its `index.html` shows the
same header, a suite table, and each suite's observable statistics. Each
suite's result is also kept in `SUITE/result.json` beside the report.

### Full reports

`report [RUN]` regenerates and prints a full candidate's report; see
[Reports and charts](#reports-and-charts) for its contents. `--output PATH`
writes the rendered report to `PATH` instead of standard output, so
`report --format json --output PATH` writes the JSON report to `PATH`. Prefer
it to redirecting standard output: `report` sends its own messages to standard
error, but on GPU hosts NVIDIA Warp prints its startup banner to standard
output while `fairchem` is imported, before the report.

### W&B

W&B publication is opt-in and Meta-internal. `--wandb` publishes one run, and
`--tag` adds W&B tags. To publish by default, log in once
(`wandb login --host https://meta-fair.wandb.io`) and set
`wandb.auto_publish: true` in a `FAIRCHEM_REGRESSION_CONFIG` file;
`--no-wandb` then skips one run.

Reports go to `fairchem/uma-regression` on Meta's W&B, grouped by baseline.
Full candidates publish from their finalizer. `quick` and `fast` publish from
the host that launched them, as one run per check with each suite's status
(and a fast check's tier) as summary metrics, the HTML page, and the check
directory as an artifact. `report --wandb` publishes a report as a new W&B run.
Each report records its publication in `wandb.json`, and `status` lists the
links. W&B runs also carry per-observable statuses and ratios and the
performance factors as summary metrics for comparing candidates. W&B logs one
scalar history row per atom count for candidate step time, throughput, and
peak GPU memory. The atom count is the step metric, so W&B renders its
standard interactive plots against system size. Compute nodes need direct
access to the W&B host; hosts under `wandb.no_proxy_hosts` bypass the proxy
during uploads.

## Baselines

A finalized baseline is a single sealed file, for example
`baselines/uma-s-1p2p1-ffea40cf.baseline`: a zip archive of `baseline.json`
(calibrated limits, statistics, plot axes, and provenance) and every run it was
built from. Its `baseline_id` hashes the run manifests, which hash their values,
so the file is tamper-evident. Each member is LZMA-compressed without
changing its bytes, so those hashes still verify. Fast and full checks and
reports read only this file.

### Select a baseline

`fast`, `full`, and `report --verification` use the baseline that `--baseline`
names, or else the `baseline` setting,
`hf:facebook/UMA/regression_baselines/{checkpoint}.baseline`, where
`{checkpoint}` is the name of the checkpoint (`--checkpoint-id`,
`--checkpoint`, or the `checkpoint` setting). A reference takes one of three
forms:

| Reference | Source |
| --- | --- |
| `hf:ORG/REPO/PATH[@REVISION]` | a file in a Hugging Face model repository, at `REVISION` (a commit, branch, or tag) or `main` |
| `wandb:[ENTITY/PROJECT/]NAME[:ALIAS]` | a version of a W&B artifact; entity and project default to the `wandb` settings and the alias to `latest` |
| anything else | a local sealed file or run directory, by path or by name under `baselines/`; `current` is the last baseline finalized here |

Hugging Face downloads go through the Hugging Face cache, so with
`HF_HUB_OFFLINE=1` a cached baseline is used without a network request. A
remote baseline is copied once into `regression-artifacts/baselines/`, beside
local ones, so compute nodes read it from the artifact directory. Each copy is
named after the revision it came from, such as `uma-s-1p2p1@1f43bb7f.baseline`
(a Hugging Face commit) or `uma-s-1p2p1-ffea40cf@v3.baseline` (a W&B
version), so revisions never replace each other or a local baseline. A
candidate's launch manifest records the pinned source in `baseline_source`: the
Hugging Face commit that was downloaded, the concrete W&B version (`:vN`), or
the local path. Passing that value back as `--baseline` selects the same file.
Releases pin the revision that holds their baseline, for example
`baseline: hf:facebook/UMA/regression_baselines/{checkpoint}.baseline@1f43bb7f`
(the commit that published `uma-s-1p2p1-c7730ed6`) in a `FAIRCHEM_REGRESSION_CONFIG` file, because `main` moves when a new
baseline is uploaded. If nothing is published at the path, the check stops
with `no baseline is published at ...`; pass `--baseline <local file>`.

Before anything starts, the submitting host loads the baseline, confirms its
identity, and checks that `baseline.json` names the requested checkpoint
(names and paths compare by stem; the file name is never trusted). If the
baseline is unreachable, missing, corrupt, or built from another checkpoint,
the command prints `Baseline unavailable; nothing was started.` and the cause,
and exits with code 1. Every candidate task checks the baseline again on its
compute node before any GPU work.

### Build or re-run a baseline (maintainers)

A new baseline is needed when a suite's plan or observable contract changes,
when a suite is added, for a new checkpoint, or to calibrate on different
hardware. Run it from a clean checkout of `main`:

```bash
python -m fairchem.core.regression_suite baseline --dry-run   # preview the task graph
python -m fairchem.core.regression_suite baseline             # submit to SLURM
python -m fairchem.core.regression_suite status
```

The standard campaign submits 8 FP64 reference tasks, 24 FP64/FP32/TF32
calibration tasks, 24 held-out verification tasks, and a CPU finalizer, and
takes about 50 minutes with enough GPUs. The finalizer seals the baseline into
one file, makes it the current local baseline, publishes the three verification
reports, and, when W&B publication is on, uploads the sealed file to W&B as a
new version of the `<checkpoint>-baseline` artifact, aliased `latest` and with
the baseline name.
`--wait` blocks until finalization, `--full` adds the NVE suite
(about 45 times the GPU time and a few hours of wall time with enough GPUs),
`--wandb` publishes to W&B, `--finalize` recomputes limits from existing
runs (CPU only; NVE stays in if any launch used `--full`), and `--publish` uploads an existing sealed baseline. Rerunning
the command resumes: completed work is never resubmitted.

Baselines are named after the checkpoint and commit, so a new commit starts a
new baseline and the same commit resumes; `--name` selects another name.
`baseline.json` also records the device and scope that every run shares, its
creation time, each suite's commit, and, for baselines sealed from now on, the
fairchem and torch versions and the device name of the reference runs. Older
baselines without these fields still load.

`--scope smoke` builds the same campaign on each suite's smoke cases only,
without per-case shards. It is for testing the suite itself, not for
qualifying candidates: its name gets a `-smoke` suffix, it never becomes the
current baseline, and it is never published to W&B. `fast` and `full` against
a smoke baseline run its smoke cases, and saved runs are reused only by
launches of their own scope.

### Publish a baseline to Hugging Face (maintainers)

Local baselines keep their checkpoint-and-commit names, such as
`uma-s-1p2p1-ffea40cf.baseline`. The published copy drops the commit, so
checks find it from the checkpoint name alone; the commits and creation time
stay inside `baseline.json` and appear in every report. `facebook/UMA` has
discussions disabled, so it accepts no Hub pull requests; a maintainer with a
token that can write to the repository uploads a sealed baseline as one
commit:

```bash
hf upload facebook/UMA \
    regression-artifacts/baselines/uma-s-1p2p1-ffea40cf.baseline \
    regression_baselines/uma-s-1p2p1.baseline \
    --commit-message "Add the uma-s-1p2p1 regression baseline"
```

`huggingface_hub.HfApi().upload_file(...)` does the same from Python. The
upload's commit is the revision to pin. Each published baseline is listed
below.

| Checkpoint | Baseline | Revision |
| --- | --- | --- |
| `uma-s-1p2p1` | `uma-s-1p2p1-c7730ed6` (id `e205a498`) | `1f43bb7f` |

## How grading works

### Suites

The standard campaign runs eight suites; `--full` adds `nve_md`.

- `prediction_fidelity`: energies, forces, and stress (omat and omc only) for
  ten structures from each UMA task (omat, omol, oc20, oc22, oc25, odac, and
  omc) in nine orientations. Energies are total system energies, and forces on
  atoms held fixed by `FixAtoms` are excluded.
- `edge_cases`: graph-shape probes (isolated atom, cutoff dimer, large cells, a
  sparse-dense-sparse graph churn on one calculator, and clusters whose
  central atom has 299, 300, 301, or 320 neighbors around UMA's 300-neighbor
  cap), NPT contraction,
  expansion, and pressure-cycle trajectories, and a million-kelvin NVT dense
  transient that must finish with finite values (its differences are
  report-only).
- `inference`: energies, forces, and stress of FCC carbon cells with 50 to
  1,000 atoms, plus report-only throughput and memory measurements.
- `physical_constraints`: energy-force consistency, net-force conservation,
  stress finite differences, and force-constant symmetry and step convergence.
- `elastic`: the elastic tensor and Voigt-Reuss-Hill bulk and shear moduli of
  ten crystals.
- `phonon_crystal`: the commensurate-point phonon spectrum of ten crystals after
  a symmetry-preserving relaxation; 300 K thermal properties are report-only.
- `finite_difference_frequencies`: vibrational spectra of small molecules from
  finite-difference Hessians.
- `autograd_frequencies`: vibrational spectra of the same molecules from
  autograd Hessians. The two methods are compared only with their own baseline
  results, not with each other.
- `nve_md`: energy conservation of seven MD22 molecules and three TM23 melts,
  twenty velocity seeds each, run for 1,000 to 10,000 steps. Each system has three
  gates, each computed across its seeds from the total energy per atom
  recorded every 20 steps, and each graded as the time average of one curve:
  `<system>_fp64_gap`, the median over seeds of the smoothed distance from the
  FP64 reference run of the same seed; `<system>_drift`, the median over seeds
  of how far the energy, averaged over about 1 ps, has moved from its first
  averaged value; and `<system>_spread`, the variance over seeds of that
  movement. The full-length deviation RMS, the deviation series, the endpoint
  drift, and the temperature are report-only.

Plans are declarative and hash-audited: each suite's cases, structures, and
rotations reproduce a fixed hash at load time. Rotation matrices are hashed
after truncation to 1e-7, so plans agree across CPU architectures.

### Qualification

Each structure in each orientation is one decision cluster. Only the median and
p95 of the per-cluster errors decide qualification; mean, IQR, and p99 are
diagnostic. Each limit is twice the calibrated statistic of its profile, never
less than twice the FP64-equivalence floor of 1e-10, and limits only loosen from
FP64 to TF32. The absolute-error channel of a `relative_with_floor` observable
is floored at 1e-10 times the observable's denominator floor instead, and a
`log_ratio` observable at its own floor when that is larger. A suite can decide on other statistics with its own safety
factor. `nve_md` grades each gate as one value per system with a factor of 4.
Where a precision's calibration run stays within the limit of the next
stricter precision, the gate cannot tell the two apart, so the stricter one
takes the looser limit and the gate only passes or fails there. Verification
runs are held out from limit construction and must pass their own tier before
a baseline is sealed.

Absolute observables must keep both statistics within their limits. For
`relative_with_floor` observables, a tier passes when either the complete
normalized channel or the complete absolute-error channel passes; the absolute
channel protects small reference values from ill-conditioned ratios.

### Reports and charts

Every report, the `fast` and `full` output, and `status` open with the
baseline's provenance, read from `baseline.json`, so older baselines need no
reseal:

```text
Baseline: uma-s-1p2p1-ffea40cf (id 5d41402a), created 2026-11-05 03:25 UTC
  uma-s-1p2p1, full campaign, full scope, cuda
  commit 2b7e1516: autograd-frequencies, edge-cases, inference
  commit ffea40cf: nve-md
```

A baseline extended over several commits lists each commit with the suites
built from it. Baselines sealed with versions add a `fairchem ..., torch ...`
line, and `fast` and `full` add the pinned source. W&B runs record the same
fields in their configuration.

`report [RUN]` prints complete statistics for every observable and the raw
inference performance, then ends with the overviews: the suite qualification
table, an overview of every observable, the worst clusters of failing or
near-limit observables, and the inference performance comparison. It exits
with zero only if every supported suite passes. `--format summary` prints only
the overviews, `--format json` prints the full report as JSON, and
`report --verification [fp64|fp32|tf32]` regenerates a baseline's verification
reports.

The HTML report (`index.html` in the run's report directory) has three levels:

- each suite row of the qualification table expands into its observables, with
  their status, limiting check, median and p95 against their limits, and the
  ratio to the limit (green below 0.8, amber up to 1, red above 1);
- each observable expands into all of its clusters (one structure in one
  orientation), worst first, so the structures driving a failure are visible;
- a chart of each observable, on axes frozen in the baseline so a
  candidate cannot change its own plots.

Each chart draws the values its grade uses: one point per cluster, with the
candidate in orange beside the FP64, FP32, and TF32 calibration runs and every
tier's median and 95th-percentile limits. Its form follows the data. More than
12 items get cumulative curves with red lines at 50% and 95%, the two points
the grade checks. Twelve or fewer items get one row per item with violins and
median and p95 summary rows. Four or fewer recorded series get one chart per
run with its gaps to FP64 over time, and four or fewer single values get a
table. Report-only observables say why they are not graded, and observables
graded on 20 or fewer points carry a note that their 95th percentile sits
between their two worst points.

A short guide above the suite details explains how to read the results, and
each suite's section opens with a plain-language explanation of what it
checks, which cases it runs, what it measures, and how it is graded. A suite
declares that text in its `explanation` attribute, beside the code it
describes. Every plot also has a caption saying what it shows and how the
plotted values were calculated, built from the suite's
`observable_descriptions` and each observable's comparison settings.

The `nve_md` section explains its gates in plain language, then plots each
system's three gate curves for the candidate's seeds beside the FP64, FP32, and
TF32 calibration runs, with a table of each gate's value, limits, result, and
the precisions it can tell apart.

Performance is report-only and never fails a suite. Inference step time,
throughput, and peak memory are compared with main-branch FP32 eager; factors
above one are better (faster, more throughput, or less memory) and are
highlighted as gains, while regressions are flagged without failing. The HTML
and report artifact retain the numeric comparisons with the main eager
profiles. The comparison requires matching hardware and benchmark protocol; on
another GPU or CPU the report shows `NOT COMPARABLE` with the reason, while a
different PyTorch or CUDA version alone does not prevent it.

### Artifacts

```text
<repo>/regression-artifacts/
├── baselines/
│   ├── current -> uma-s-1p2p1-ffea40cf.baseline
│   ├── uma-s-1p2p1-ffea40cf.baseline    sealed baseline
│   └── uma-s-1p2p1-ffea40cf/            campaign work area: fragments/, launches/, reports/
├── candidates/
│   └── nitro-ffea40cf-20260930T183112Z/  fragments/, launches/
├── fast/
│   └── nitro-20260930T183112Z/
│       └── <suite>/                      result.json, fragments/
├── quick/
│   └── nitro-20260930T183112Z/
│       └── <suite>/                      result.json, reference/, candidate/
└── reports/
    └── nitro/
        ├── full/ffea40cf-20260930T183112Z/  report.json, index.html, wandb.json, charts/
        ├── fast/20260930T183112Z/           report.json, index.html, wandb.json
        └── quick/20260930T183112Z/          report.json, index.html, wandb.json
```

Every run saves full-resolution records in the same format, and the atomically
published records directory is its completion marker. GPU tasks never read the
FP64 reference: summaries are computed from saved records when finalization or
reports need them, so changing summary or tolerance policy never requires new
GPU runs.

Submitted jobs never import the live checkout. Each launch exports its clean
commit to `launches/LAUNCH_ID/source`, and every task imports that snapshot, so
later edits cannot change queued or running work. Each suite's reference,
calibration, and verification runs must share one commit, while different suites
may use different commits; runs whose plan or observable contract no longer
matches the checkout are reported as invalid.

## Extend the suite

### Add a suite

A suite is one Python class. It declares a frozen list of cases and the
observables it measures, and it evaluates one case at a time with a
`FAIRChemCalculator`. The harness runs it under every profile, compares the
results with FP64, and grades them. The suite never sees the reference values.
These steps build a complete suite that compares energies, forces, and stress
of two small crystals. `suites/elastic/suite.py` is a fuller example that reads
its structures from a file.

1. Create `src/fairchem/core/regression_suite/suites/toy_energy/suite.py`.
   The directory needs no `__init__.py`.

   ```python
   """
   Copyright (c) Meta Platforms, Inc. and affiliates.

   This source code is licensed under the MIT license found in the
   LICENSE file in the root directory of this source tree.
   """

   from __future__ import annotations

   import functools
   from typing import TYPE_CHECKING, ClassVar

   import numpy as np
   from ase.build import bulk

   from fairchem.core.regression_suite.spec.plan import unrotated_case
   from fairchem.core.regression_suite.spec.structures import frame_identities
   from fairchem.core.regression_suite.suites.base import (
       STATIC_PREDICTION_OBSERVABLES,
       ObservableDescription,
       Suite,
       static_predictions,
   )

   if TYPE_CHECKING:
       from ase import Atoms

       from fairchem.core.regression_suite.spec.schema import (
           CaseSpec,
           ObservableSpec,
           SuiteCase,
       )


   @functools.cache
   def source_frames() -> tuple[Atoms, ...]:
       """
       Build the suite's structures, the same on every call and machine.
       """
       copper = bulk("Cu", "fcc", a=3.6, cubic=True)
       copper.rattle(stdev=0.05, seed=0)
       salt = bulk("NaCl", "rocksalt", a=5.6)
       salt.rattle(stdev=0.05, seed=1)
       frames = (copper, salt)
       for atoms in frames:
           atoms.positions = np.round(atoms.positions, 8)
           atoms.info["structure_id"] = atoms.get_chemical_formula()
       return frames


   class ToyEnergySuite(Suite):
       """
       Compare energies, forces, and stress of two small crystals with FP64.
       """

       name = "toy-energy"
       plan_hash = ""
       smoke_case_ids = ("toy-energy/Cu4",)
       observables: ClassVar[dict[str, ObservableSpec]] = STATIC_PREDICTION_OBSERVABLES
       generated_frames = staticmethod(source_frames)
       explanation = (
           ("What it checks", "Whether two small crystals get FP64's energies."),
           ("How it is graded", "Relative differences with a floor, per crystal."),
       )
       observable_descriptions: ClassVar[dict[str, ObservableDescription]] = {
           "energy": ObservableDescription("Total energy, in eV", "one crystal"),
           "forces": ObservableDescription("Forces on every atom, in eV/A", "one crystal"),
           "stress": ObservableDescription("Voigt stress, in eV/A^3", "one crystal"),
       }

       def cases(self) -> list[CaseSpec]:
           return [
               unrotated_case(
                   case_id=f"toy-energy/{structure_id}",
                   structure="generated",
                   structure_hash=structure_hash,
                   structure_index=index,
                   task_name="omat",
                   runtime_seed=42,
               )
               for index, (structure_id, structure_hash) in enumerate(
                   frame_identities(source_frames())
               )
           ]

       def evaluate(self, calculator, case: SuiteCase) -> dict[str, np.ndarray]:
           return static_predictions(calculator, case)
   ```

2. Choose where the structures come from. Every frame needs a unique
   `info["structure_id"]`, and `frame_identities` returns each frame's ID and
   hash for `cases()`.
   - To generate them in code, set `generated_frames` to a cached function, as
     above. `structure` is then only a label. Make the function deterministic
     and round positions, so the hash does not depend on the last bits of the
     machine's floating-point math.
   - To read them from a file, put an xz-compressed extended XYZ file in
     `toy_energy/structures/`, call `frame_identities(path)`, and set
     `structure` to the file name. Leave `generated_frames` unset. Add a row
     for the file to `suites/STRUCTURES.md` with its source and changes.
     For generated structures, add the suite to the paragraph at the end of
     that file instead.
3. Declare the cases in `cases()`. Each case is one structure in one
   orientation and is one decision cluster. Build it with a helper from
   `spec/plan.py`:
   - `unrotated_case(...)` keeps the source orientation.
   - `oriented_case(index, orientation, ...)` gives orientation 0 as the
     identity and orientations 1 to 8 as seeded rotations.
   - `precision_rotation_sweep(sample_seed=..., ...)` returns the identity
     followed by eight seeded rotations.

   Put any per-case parameters that `evaluate` reads in `metadata`, as
   `suites/elastic/suite.py` does. `task_name` is the UMA task, and
   `runtime_seed` seeds the case. Each `case_id` and transform pair must be
   unique. Every field is part of the plan hash.
4. Declare the observables. Each key of `observables` is an observable that
   `evaluate` returns, with an `ObservableSpec`:
   - `units`, shown in reports.
   - `comparison`. `absolute` grades the absolute difference.
     `relative_with_floor` divides it by the reference value, but by no less
     than `floor`, and also keeps the absolute channel (see
     [Qualification](#qualification)). `log_ratio` grades the log of the ratio
     and needs positive values.
   - `histogram=HistogramSpec(minimum, maximum)`, the log-axis bounds that
     plots fall back to when a baseline has no positive values.
   - `report_only=True` to show an observable without grading it.

   For energy, forces, and stress, reuse `STATIC_PREDICTION_OBSERVABLES` and
   `static_predictions(calculator, case)`, as above.
5. Implement `evaluate(calculator, case)`. It runs one case and returns a
   dictionary of NumPy arrays keyed by observable. Work on
   `case.transformed_atoms.copy()`. Return values in the source frame:
   `canonicalize_vectors` and `canonicalize_rank_two` in `spec/transforms.py`
   rotate vectors and tensors back, and `static_predictions` already does
   this. Every value must be finite.
6. Pick `smoke_case_ids`. Each ID must name exactly one case in its identity
   orientation. Quick and fast checks, smoke baselines, and CI run only these
   cases, so choose a few cheap ones that still exercise the suite.
7. Write `explanation` and `observable_descriptions`. The full HTML report
   opens the suite's section with the `(heading, paragraph)` pairs of
   `explanation`, and builds each plot's caption from the observable's
   `ObservableDescription(shows, point, report_only_reason)`.
8. Set the optional attributes only when you need them:

   | Attribute | Effect |
   | --- | --- |
   | `decision_statistics`, `safety_factor` | grade on other statistics or with another factor than the campaign's |
   | `merge_unresolved_tiers` | let a tier take its looser neighbour's limit where the two cannot be told apart |
   | `inference_settings_overrides` | settings applied over every profile's, such as `{"hessian_vmap": False}` |
   | `structures_directory` | read structure files from another suite's directory |
   | `slurm_case_parallel` | run each case of a full campaign as its own SLURM task |

9. Register the suite in `_SUITE_CLASSES` in `suites/__init__.py`. Its
   position sets where it runs and appears in reports. Every suite except
   `nve-md` is a standard suite, so quick checks, the standard campaign, and
   the CI end-to-end test pick it up.

   ```text
   "toy-energy": "toy_energy.suite:ToyEnergySuite",
   ```

10. Freeze the plan hash. Leave `plan_hash = ""` and run
    `python -m fairchem.core.regression_suite.suites.plan_digest`. It prints
    `plan toy-energy FAILS: ... it now hashes to <hash>`. Paste that hash into
    `plan_hash` and rerun it until it prints `plan toy-energy ok`. Repeat this
    step after any change to the cases or structures.
11. Freeze the contract hash. Add the suite to `CONTRACT_HASHES` in
    `tests/core/regression_suite/test_plans.py` with an empty string, run that
    test, and paste the hash from its failure message. The
    contract changes with any observable's units, comparison, floor,
    histogram, or report-only flag, and with the suite's grading overrides.
12. If a profile cannot run the suite, add the suite to that profile's
    `unsupported_suites`. Profiles then skip it by default, and `--suite`
    rejects it.
13. Run the suite on the CPU with the tiny random-weight model that CI uses:

    ```bash
    python -c "from pathlib import Path; \
    from tests.core.regression_suite.tiny_model import write_tiny_checkpoint; \
    write_tiny_checkpoint(Path('/tmp/tiny.pt'))"
    python -m fairchem.core.regression_suite quick fp32 --reference fp64 \
        --suite toy-energy --device cpu --run-on local \
        --checkpoint /tmp/tiny.pt --checkpoint-id tiny-uma
    ```

    The suite should report `OK`. Then run the regression tests:

    ```bash
    pytest -c packages/fairchem-core/pyproject.toml \
        tests/core/regression_suite/test_plans.py \
        tests/core/regression_suite/test_plan_digest.py \
        tests/core/regression_suite/test_settings.py \
        tests/core/regression_suite/test_layers.py \
        tests/core/regression_suite/test_end_to_end.py
    ```

    `test_end_to_end.py` takes a few minutes. It requires `OK` from every
    standard suite in a quick check and `SMOKE_PASS@FP32` in a fast check
    against a smoke baseline, so keep the smoke cases cheap on the CPU.
14. Seal a new baseline after the change merges. A published baseline has no
    entry for a new suite. Until a maintainer seals one at a commit that
    contains the suite (see
    [Build or re-run a baseline](#build-or-re-run-a-baseline-maintainers)),
    `fast` and `full` run only the suites the baseline contains, and their
    baseline header lists the new one under `not in this baseline, so not
    checked`. `--suite toy-energy` stops with
    `the baseline does not contain suites: ['toy-energy']`. Quick checks need
    no baseline and run the suite at once. A change to an existing suite's
    plan or contract instead stops fast and full checks with
    `this code's suite plans do not match the baseline` (see
    [Troubleshooting](#troubleshooting)).

### Add a profile

A profile is a YAML file in
`src/fairchem/core/regression_suite/configs/profile/`. Profiles do not change
plan or contract hashes, so existing baselines grade a new profile without a
rebuild.

1. Create `configs/profile/my_profile.yaml`. Set `label` to the file name with
   hyphens (`my-profile`); `test_settings.py` checks it.
2. Choose the settings in one of two ways:
   - `preset:` names a FAIRChem inference preset (`default`,
     `high-precision`, `batch`, `multigpu`, `traineval`, or `nitro`), and an
     optional `inference_settings:` mapping overrides some of its fields.
     The profile then follows the library's preset.
   - `inference_settings:` alone, with
     `_target_: fairchem.core.units.mlip_unit.api.inference.InferenceSettings`,
     spells out every setting, as `fp32_eager.yaml` does.

   A suite's `inference_settings_overrides` apply last.
3. Set `predictor_scope: structure` when the settings merge MOLE, because a
   merged model serves one composition. The default is `suite`.
4. List the suites the profile cannot run in `unsupported_suites`, with
   hyphenated names. Checks skip them by default, and `--suite` stops with
   `profile '...' does not support suites: [...]`. `test_settings.py`
   requires `autograd-frequencies` in the list exactly when the profile
   compiles or uses `umas_fast_gpu`.
5. To give the profile a short name, add it to `PROFILE_ALIASES` in
   `spec/profiles.py`. Hyphenated and underscored names already resolve.

```yaml
# The batch preset without activation checkpointing.
label: my-profile
preset: batch
inference_settings:
  execution_mode: general
  activation_checkpointing: false
unsupported_suites: [nve-md]
```

Check it with
`python -m fairchem.core.regression_suite quick my-profile --suite inference --run-on local`
and `tests/core/regression_suite/test_settings.py`.

## Operations

### Troubleshooting

- **`Baseline unavailable; nothing was started.`** The next line names the
  cause. `no baseline is published at ...` means nothing is uploaded for the
  checkpoint at that revision; pass `--baseline <local file>`. With
  `HF_HUB_OFFLINE=1`, a Hugging Face baseline must already be cached; run once
  without it to download. For a missing local baseline, run `baseline`. For
  W&B, run `wandb login --host https://meta-fair.wandb.io` and set
  `WANDB_BASE_URL` if your client defaults to another server.
- **`this code's suite plans do not match the baseline`.** A suite's cases or
  observable contract changed since the baseline was sealed. The message lists
  each such suite with the commit its baseline entry was built at. Pin an
  older baseline revision built from matching code
  (`--baseline hf:...@<revision>`), or seal a new baseline at this commit.
- **`the baseline does not contain suites: [...]`.** The baseline predates a
  suite named with `--suite`. Drop the option, or seal a new baseline.
- **`declarative plan ... no longer matches its audited hash`.** Run
  `python -m fairchem.core.regression_suite.suites.plan_digest` on both machines and
  compare the output; it reports the structure and rotation digests that plan
  hashes depend on and which plans load.
- **`ProxyError: 403 Forbidden` while loading the model.** Set
  `HF_HUB_OFFLINE=1` to use the cached checkpoint.
- **Slow local runs.** Other processes may share the GPU; check `nvidia-smi`.
  Setting `TORCHINDUCTOR_CACHE_DIR` to a persistent directory lets repeated
  compiled runs reuse kernels.

### CI coverage

CI runs these tests on their own CPU runners, the `regression` shard of the
`test` job, for each supported Python version. Unit tests use a fake harmonic
calculator and mock the model, SLURM, and W&B.
`tests/core/regression_suite/test_end_to_end.py` (marked `serial`) also runs
the real command line end to end on the CPU with a seeded random-weight UMA of
about 4,000 parameters (`tiny_model.py`), in a few minutes:

- `quick fp32 --reference fp64` must report `OK` for every standard suite;
- `baseline --scope smoke` must seal a smoke baseline on the CPU with its
  verification reports, `full fp32` and `fast fp32` must pass against it, and
  `full fp32` with a checkpoint written from another seed must fail.

Launches export a snapshot of the working tree instead of a commit, so the
tests also cover uncommitted changes. NVE MD stays out of these tests; even
the tiny model needs about 30 minutes per profile on the CPU.

### Debugging one task

`run` executes one suite, profile, and role in process with Hydra overrides:

```bash
python -m fairchem.core.regression_suite run suite=elastic role=calibration \
  profile=fp32_eager checkpoint=uma-s-1p2p1 output_dir=/tmp/elastic-debug run_id=debug-1
```

Each task compiles into its own cache under `SLURM_TMPDIR` or the system
temporary directory. A prebuilt Triton `cuda_utils` helper is copied from
`~/.triton/cache` (or `FAIRCHEM_TRITON_SEED_CACHE_DIR`) because some compute
images lack Python development headers. Explicit `TORCHINDUCTOR_CACHE_DIR` and
`TRITON_CACHE_DIR` values are respected.
