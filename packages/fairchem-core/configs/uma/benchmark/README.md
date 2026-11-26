Running UMA Benchmarks
-----------------------
## List of benchmarks

### Materials
- Kappa SRME
- MDR Phonons
- MP binary PBE elasticity
- MP PBE elasticity
- HEA IS2RE
- NVE MD conservation TM23

### OMC
- OMC S2E Polymorphs
- OMC IS2RE Polymorphs

### Catalysis
- OC20 S2EF Adsorption
- OC20 IS2RE
- AdsorbML

### Molecules
- NVE MD conservation MD22
- OMol liquid density (62 experimental liquids, 1 ns NPT)
- GSCDB

## Running Benchmarks

#### To run matbench-discovery / phonon benchmark / kSRME, install the following requirements first:

```bash
pip install git+https://github.com/janosh/matbench-discovery.git@0ae0a46ce767f12c252340970f1285b1c2d3fe23
pip install phonopy==2.38.0
pip install phono3py==3.15.0
pip install moyopy
```

#### Running OMC benchmarks requires scikit-learn and scipy
```bash
pip install scipy==1.14.1
pip install scikit-learn==1.6.1
```

#### To run a benchmark just call:
```bash
fairchem -c configs/uma/benchmark/oc20-s2ef.yaml
```

#### Running different checkpoints and/or different clusters
If you want to use a different model / are on a different cluster (e.g. V100):

```
fairchem -c configs/uma/benchmark/mp-pbe-elasticity.yaml checkpoint=uma_sm cluster=v100
```

## Materials benchmarks:
```bash
fairchem -c configs/uma/benchmark/kappa103.yaml checkpoint=uma_sm_mpa cluster=h100
fairchem -c configs/uma/benchmark/mdr-phonon.yaml checkpoint=uma_sm_mpa cluster=h100
fairchem -c configs/uma/benchmark/mp-binary-pbe-elasticity.yaml checkpoint=uma_sm_mpa cluster=h100
fairchem -c configs/uma/benchmark/mp-pbe-elasticity.yaml checkpoint=uma_sm_mpa cluster=h100
```
##### Default on V100 to use more jobs:

```bash
fairchem -c configs/uma/benchmark/matbench-discovery-discovery.yaml checkpoint=uma_sm_mpa cluster=v100
```

##### Using OMat head (not MPA!)
```bash
fairchem -c configs/uma/benchmark/hea-is2re.yaml checkpoint=uma_sm cluster=h100
```

## OMC benchmarks

##### On h100
```bash
fairchem -c configs/uma/benchmark/omc-s2e-polymorphs.yaml checkpoint=uma_sm cluster=h100
```
##### On v100

```bash
fairchem -c configs/uma/benchmark/omc-is2re-polymorphs.yaml checkpoint=uma_sm cluster=v100
fairchem -c configs/uma/benchmark/omc-is2re-10k.yaml checkpoint=uma_sm cluster=v100
```

## Catalysis benchmarks
```bash
fairchem -c configs/uma/benchmark/oc20-s2ef-id.yaml checkpoint=uma_sm cluster=h100
fairchem -c configs/uma/benchmark/oc20-s2ef-ood-both.yaml checkpoint=uma_sm cluster=h100
fairchem -c configs/uma/benchmark/oc20-is2re-adsorption.yaml checkpoint=uma_sm cluster=h100
fairchem -c configs/uma/benchmark/adsorbml.yaml checkpoint=uma_sm cluster=h100
```

## NVE MD conservation
```bash
fairchem -c configs/uma/benchmark/nvemd-materials.yaml checkpoint=uma_sm cluster=h100
fairchem -c configs/uma/benchmark/nvemd-molecules.yaml checkpoint=uma_sm cluster=h100
```

## OMol liquid density

The liquid-density benchmark launches one independent NPT simulation for each
of 62 liquids, then runs a reducer that compares the predicted densities with
experiment and logs aggregate and per-system results to W&B.

```bash
fairchem -c configs/uma/benchmark/omol-liquid-density.yaml checkpoint=uma_sm cluster=h200
```

The default protocol is 1 ns with a 0.5 fs timestep. The first 30% is discarded
as equilibration and density is sampled every 100 steps. Protocol values are
normal Hydra parameters and can be overridden from the command line:

```bash
fairchem -c configs/uma/benchmark/omol-liquid-density.yaml \
  runner.timestep_fs=0.5 runner.total_time_ps=500
```

For a short local smoke test, process the complete catalog in one job and
override the runtime:

```bash
fairchem -c configs/uma/benchmark/omol-liquid-density.yaml \
  cluster=h100_local job.scheduler.num_array_jobs=1 runner.total_time_ps=1
```

The data are provisioned outside the repository under
`${cluster.data_root_dir}/molecules/liquid_density/`:

- `liquid_density_inputs.json` contains a versioned list of systems. Each entry
  stores the system ID, ASE-encoded atoms, representative one-molecule SMILES,
  molecule and atom counts, experimental temperature, and experimental density.
- `liquid_density_references.pkl` contains the authoritative experimental
  density keyed by the same system IDs. The reducer verifies that these values
  match the duplicated JSON metadata before scoring.

The reducer writes a complete per-system results table plus density MAE, RMSE,
mean and maximum absolute relative errors, signed bias, and success/failure
counts. It also saves and logs a predicted-versus-experimental density parity
plot. Full trajectories are not retained.

## GSCDB

Place the pinned `gscdb_inputs.pkl` artifact at
`${cluster.data_root_dir}/gscdb_inputs.pkl`, then run:

```bash
fairchem -c configs/uma/benchmark/gscdb.yaml checkpoint=uma_sm cluster=h100
```

For a file elsewhere, add
`gscdb_input_data=/path/to/gscdb_inputs.pkl` to the command. The artifact
contains all ASE structures and verbatim copies of the benchmark's `Info/*.csv`
metadata, so running the evaluation does not clone or read a mutable GitHub
checkout. Its C60ISO references incorporate the nine revised values from Table 1
of Reimann and Kaupp, DOI `10.1021/acs.jpclett.6c01636`; the two reactions that
GSCDB assigns to the excluded `SC74` set remain excluded from metrics. The
artifact embeds a `Reference_updates.csv` provenance manifest. The pinned
artifact contains 14,006 structures and has SHA256
`4c84fb3e9109a18b26ff49bdeb334d7b66d4707280370e891d90c94e2174b892`.

The reducer writes raw molecular energies to `gscdb_results.json.gz`, signed
errors to `gscdb_reaction_errors.csv`, and full-distribution category metrics to
`gscdb_metrics_by_category.csv`. The reported metrics are MAE, RMSE, mean signed
error, standard deviation, and 95th-percentile absolute error; no reactions are
removed by percentile trimming. The reaction-error CSV retains all 8,448 GSCDB
rows, including unevaluated property rows with missing predictions. Metrics omit
the non-energy and historically excluded datasets configured in
`metric_excluded_datasets`.
