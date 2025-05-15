Running PUMA Benchmarks
-----------------------
## List of benchmarks

### Materials
- Kappa SRME
- MDR Phonons
- MP binary PBE elasticity
- MP PBE elasticity
- HEA IS2RE
- NVE MD conservation TM23

### OSC
- OSC S2E Polymorphs
- OSC IS2RE Polymorphs

### Catalysis
- OC20 S2EF Adsorption
- OC20 IS2RE
- AdsorbML

### Molecules
- NVE MD conservation MD22

## Running Benchmarks

#### To run matbench-discovery / phonon benchmark / kSRME, install the following requirements first:

```bash
pip install git+https://github.com/janosh/matbench-discovery.git@0ae0a46ce767f12c252340970f1285b1c2d3fe23
pip install phonopy==2.38.0
pip install phono3py==3.15.0
pip install moyopy
```

#### Running OSC benchmarks requires scikit-learn and scipy
```bash
pip install scipy==1.14.1
pip install scikit-learn==1.6.1
```

#### To run a benchmark just call:
```bash
fairchem -c configs/puma/benchmark/oc20-s2ef.yaml
```

#### Running different checkpoints and/or different clusters
If you want to use a different model / are on a different cluster (e.g. V100):

```
fairchem -c configs/puma/benchmark/mp-pbe-elasticity.yaml checkpoint=puma_sm cluster=v100
```

## Materials benchmarks:
```bash
fairchem -c configs/puma/benchmark/kappa103.yaml checkpoint=puma_sm_mpa_0428 cluster=h100
fairchem -c configs/puma/benchmark/mdr-phonon.yaml checkpoint=puma_sm_mpa_0428 cluster=h100
fairchem -c configs/puma/benchmark/mp-binary-pbe-elasticity.yaml checkpoint=puma_sm_mpa_0428 cluster=h100
fairchem -c configs/puma/benchmark/mp-pbe-elasticity.yaml checkpoint=puma_sm_mpa_0428 cluster=h100
```
##### Default on V100 to use more jobs:

```bash
fairchem -c configs/puma/benchmark/matbench-discovery-discovery.yaml checkpoint=puma_sm_mpa_0428 cluster=v100
```

##### Using OMat head (not MPA!)
```bash
fairchem -c configs/puma/benchmark/hea-is2re.yaml checkpoint=puma_sm cluster=h100
```

## OSC benchmarks

##### On h100
```bash
fairchem -c configs/puma/benchmark/osc-s2e-polymorphs.yaml checkpoint=puma_sm cluster=h100
```
##### On v100

```bash
fairchem -c configs/puma/benchmark/osc-is2re-polymorphs.yaml checkpoint=puma_sm cluster=v100
fairchem -c configs/puma/benchmark/osc-is2re-10k.yaml checkpoint=puma_sm cluster=v100
```

## Catalysis benchmarks
```bash
fairchem -c configs/puma/benchmark/oc20-s2ef-id.yaml checkpoint=puma_sm cluster=h100
fairchem -c configs/puma/benchmark/oc20-s2ef-ood-both.yaml checkpoint=puma_sm cluster=h100
fairchem -c configs/puma/benchmark/oc20-is2re-adsorption.yaml checkpoint=puma_sm cluster=h100
fairchem -c configs/puma/benchmark/adsorbml.yaml checkpoint=puma_sm cluster=h100
```

## NVE MD conservation
```bash
fairchem -c configs/puma/benchmark/nvemd_materials.yaml checkpoint=puma_sm cluster=h100
fairchem -c configs/puma/benchmark/nvemd_molecules.yaml checkpoint=puma_sm cluster=h100
```
