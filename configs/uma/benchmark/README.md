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
