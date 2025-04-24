to run matbench-discovery / phonon benchmark / kSRME, install the requirements first:

```
git clone https://github.com/janosh/matbench-discovery
git checkout 0ae0a46ce767f12c252340970f1285b1c2d3fe23
pip install -e ./matbench-discovery --config-settings editable-mode=compat
pip install phonopy==2.38.0
pip install phono3py==3.15.0
pip install moyopy
```

run all benchmarks on H100-1:

- OC20 S2EF
- OC20 IS2RE
- Kappa SRME
- MDR Phonon
- MP binary PBE elasticity
- MP PBE elasticity
- NVE MD conservation TM23
- NVE MD conservation MD22

```
fairchem -c configs/puma/benchmark/oc20-s2ef.yaml checkpoint=puma_sm_mpa
fairchem -c configs/puma/benchmark/oc20-is2re-adsorption.yaml checkpoint=puma_sm_mpa
fairchem -c configs/puma/benchmark/kappa103.yaml checkpoint=puma_sm_mpa
fairchem -c configs/puma/benchmark/mdr-phonon.yaml checkpoint=puma_sm_mpa
fairchem -c configs/puma/benchmark/mp-binary-pbe-elasticity.yaml checkpoint=puma_sm_mpa
fairchem -c configs/puma/benchmark/mp-pbe-elasticity.yaml checkpoint=puma_sm_mpa
fairchem -c configs/puma/benchmark/nvemd_materials.yaml checkpoint=puma_sm_mpa cluster=h100
fairchem -c configs/puma/benchmark/nvemd_molecules.yaml checkpoint=puma_sm_mpa cluster=h100
```

default on V100 to use more jobs:

- Matbench-Discovery

```
fairchem -c configs/puma/benchmark/matbench-discovery-discovery.yaml checkpoint=puma_sm_mpa
```

if you want to use a different model / are on a different cluster (e.g. V100):

```
fairchem -c configs/puma/benchmark/mp-pbe-elasticity.yaml checkpoint=puma_sm cluster=v100
```

To be finalized:

```
# Matbench-Discovery
fairchem -c configs/puma/benchmark/matbench-discovery-discovery.yaml
# HEA S2EF
fairchem -c configs/puma/benchmark/hea-s2ef.yaml
# OSC IS2RE: taking too long, needs downsampling
fairchem -c configs/puma/benchmark/osc-is2re.yaml
```
