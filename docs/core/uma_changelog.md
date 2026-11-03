---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.1
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# UMA Release Changelog

This page documents the release history of UMA models, including new features, improvements, and bug fixes.

---

## Library compatibility — UMA 1.0 deprecation

UMA 1.0 checkpoints are no longer supported (their MoE `include_self` behavior diverges from later releases); loading one raises a `RuntimeError`. To use a 1.0 checkpoint, `pip install 'fairchem-core<=2.21.0'`.

Loading now requires a `model_id` to be defined. UMA 1.1 (which ships without one) is auto-tagged `model_id = "UMA-1.1"` in memory; UMA 1.2+ already carry `model_id`; a checkpoint with neither `model_id` nor `backbone.model_version` is rejected. When training, set `model_id` on the model config (e.g. `model_id: UMA-1.2.1`).

---

## UMA-S-1.2.1-FT-D4rev and UMA-S-1.2.1-FT-r2SCAN — October 2026

:::{admonition} New models
:class: tip

**Two fine-tunes of UMA-S-1.2.1 at higher levels of theory, both served by UMA Nitro.**
:::

- `uma-s-1p2p1-ft-d4rev`: `omol` task at the ωB97M-D4rev level of theory, fine-tuned on approximately 4 million OMol25 structures recomputed with ωB97M-D4rev labels.
- `uma-s-1p2p1-ft-r2scan`: `omat` task at the r²SCAN level of theory, trained on [MAD-1.6](https://arxiv.org/abs/2603.02089).

See [the UMA models page](uma.md#uma-s-1p2p1-fine-tunes) for usage, limitations and acknowledgements.

---

## UMA Nitro and D3 dispersion — October 2026

- **UMA Nitro** is the default inference mode and is much faster than turbo for single-GPU MD and relaxations. It runs `uma-s-1p2p1`, `uma-s-1p2p1-ft-d4rev` and `uma-s-1p2p1-ft-r2scan` on Ampere, Hopper and Blackwell GPUs. Special thanks to Emanuele Arcelli and Max Zhdanov (for their [FlashUMA work](https://github.com/ArcelliEmanuele/flash-uma)), Carole-Jean Wu, Christian Puhrsch, Zachary DeVito, Aditya Venkataraman, Michael Lazos, and Animesh Jain, all of whom contributed to and helped improve UMA Nitro.
  On a single NVIDIA H200, it reaches **~6 ns/day for 1,000 atoms** in ASE MD
  and raw inference throughput equivalent to **~33 ns/day for 150 atoms**,
  with support for systems up to **15,000 atoms**.
- **`DFTD3Calculator`** adds GPU DFT-D3(BJ) energies, forces and analytic stresses to any ASE calculator, with `pbe` and `r2scan` presets. See [Adding a DFT-D3(BJ) dispersion correction](common_tasks/ase_calculator.md#adding-a-dft-d3-bj-dispersion-correction).

---

## Library change — the default inference mode is nitro, and turbo is no longer available

- `inference_settings="default"`, the default for `get_predict_unit`, `load_predict_unit` and `FAIRChemCalculator.from_model_checkpoint`, is now the `nitro` preset, which we recommend for MD and other standard workflows. It runs UMA Nitro for `uma-s-1p2p1` (and the fine-tunes of it on Nitro's checkpoint allowlist) on Ampere, Hopper and Blackwell GPUs, and `multigpu` otherwise, warning once that results are reduced precision.
- The previous default (full FP32 with `merge_mole + compile`) is now `inference_settings="high-precision"`.
- **Breaking:** `turbo` is no longer available; please use `nitro` (the default). Its former settings are `inference_settings="multigpu"`, which multi-worker inference (`workers > 1`) uses.
- The batch server's `ModelSpec` and the torch-sim interface default to `"batch"`.
- Fallback warnings name the settings that replace the requested ones.
- On Linux x86_64 and aarch64, `nvidia-cutlass-dsl` and `cuda-python` are now installed with `fairchem-core`.

---

## UMA 1.2

:::{admonition} Latest Release
:class: tip

**~50% faster, ~40% more accurate on Open Molecules test set, and expanded data coverage!**
:::

### Model Highlights

::::{grid} 2

:::{grid-item-card} 🔋 Better Charge Handling
Leading to improvements across molecular systems:
- **~50%** relative improvement on biomolecules vs UMA-S-1.1
- **~30%** relative improvement on electrolytes vs UMA-S-1.1
:::

:::{grid-item-card} 📊 Expanded Data Coverage
Now **~520M total DFT calculations** including:
- OC22 (oxide catalysts)
- OC25 (electrolyte/inorganic interfaces)
- Expanded OMol25 data including polymers (OPoly26)
:::

:::{grid-item-card} ⚡ Faster Inference
**~50% speedup** for UMA-S with turbo mode compared to previous releases
:::

:::{grid-item-card} 🔧 Bug Fixes & Stability
- Numerical and stability improvements for Hessians and phonons
- More accurate diatomics and ionization potentials
:::

::::

### Available Tasks (7 total)

| Task | Dataset | Domain | Status |
|------|---------|--------|--------|
| **oc20** | [OC20](https://arxiv.org/abs/2010.09990) | Heterogeneous Catalysis | No update |
| **omat** | [OMat24](https://arxiv.org/abs/2410.12771) | Inorganic Materials | 📊 **Updated** — dimer data |
| **omol** | [OMol-1.0](https://arxiv.org/abs/2505.08762) | Organic Molecules | 📊 **Updated** — polymers (OPoly26), ionization potentials, small molecule clusters, dimers |
| **odac** | [ODAC23](https://arxiv.org/abs/2311.00341) | MOFs for Direct Air Capture | No update |
| **omc** | [OMC25](https://arxiv.org/abs/2508.02651) | Molecular Crystals | No update |
| **oc25** | [OC25](https://arxiv.org/abs/2509.17862) | Electrolyte/Inorganic Interfaces | 🆕 **Added to UMA** |
| **oc22** | [OC22](https://arxiv.org/abs/2206.08917) | Oxide Catalysts | 🆕 **Added to UMA** |

### Large-Scale Inference & Molecular Dynamics

:::{note}
UMA is built to easily scale up to multi-node, multi-GPU parallel inference with [Ray](https://www.ray.io/) under the hood.
:::

::::{grid} 2

:::{grid-item-card} 🚀 Large-Scale MD Simulations
Run MD with ASE and LAMMPS on large-scale systems with ns/day speeds:
- Battle tested up to **hundreds of GPUs**
- Systems up to **1M atoms**
- UMA handles all parallelism—no need to manually configure LAMMPS, Kokkos, MPI, CUDA, etc.
:::

:::{grid-item-card} 📦 Batched Simulations
Client–server Ray framework enables batched simulations:
- Structural relaxations
- Molecular dynamics
- **Up to 4x speedup** over serial calculations for batches of small systems on a single H100 GPU
:::

::::

---

## UMA 1.1

:::{admonition} Bug Fix Release
:class: warning

Fixed bug with size extensivity.
:::

### Available Tasks (5 total)

| Task | Dataset | Domain | Status |
|------|---------|--------|--------|
| **oc20** | [OC20](https://arxiv.org/abs/2010.09990) | Heterogeneous Catalysis | No update |
| **omat** | [OMat24](https://arxiv.org/abs/2410.12771) | Inorganic Materials | No update |
| **omol** | [OMol25](https://arxiv.org/abs/2505.08762) | Organic Molecules | No update |
| **odac** | [ODAC23](https://arxiv.org/abs/2311.00341) | MOFs for Direct Air Capture | No update |
| **omc** | [OMC25](https://arxiv.org/abs/2508.02651) | Molecular Crystals | No update |

---

## UMA 1.0

:::{admonition} Initial Release — May 2025
:class: note

The first public release of UMA, introducing a unified model for atoms across multiple domains.
:::

### Available Tasks (5 total)

| Task | Dataset | Domain | Status |
|------|---------|--------|--------|
| **oc20** | [OC20](https://arxiv.org/abs/2010.09990) | Heterogeneous Catalysis | 🆕 **New** |
| **omat** | [OMat24](https://arxiv.org/abs/2410.12771) | Inorganic Materials | 🆕 **New** |
| **omol** | [OMol25](https://arxiv.org/abs/2505.08762) | Organic Molecules | 🆕 **New** |
| **odac** | [ODAC23](https://arxiv.org/abs/2311.00341) | MOFs for Direct Air Capture | 🆕 **New** |
| **omc** | [OMC25](https://arxiv.org/abs/2508.02651) | Molecular Crystals | 🆕 **New** |

### Key Features

- **Mixture-of-Linear-Experts (MoLE)** architecture for high parameter count with fast inference
- **Equivariant GNN** preserving energy conservation
- **Multi-domain training** on 500M+ DFT calculations
- **ASE Calculator integration** for seamless workflow adoption
