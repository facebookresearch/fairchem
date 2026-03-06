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

## UMA 1.2

:::{admonition} Latest Release
:class: tip

**~50% faster, ~40% more accurate on Open Molecules test set, and expanded data coverage!**
:::

### Model Highlights

::::{grid} 2
:gutter: 3

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
| **omat** | [OMat24](https://arxiv.org/abs/2410.12771) | Inorganic Materials | No update |
| **omol** | [OMol-1.0](https://arxiv.org/abs/2505.08762) | Organic Molecules | 📊 **Updated** — polymers (OPoly26), ionization potentials, small molecule clusters, dimers |
| **odac** | [ODAC23](https://arxiv.org/abs/2311.00341) | MOFs for Direct Air Capture | No update |
| **omc** | [OMC25](https://arxiv.org/abs/2508.02651) | Molecular Crystals | No update |
| **oc25** | [OC25](https://arxiv.org/abs/2509.17862) | Electrolyte/Inorganic Interfaces | 🆕 **New** |
| **oc22** | [OC22](https://arxiv.org/abs/2206.08917) | Oxide Catalysts | 🆕 **New** |

### Large-Scale Inference & Molecular Dynamics

:::{note}
UMA is built to easily scale up to multi-node, multi-GPU parallel inference with [Ray](https://www.ray.io/) under the hood.
:::

::::{grid} 2
:gutter: 3

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
