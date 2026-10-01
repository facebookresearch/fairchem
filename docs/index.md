---
title: FAIR Chemistry
site:
  hide_outline: true
  hide_toc: true
  hide_title_block: true
---

+++ {"class": "col-page-inset"}

```{image} assets/fair-chemistry-logo-light-mode.png
:alt: FAIR Chemistry
:width: 600px
:align: center
:class: dark:hidden
```

```{image} assets/fair-chemistry-logo-dark-mode.png
:alt: FAIR Chemistry
:width: 600px
:align: center
:class: hidden dark:block
```

## Open data and universal models for atomic systems

FAIR Chemistry develops open datasets and machine-learning models for
molecules, materials, and catalysts. **UMA** brings these domains together in
one pretrained model while preserving each dataset's level of theory.

```{code-block} bash
:filename: Install
pip install fairchem-core
```

{button}`Understand FAIR Chemistry → <./core/introduction.md>`
{button}`Run your first calculation → <./core/quickstart.md>`

+++ {"class": "col-page-inset"}

## From open datasets to UMA

UMA learns from more than 500 million density functional theory calculations.
At inference time, you choose a task that matches your chemistry domain and
the corresponding level of theory.

:::::{grid} 1 2 3 5
::::{card} Heterogeneous catalysis
:link: catalysts/datasets/summary.md

```{image} assets/icons/catalysis.svg
:alt: Catalysis
:width: 60px
:align: center
```

Surface reactions, adsorption, and catalyst design.
+++
**Datasets:** OC20, OC22, OC25<br>
**UMA tasks:** `oc20`, `oc22`, `oc25`
::::

::::{card} Inorganic materials
:link: inorganic_materials/datasets/summary.md

```{image} assets/icons/inorganic.svg
:alt: Inorganic materials
:width: 60px
:align: center
```

Bulk materials, phonons, and elastic properties.
+++
**Dataset:** OMat24<br>
**UMA task:** `omat`
::::

::::{card} Molecules & polymers
:link: molecules/datasets/summary.md

```{image} assets/icons/molecules.svg
:alt: Molecules and polymers
:width: 60px
:align: center
```

Conformers, reactions, and electronic properties.
+++
**Dataset:** OMol25<br>
**UMA task:** `omol`
::::

::::{card} Molecular crystals
:link: molecules/datasets/omc25.md

```{image} assets/icons/molecular-crystals.svg
:alt: Molecular crystals
:width: 60px
:align: center
```

Packed organic molecules in crystal structures.
+++
**Dataset:** OMC25<br>
**UMA task:** `omc`
::::

::::{card} MOFs for direct air capture
:link: dac/datasets/summary.md

```{image} assets/icons/mofs-dac.svg
:alt: Metal-organic frameworks for direct air capture
:width: 60px
:align: center
```

CO₂ and H₂O adsorption in metal-organic frameworks.
+++
**Datasets:** ODAC23, ODAC25<br>
**UMA task:** `odac`
::::
:::::

:::{card} UMA: one model family, multiple levels of theory
UMA uses a learned task embedding to apply one pretrained model across these
domains without mixing their reference methods. Start with
[`uma-s-1p2p1`](./core/uma.md), the fastest current UMA model with
state-of-the-art accuracy on most supported benchmarks.
:::

+++ {"class": "col-page-inset"}

## See what UMA can do

::::{grid} 1 2 2 2
:::{card} Interactive UMA playground
:link: https://aidemos.atmeta.com/uma?view=playground

Explore atomistic simulations without installing anything.

<img src="https://gist.githubusercontent.com/rayg1234/bc9c41122ee5faa546b561923ec5d477/raw/b088b0eaf5f0253966458094afa3f6cac8f72b8d/uma_playground_demo.gif" alt="Animation of the interactive UMA playground" width="100%">

+++
[Open the playground →](https://aidemos.atmeta.com/uma?view=playground)
:::

:::{card} Faster molecular simulation
:link: ./core/uma_changelog.md

```{image} https://raw.githubusercontent.com/facebookresearch/fairchem/main/benchmarks/omol/omol_force_mae_vs_speed.png
:alt: Force accuracy versus molecular-dynamics runtime for UMA and other models.
:width: 100%
:align: center
```

+++
[Read the benchmark details →](https://github.com/facebookresearch/fairchem/tree/main/benchmarks/omol)
:::
::::

+++ {"class": "col-page-inset"}

## Choose your next step

:::::{grid} 1 2 3 3
::::{card} Install FAIR Chemistry
:link: ./core/install.md

Set up `fairchem-core` and request access to UMA checkpoints.
+++
[Install →](./core/install.md)
::::

::::{card} Hello World
:link: ./core/quickstart.md

Run a molecular calculation and relax an inorganic crystal.
+++
[Get started →](./core/quickstart.md)
::::

::::{card} Explore UMA
:link: ./core/introduction.md#what-you-can-do-with-uma

Find the right task and workflow for your scientific problem.
+++
[Explore capabilities →](./core/introduction.md#what-you-can-do-with-uma)
::::

::::{card} Model guide
:link: ./core/uma.md

Understand UMA tasks, inputs, architecture, and limitations.
+++
[Read the guide →](./core/uma.md)
::::

::::{card} Common workflows
:link: ./core/common_tasks/summary.md

Scale from ASE calculations to training and batched inference.
+++
[Browse workflows →](./core/common_tasks/summary.md)
::::

::::{card} Learning resources
:link: ./videos.md

Watch introductory videos and technical presentations.
+++
[Start learning →](./videos.md)
::::
:::::

Copyright © Meta Platforms, Inc | [Terms of Use](https://opensource.fb.com/legal/terms) | [Privacy Policy](https://opensource.fb.com/legal/privacy)
