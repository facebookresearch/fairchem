---
title: FAIRChem
site:
  hide_outline: true
  hide_toc: true
  hide_title_block: true
---

+++ {"class": "col-page-inset"}

```{image} assets/fair_chem_logo_v2.png
:alt: FAIRChem Logo
:width: 600px
:align: center
```

::::::{grid} 1 2 2 2
:::::{grid-item}
**Machine learning models for materials science and quantum chemistry.**

_State-of-the-art universal interatomic potentials for molecules, materials, and catalysts — built by the [Meta FAIR](https://ai.meta.com/research/) Chemistry team._

<p>
<a href="https://github.com/facebookresearch/fairchem/actions/workflows/test.yml"><img src="https://github.com/facebookresearch/fairchem/actions/workflows/test.yml/badge.svg?branch=main" alt="tests"></a>
<a href="https://pypi.org/project/fairchem-core/"><img src="https://img.shields.io/pypi/v/fairchem-core" alt="PyPI"></a>
<img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+">
</p>

:::::
:::::{grid-item}
```{code-block} bash
:filename: install
pip install fairchem-core
```

{button}`Get Started → <./core/quickstart.md>` {button}`Try the Demo → <https://facebook-fairchem-uma-demo.hf.space/>`



::::::



+++ {"kind": "justified"}

## Application Domains

:::::{grid} 1 2 3 5
::::{card} Heterogeneous Catalysis
:link: catalysts/datasets/summary.md

```{image} assets/icons/catalysis.svg
:alt: Catalysis
:width: 60px
:align: center
```

Surface reactions, adsorption energies, and catalyst design.
+++
[Explore Catalysis →](catalysts/datasets/summary.md)
::::

::::{card} Inorganic Materials
:link: inorganic_materials/datasets/summary.md

```{image} assets/icons/inorganic.svg
:alt: Inorganic Materials
:width: 60px
:align: center
```

Bulk materials, phonons, and elastic properties.
+++
[Explore Materials →](inorganic_materials/datasets/summary.md)
::::

::::{card} Molecules & Polymers
:link: molecules/datasets/summary.md

```{image} assets/icons/molecules.svg
:alt: Molecules & Polymers
:width: 60px
:align: center
```

Molecular conformers and electronic properties.
+++
[Explore Molecules →](molecules/datasets/summary.md)
::::

::::{card} Molecular Crystals
:link: molecules/datasets/omc25.md

```{image} assets/icons/molecular-crystals.svg
:alt: Molecular Crystals
:width: 60px
:align: center
```

Packed molecular arrangements in crystal structures.
+++
[Explore Crystals →](molecules/datasets/omc25.md)
::::

::::{card} MOFs for Direct Air Capture
:link: dac/datasets/summary.md

```{image} assets/icons/mofs-dac.svg
:alt: MOFs for DAC
:width: 60px
:align: center
```

Metal-organic frameworks for CO₂ capture.
+++
[Explore DAC →](dac/datasets/summary.md)
::::
:::::

+++ {"kind": "justified"}

## Get Started

:::::{grid} 1 2 3 3
::::{card} 📘 Quickstart
:link: ./core/quickstart.md

Build your first calculation in minutes
+++
[Start now →](./core/quickstart.md)
::::

::::{card} 📖 UMA Model Guide
:link: ./core/uma.md

Learn about task selection and model inputs
+++
[Read guide →](./core/uma.md)
::::

::::{card} 🔧 Installation
:link: ./core/install.md

Set up FAIRChem and HuggingFace access
+++
[Install →](./core/install.md)
::::

::::{card} 🎓 Tutorials
:link: ./uma_tutorials/summary.md

Hands-on tutorials for common workflows
+++
[View tutorials →](./uma_tutorials/summary.md)
::::

::::{card} 📊 Common Tasks
:link: ./core/common_tasks/summary.md

Inference, training, fine-tuning, and more
+++
[Browse tasks →](./core/common_tasks/summary.md)
::::

::::{card} 🎬 Technical Talks
:link: ./videos/technical_talks.md

Video presentations from the team
+++
[Watch →](./videos/technical_talks.md)
::::
:::::

+++ {"kind": "justified"}

## The UMA Model

Read about our latest release: the **Universal Machine-learning for Atomistic systems (UMA)** model.

:::{card} Meta FAIR Science Release
:link: https://ai.meta.com/blog/meta-fair-science-new-open-source-releases/

```{image} https://github.com/user-attachments/assets/acddd09b-ed6f-4d05-9a4b-9ba5e2301150
:alt: Meta FAIR Science Release
:width: 100%
```

UMA is trained on 500M+ DFT calculations across molecules, materials, and catalysts — achieving state-of-the-art accuracy with energy conservation and fast inference.
:::

:::{attention} FAIRChem v2 is here!
FAIRChem v2 introduces the **UMA model** — a universal machine learning potential for atoms.
This is a breaking change from v1 and is not compatible with previous pretrained models.

[Migration Guide](core/fairchemv1_v2.md) • [Version 1 Documentation](https://pypi.org/project/fairchem-core/1.10.0/)
:::

+++ {"kind": "justified"}

## Resources

::::{grid} 1 1 2 2
:::{card} GitHub Repository
:link: https://github.com/facebookresearch/fairchem

Browse source code, report issues, and contribute
+++
[facebookresearch/fairchem](https://github.com/facebookresearch/fairchem)
:::

:::{card} Papers Using FAIRChem
:link: ./core/fair_chemistry_papers.md

See how researchers are using our models
+++
[View publications →](./core/fair_chemistry_papers.md)
:::

:::{card} Open in Codespaces
:link: https://github.com/codespaces/new/facebookresearch/fairchem?quickstart=1

Try FAIRChem in your browser — no local setup required
+++
[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://github.com/codespaces/new/facebookresearch/fairchem?quickstart=1)
:::

:::{card} HuggingFace Models
:link: https://huggingface.co/facebook/UMA

Access pretrained UMA checkpoints
+++
[facebook/UMA](https://huggingface.co/facebook/UMA)
:::
::::

[Terms of Use](https://opensource.fb.com/legal/terms) | [Privacy Policy](https://opensource.fb.com/legal/privacy)
