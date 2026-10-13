# Introduction to FAIR Chemistry

FAIR Chemistry is Meta FAIR's open ecosystem for machine learning in
atomistic simulation. It brings together large quantum-chemistry datasets,
pretrained models, and tools that connect those models to familiar simulation
workflows.

The central idea is simple: expensive density functional theory (DFT)
calculations can be used to train machine-learned interatomic potentials. Once
trained, those models estimate energies and forces much faster, making it
possible to explore more structures and longer trajectories before confirming
the most important results with higher-fidelity methods.

## How the pieces fit together

```{image} ../assets/uma-diagram-light-mode.png
:alt: UMA connects several chemistry domains through one universal model.
:width: 700px
:align: center
:class: dark:hidden
```

```{image} ../assets/uma-diagram-dark-mode.png
:alt: UMA connects several chemistry domains through one universal model.
:width: 700px
:align: center
:class: hidden dark:block
```

FAIR Chemistry provides three connected pieces:

1. **Open datasets** contain atomistic structures and DFT labels for distinct
   chemistry domains.
2. **UMA** is a family of Universal Models for Atoms pretrained across those
   domains.
3. **`fairchem`** connects UMA to tools such as ASE, LAMMPS, and quacc for
   calculations and simulations.

## One model, several tasks

Each dataset was calculated with a particular scientific method and set of
approximations. UMA preserves those distinctions through a **task** input. You
select the task that matches your system and the level of theory you want UMA
to emulate.

| Domain | Representative training data | UMA task | Example uses |
| --- | --- | --- | --- |
| Organic molecules and polymers | OMol25 | `omol` | Conformers, reactions, molecular dynamics |
| Inorganic materials | OMat24 | `omat` | Relaxations, phonons, elastic properties |
| Heterogeneous catalysts | OC20, OC22, OC25 | `oc20`, `oc22`, `oc25` | Adsorption, surfaces, reaction pathways |
| Molecular crystals | OMC25 | `omc` | Crystal packing and polymorph ranking |
| MOFs and direct air capture | ODAC23 | `odac` | CO₂ and H₂O adsorption |

Task selection matters because predictions from different tasks generally
represent different DFT levels of theory. They should not be mixed in one
energy comparison without careful validation. See the [UMA model
guide](./uma.md) for task-specific caveats.

## What you can do with UMA

UMA provides energies, forces, and—for supported periodic tasks—stresses
through the standard ASE calculator interface. Those predictions can drive
many atomistic workflows without changing models as you move between domains.

:::{tip}
Start with [`uma-s-1p2p1`](./uma.md) and choose the task that matches the
dataset and level of theory relevant to your system.
:::

::::{grid} 1 2 2 2
:::{card} Molecules and polymers · `omol`
Calculate conformer energies, spin gaps, vibrations, and molecular dynamics.

[Explore molecular data →](../molecules/datasets/summary.md)
:::

:::{card} Inorganic materials · `omat`
Relax atomic positions and cells, calculate elastic properties, and construct
phonon spectra.

[Explore materials tutorials →](../inorganic_materials/examples_tutorials/summary.md)
:::

:::{card} Heterogeneous catalysts · `oc20`, `oc22`, `oc25`
Study adsorption, surface stability, reaction thermochemistry, and transition
states with the task appropriate to the interface.

[Explore catalysis tutorials →](../catalysts/examples_tutorials/summary.md)
:::

:::{card} Molecular crystals · `omc`
Score periodic molecular crystals and support crystal-structure prediction
workflows.

[Explore OMC25 →](../molecules/datasets/omc25.md)
:::

:::{card} MOFs and direct air capture · `odac`
Estimate adsorption energies and study framework deformation for CO₂ and H₂O.

[Explore the adsorption tutorial →](../dac/examples_tutorials/adsorption_energy.md)
:::

:::{card} Scaled simulation
Use batched inference, multiple GPUs, LAMMPS, or workflow engines for larger
and more numerous simulations.

[Browse common workflows →](./common_tasks/summary.md)
:::
::::

## A typical workflow

1. Install `fairchem-core` and obtain access to the gated UMA repository.
2. Create or load an atomic structure as an ASE `Atoms` object.
3. Load `uma-s-1p2p1` and select the appropriate task.
4. Attach a `FAIRChemCalculator` to the structure.
5. Run an energy, force, relaxation, dynamics, or downstream-property
   calculation.
6. Inspect the structure and validate important conclusions against reference
   data or higher-fidelity calculations.

:::{important}
UMA accelerates atomistic modeling; it does not remove the need to check the
training domain, level of theory, physical constraints, and uncertainty for
your application.
:::

## Try UMA without writing code

The [Meta AI Demo Lab UMA playground](https://aidemos.atmeta.com/uma?view=playground)
is the recommended browser-based experience. Use it to manipulate structures
and build intuition before setting up a local workflow.

The separate [guided UMA demo](https://facebook-fairchem-uma-demo.hf.space/)
contains additional worked examples and is maintained outside this repository.

## Where to go next

::::{grid} 1 2 2 2
:::{card} Install
:link: ./install.md
Prepare Python and Hugging Face access.
:::

:::{card} Hello World
:link: ./quickstart.md
Run two small end-to-end calculations.
:::

:::{card} Common workflows
:link: ./common_tasks/summary.md
Scale from ASE calculations to training and batched inference.
:::

:::{card} Playground
:link: https://aidemos.atmeta.com/uma?view=playground
Explore UMA interactively in a browser.
:::
::::
