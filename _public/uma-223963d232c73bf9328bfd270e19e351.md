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

# UMA Models

[UMA](https://ai.meta.com/research/publications/uma-a-family-of-universal-models-for-atoms/) is an equivariant GNN that leverages a novel technique called Mixture of Linear Experts (MoLE) to give it the capacity to learn the largest multi-modal dataset to date (500M DFT examples), while preserving energy conservation and inference speed. Even a 6M active parameter (145M total) UMA model is able to achieve SOTA accuracy on a wide range of domains such as materials, molecules and catalysis. [arXiv Paper](https://arxiv.org/abs/2506.23971)

For most applications, start with `uma-s-1p2p1`. It is the latest patch release
of the small UMA 1.2 model and offers the best default balance of speed,
accuracy, and memory use.

![UMA model architecture](uma.svg "UMA model architecture")

## The UMA Mixture-of-Linear-Experts routing function

:::{note}
The Mixture-of-Linear-Expert (MoLE) architecture is key to UMA's performance. It enables very high parameter count with fast inference speeds by using a single output head with dynamic routing.
:::

The UMA model uses a Mixture-of-Linear-Expert (MoLE) architecture to achieve very high parameter count with fast inference speeds with a single output head. In order to route the model to the correct set parameters, the model must be given a set of inputs. The following information is required for the input to the model:

::::{grid} 2

:::{card} Task
The task specifies the level of theory/DFT calculations to emulate: `omol`, `oc20`, `omat`, `odac`, `omc`
:::

:::{card} Charge
Total known charge of the system (only used for `omol` task, defaults to 0)
:::

:::{card} Spin
Total spin multiplicity of the system (only used for `omol` task, defaults to 1)
:::

:::{card} Elemental Composition
The unordered total elemental composition. Each element has an atom embedding and the composition embedding is the mean over all atom embeddings.
:::

::::


### The UMA task

UMA is trained on 5 different DFT datasets with different levels of theory. An UMA **task** refers to a specific level of theory associated with that DFT dataset. UMA learns an embedding for the given **task**. Thus at inference time, the user must specify which one of the 5 embeddings they want to use to produce an output with the DFT level of theory they want.

:::{tip}
Choose your task based on the application domain. Each task corresponds to a specific DFT level of theory optimized for that domain.
:::

| Task | Dataset | DFT Level of Theory | Relevant Applications | Usage Notes |
|------|---------|---------------------|----------------------|-------------|
| **omol** | [OMol25](https://arxiv.org/abs/2505.08762) | wB97M-V/def2-TZVPD as implemented in ORCA6, including non-local dispersion. All solvation should be explicit. | Biology, organic chemistry, protein folding, small-molecule pharmaceuticals, organic liquid properties, homogeneous catalysis, polymers | Requires total charge and spin multiplicity. If you don't know what these are, you should be very careful if modeling charged or open-shell systems. This can be used to study radical chemistry or understand the impact of magnetic states on the structure of a molecule. All training data is aperiodic, so any periodic systems should be treated with some caution. Probably won't work well for inorganic materials. |
| **omc** | [OMC25](https://arxiv.org/abs/2508.02651) | PBE+D3 as implemented in VASP. | Pharmaceutical packaging, bio-inspired materials, organic electronics, organic LEDs | UMA has not seen varying charge or spin multiplicity for the OMC task, and expects total_charge=0 and spin multiplicity=0 as model inputs. |
| **omat** | [OMat24](https://arxiv.org/abs/2410.12771) | PBE/PBE+U as implemented in VASP using Materials Project suggested settings, except with VASP 54 pseudopotentials. No dispersion. | Inorganic materials discovery, solar photovoltaics, advanced alloys, superconductors, electronic materials, optical materials | UMA has not seen varying charge or spin multiplicity for the OMat task, and expects total_charge=0 and spin multiplicity=0 as model inputs. Spin polarization effects are included, but you can't select the magnetic state. Further, OMat24 did not fully sample possible spin states in the training data. |
| **oc20** | [OC20*](https://arxiv.org/abs/2010.09990) | RPBE as implemented in VASP, with VASP5.4 pseudopotentials. No dispersion. | Renewable energy, catalysis, fuel cells, energy conversion, sustainable fertilizer production, chemical refining, plastics synthesis/upcycling | UMA has not seen varying charge or spin multiplicity for the OC20 task, and expects total_charge=0 and spin multiplicity=0 as model inputs. No oxides or explicit solvents are included in OC20. The model works surprisingly well for transition state searches given the nature of the training data, but you should be careful. RPBE works well for small molecules, but dispersion will be important for larger molecules on surfaces. |
| **odac** | [ODAC23](https://arxiv.org/abs/2311.00341) | PBE+D3 as implemented in VASP, with VASP5.4 pseudopotentials. | Direct air capture, carbon capture and storage, CO2 conversion, catalysis | UMA has not seen varying charge or spin multiplicity for the ODAC task, and expects total_charge=0 and spin multiplicity=0 as model inputs. The ODAC23 dataset only contains CO2/H2O water absorption, so anything more than might be inaccurate (e.g. hydrocarbons in MOFs). Further, there is a limited number of bare-MOF structures in the training data, so you should be careful if you are using a new MOF structure. |
| **oc25** | [OC25](https://arxiv.org/abs/2509.17862) | RPBE+D3 as implemented in VASP, with VASP6.4 pseudopotentials, and dipole corrections. | Renewable energy, (electro)catalysis, fuel cells, energy conversion, sustainable fertilizer production, chemical refining, plastics synthesis/upcycling | UMA has not seen varying charge or spin multiplicity for the OC25 task, and expects total_charge=0 and spin multiplicity=0 as model inputs. The model works surprisingly well for charged systems despite not being explicitly provided that information, but one should be careful. Work functions are not provided by UMA, subsequent DFT calculations are required to extract such information, if desired. Only available in UMA-1.2 |
 | **oc22** | [OC22](https://arxiv.org/abs/2206.08917) | PBE+U as implemented in VASP with VASP5.4 pseudopotentials and spin-polarization | Oxide catalysis or supports, energy conversion and storage | UMA has not seen varying charge or spin multiplicity for the OC22 task, and expects total_charge=0 and spin multiplicity=0 as model inputs. No explicit solvents are included in OC22. Only available in UMA-1.2 |

:::{note}
*OC20 was updated from the original OC20 and recomputed to produce total energies instead of adsorption energies.
:::

(uma-s-1p2p1-fine-tunes)=
## UMA-S-1.2.1 fine-tunes: D4rev and r²SCAN

`uma-s-1p2p1-ft-d4rev` and `uma-s-1p2p1-ft-r2scan` are two fine-tuned variants of UMA-S-1.2.1 at higher levels of theory, both served by UMA Nitro. Each has a single task with a different level of theory than the corresponding task of `uma-s-1p2p1`.

| Model name | Task | Level of theory | Training data | Use it for |
|------|------|------|------|------|
| `uma-s-1p2p1-ft-d4rev` | `omol` | ωB97M-D4rev, dispersion included | OMol25-4M | molecules, noncovalent interactions, molecular liquids and crystals |
| `uma-s-1p2p1-ft-r2scan` | `omat` | r²SCAN, not spin-polarized, no dispersion | [MAD-1.6](https://arxiv.org/abs/2603.02089) | most inorganic and mixed chemical systems, add dispersion with `DFTD3Calculator(..., functional="r2scan")` |

`uma-s-1p2p1-ft-d4rev` was fine-tuned on approximately 4 million structures from the OMol25 dataset, whose labels were recomputed at the ωB97M-D4rev level of theory.

`uma-s-1p2p1-ft-r2scan` was trained on [MAD-1.6](https://arxiv.org/abs/2603.02089), an r²SCAN dataset from the Laboratory of Computational Science and Modelling (COSMO), EPFL. We thank the MAD authors for making it available.

Do not add dispersion to `uma-s-1p2p1-ft-d4rev`. We recommend using `uma-s-1p2p1-ft-r2scan` with `DFTD3Calculator(..., functional="r2scan")`.

```python
from fairchem.core import DFTD3Calculator, FAIRChemCalculator, pretrained_mlip

# molecules: wB97M-D4rev (dispersion is already included, do not add D3)
predictor = pretrained_mlip.get_predict_unit("uma-s-1p2p1-ft-d4rev", device="cuda")
calc = FAIRChemCalculator(predictor, task_name="omol")

# materials: r2SCAN + D3(BJ)
predictor = pretrained_mlip.get_predict_unit("uma-s-1p2p1-ft-r2scan", device="cuda")
calc = DFTD3Calculator(
    FAIRChemCalculator(predictor, task_name="omat"), functional="r2scan", device="cuda"
)
```

:::{warning}
- `uma-s-1p2p1-ft-r2scan` **is not spin-polarized** and describes every system as non-magnetic. Do not use it for magnetic systems or for cohesive energies, as isolated atoms are also often strongly magnetic; use `uma-s-1p2p1` with the `omat` task for those instead.
- Each model has only one task (`omat` or `omol`). Charge and spin multiplicity work in `uma-s-1p2p1-ft-d4rev` as in `uma-s-1p2p1`'s `omol` task.
- Both models are fine-tuned on less data than the original UMA-S-1.2.1 model. Often the latter is more accurate at predicting DFT labels at its corresponding levels of theory.
- No isolated-atom reference energies are shipped; single-atom energies from the model might be unreliable.
:::
