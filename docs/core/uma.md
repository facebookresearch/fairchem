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

# UMA models overview

[UMA](https://ai.meta.com/research/publications/uma-a-family-of-universal-models-for-atoms/) is an equivariant GNN that leverages a novel technique called Mixture of Linear Experts (MoLE) to give it the capacity to learn the largest multi-modal dataset to date (500M DFT examples), while preserving energy conservation and inference speed. Even a 6M active parameter (145M total) UMA model is able to acheieve SOTA accuracy on a wide range of domains such as materials, molecules and catalysis. 


## The UMA Mixture-of-Linear-Experts routing function

The UMA model uses a Mixture-of-Linear-Expert (MoLE) architecture to achieve very high parameter count with fast inference speeds with a single output head. In order to route the model to the correct set parameters, the model must be given a set of inputs.  The following information are required for the input to the model.

* task, ie: omol, oc20, omat, odac, omc (this affects the level of theory and DFT calculations that the model is trying to emulate) see below
* charge - total known charge of the system (only used for omol task and defaults to 0)
* spin - total spin multiplicity of the system (only used for omol task and defaults to 1)
* elemental composition - The unordered total elemental composition of the system. Each element has an atom embedding and the composition embedding is the mean over all the atom embeddings. For example H2O2 would be assigned the same embedding regardless of its conformer configuration.

![UMA model architecture](https://github.com/facebookresearch/fairchem/blob/main/docs/core/uma.svg "UMA model architecture")

### The UMA task

UMA is trained on 5 different DFT datasets with different levels of theory. An UMA **task** refers to a specific level of theory associated with that DFT dataset. UMA learns an embedding for the given **task**. Thus at inference time, the user must specify which one of the 5 embeddings they want to use to produce an output with the DFT level of theory they want. See the following table for more details.

| Task    | Dataset | DFT Level of Theory | Relevant applications | Usage Notes |
| ------- | ------- | ----- | ------ | ----- |
| omol    | [Omol25](https://arxiv.org/abs/2505.08762) | wB97M-V/def2-TZVPD as implemented in ORCA6, including non-local dispersion. All solvation should be explicit.   |  Biology, organic chemistry, protein folding, small-molecule pharmaceuticals, organic liquid properties, homogeneous catalysis | total charge and spin multiplicity. If you don't know what these are, you should be very careful if modeling charged or open-shell systems. This can be used to study radical chemistry or understand the impact of magnetic states on the structure of a molecule. All training data is aperiodic, so any periodic systems should be treated with some caution. Probably won't work well for inorganic materials.  |
| omc     | Omc25 | PBE+D3 as implemented in VASP. | Pharmaceutical packaging, bio-inspired materials, organic electronics, organic LEDs | UMA has not seen varying charge or spin multiplicity for the OMC task, and expects total_charge=0 and spin multiplicity=0 as model inputs. |
| omat    | [Omat24](https://arxiv.org/abs/2410.12771) | PBE/PBE+U as implemented in VASP using Materials Project suggested settings, except with VASP 54 pseudopotentials. No dispersion.   | Inorganic materials discovery, solar photovoltaics, advanced alloys, superconductors, electronic materials, optical materials | UMA has not seen varying charge or spin multiplicity for the OMat task, and expects total_charge=0 and spin multiplicity=0 as model inputs. Spin polarization effects are included, but you can't select the magnetic state. Further, OMat24 did not fully sample possible spin states in the training data. |
| oc20    | [OC20*](https://arxiv.org/abs/2010.09990) | RPBE as implemented in VASP, with VASP5.4 pseudopotentials. No dispersion. | Renewable energy, catalysis, fuel cells, energy conversion, sustainable fertilizer production, chemical refining, plastics synthesis/upcycling | UMA has not seen varying charge or spin multiplicity for the OC20 task, and expects total_charge=0 and spin multiplicity=0 as model inputs. No oxides or explicit solvents are included in OC20. The model works surprisingly well for transition state searches given the nature of the training data, but you should be careful. RPBE works well for small molecules, but dispersion will be important for larger molecules on surfaces. |
| odac    | [ODac23](https://arxiv.org/abs/2311.00341) | PBE+D3 as implemented in VASP, with VASP5.4 pseudopotentials. | Direct air capture, carbon capture and storage, CO2 conversion, catalysis | UMA has not seen varying charge or spin multiplicity for the ODAC task, and expects total_charge=0 and spin multiplicity=0 as model inputs. The ODAC23 dataset only contains CO2/H2O water absorption, so anything more than might be inaccurate (e.g. hydrocarbons in MOFs). Further, there is a limited number of bare-MOF structures in the training data, so you should be careful if you are using a new MOF structure. |

*Note: OC20 is was updated from the original OC20 and recomputed to produce total energies instead of adsorption energies.

## Inference 

Inference is done using [MLIPPredictUnit](https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/core/units/mlip_unit/mlip_unit.py#L867). The [FairchemCalculator](https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/core/calculate/ase_calculator.py#L3) (an ASE calculator) is simply a convenience wrapper around the MLIPPredictUnit.

For simple cases such as doing demos or education, the ASE calculator is very easy to use but for any more complex cases such as running MD, batched inference etc, we do not recommend using the calculator interface.

```python
from fairchem.core import pretrained_mlip, FAIRChemCalculator

predictor = pretrained_mlip.get_predict_unit("uma-s-1", device="cuda")
calc = FAIRChemCalculator(predictor, task_name="oc20")
```

### Batch Inference

Coming soon!

### Inference settings

#### Default mode

UMA is designed for both general purpose usage (single or batched systems) and single-system long rollout (MD simulations, Relaxations etc). For general purpose, we suggest using the [default settings](https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/core/units/mlip_unit/api/inference.py#L92). This is a good trade-off between accuracy, speed and memory consumption and should suffice for most applications. In this setting, on a single 80GB H100 GPU, we expect a user should be able to compute on systems as large 50-100k neighbors (depending on their atomic density). Batching is also supported in this mode.

#### Turbo mode

For long rollout trajectory use-cases where UMA can excel at such as MD or relaxations. We provide a special mode called **turbo** which optimizes for speed but restricts the user to use a single system where the atomic composition is held constant. Turbo mode is approximately 1.5-2x faster than default mode depending on the situation. Batching is not supported in this mode. It can be easily activated as shown below.

```python
predictor = pretrained_mlip.get_predict_unit("uma-s-1", device="cuda", inference_settings="turbo")
```

#### Custom modes for advanced users

The advanced user might quickly see that **default** mode and **turbo** mode are special cases of our [inference settings api](https://github.com/facebookresearch/fairchem/blob/main/src/fairchem/core/units/mlip_unit/api/inference.py#L47). You can customize it for your application if you understand what you are doing. The following table provides more information.

| Setting Flag  | Description |
| ----- | ----- |
| tf32 | enables torch [tf32](https://docs.pytorch.org/docs/stable/notes/cuda.html) format for matrix multiplication. This will speed up inference at a slight trade-off for precision. In our tests, this will make minimal difference to most applications. It is able to preserve equivariance, energy conservation for long rollouts. However, if you are computing higher order derivatives such as Hessians, we recommend turning this off |
| activation_checkpointing | this uses a custom chunked activation checkpointing algorithm and allows significant savings in memory for a small inference speed boost. If you are predicting on systems >1000 atoms, we recommend leaving this on. However, if you want the absolute fastest inference possible for small systems, you can turn this off |
| merge_mole | This is useful in long rollout applications where the system composition stays constant we pre-merge the MoLE weights, saving both memory and reducing some computation cost |
| compile | This uses torch.compile to significantly speed up computation. Due to the way pytorch traces the internal graph, it requires a long compile time during the first iteration and can even recompile anytime it detected a significant change in input dimensions. It is not recommended if you are computing frequently on very different atomic systems. |
| wigner_cuda | This is a special mode that turns on cuda graphs for the internal Wigner matrix calculations and will lead to significant speed ups for smaller systems |
| external_graph_gen | Only use this if you want to use an external graph generator. This should be rarely used except for development |

For example, if I have a specific application that requires the fastest speed

```python
from fairchem.core.units.mlip_unit.api.inference import InferenceSettings

settings = InferenceSettings(
    tf32=True,
    activation_checkpointing=True,
    merge_mole=False,
    compile=False,
    wigner_cuda=False,
    external_graph_gen=False,
    internal_graph_gen_version=2,
)

predictor = pretrained_mlip.get_predict_unit("uma-sm", device="cuda", inference_settings=settings)
```

## Finetuning UMA
Coming soon!
