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

GNNs for Chemistry
----------

The most recent, state of the art machine learned potentials in atomistic simulations are based on graph models that are trained on large (1M+) datasets. These models can be downloaded and used in a wide array of applications ranging from catalysis to materials properties. These pre-trained models can be used on their own, to accelerate DFT calculation, and they can also be used as a starting point to fine-tune new models for specific tasks. 

# Background on DFT and machine learning potentials

Density functional theory (DFT) has been a mainstay in molecular simulation, but its high computational cost limits the number and size of simulations that are practical. Over the past two decades machine learning has increasingly been used to build surrogate models to supplement DFT. We call these models machine learned potentials (MLP) In the early days, neural networks were trained using the cartesian coordinates of atomistic systems as features with some success. These features lack important physical properties, notably they lack invariance to rotations, translations and permutations, and they are extensive features, which limit them to the specific system being investigated. About 15 years ago, a new set of features called symmetry functions were developed that were intensive, and which had these invariances. These functions enabled substantial progress in MLP, but they had a few important limitations. First, the size of the feature vector scaled quadratically with the number of elements, practically limiting the MLP to 4-5 elements. Second, composition was usually implicit in the functions, which limited the transferrability of the MLP to new systems. Finally, these functions were "hand-crafted", with limited or no adaptability to the systems being explored, thus one needed to use judgement and experience to select them. While progess has been made in mitigating these limitations, a new approach has overtaken these methods.

Today, the state of the art in machine learned potentials uses graph convolutions to generate the feature vectors. In this approach, atomistic systems are represented as graphs where each node is an atom, and the edges connect the nodes (atoms) and roughly represent interactions or bonds between atoms. Then, there are machine learnable convolution functions that operate on the graph to generate feature vectors. These operators can work on pairs, triplets and quadruplets of nodes to compute "messages" that are passed to the central node (atom) and accumulated into the feature vector. This feature generate method can be constructed with all the desired invariances, the functions are machine learnable, and adapt to the systems being studied, and it scales well to high numbers of elements (the current models handle 50+ elements). These kind of MLPs began appearing regularly in the literature around 2016.

Today an MLP consists of three things:

1. A model that takes an atomistic system, generates features and relates those features to some output.
2. A dataset that provides the atomistic systems and the desired output labels. This label could be energy, forces, or other atomistic properties.
3. A checkpoint that stores the trained model for use in predictions.
