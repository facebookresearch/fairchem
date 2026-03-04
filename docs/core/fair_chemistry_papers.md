# FAIR Chemistry Papers

Research publications from the FAIR Chemistry team at Meta, advancing machine learning for atomic simulations across molecules, materials, and catalysts.

:::{tip}
Click on any paper card to expand and see the full abstract and author list.
:::

## Universal Models & Architectures

State-of-the-art neural network architectures for atomic property prediction.

::::{grid} 1 1 2 2

:::{card} UMA: A Family of Universal Models for Atoms
:link: https://arxiv.org/abs/2506.23971

**2025** — Trained on 500M+ structures with 1.4B parameters but only ~50M active per structure using mixture of linear experts
:::

:::{card} eSEN: Learning Smooth and Expressive Interatomic Potentials
:link: https://arxiv.org/abs/2502.12147

**2025** — Passes energy conservation tests; SOTA on thermal conductivity, phonons, and materials stability prediction
:::

:::{card} EquiformerV2: Improved Equivariant Transformer
:link: https://arxiv.org/abs/2306.12059

**2023** — Up to 9% improvement on forces, 4% on energies; 2x reduction in DFT calculations for adsorption energies
:::

:::{card} eSCN: Reducing SO(3) Convolutions to SO(2)
:link: https://arxiv.org/abs/2302.03655

**2023** — Reduces equivariant convolution complexity from O(L⁶) to O(L³); SOTA on OC-20 and OC-22
:::

::::

```{admonition} More Architecture Papers
:class: dropdown

### GemNet-OC: Graph Neural Networks for Large Datasets
**arXiv:** [2204.02782](https://arxiv.org/abs/2204.02782) (2022)

**Authors:** Johannes Gasteiger, Muhammed Shuaibi, Anuroop Sriram, Stephan Günnemann, Zachary Ulissi, C. Lawrence Zitnick, Abhishek Das

**Abstract:** Recent years have seen the advent of molecular simulation datasets that are orders of magnitude larger and more diverse. These new datasets differ substantially in four aspects of complexity: 1. Chemical diversity, 2. system size, 3. dataset size, and 4. domain shift. We develop GemNet-OC which outperforms the previous state-of-the-art on OC20 by 16% while reducing training time by a factor of 10.

---

### Spherical Channels for Modeling Atomic Interactions
**arXiv:** [2206.14331](https://arxiv.org/abs/2206.14331) (2022)

**Authors:** C. Lawrence Zitnick, Abhishek Das, Adeesh Kolluru, Janice Lan, Muhammed Shuaibi, Anuroop Sriram, Zachary Ulissi, Brandon Wood

**Abstract:** We propose the Spherical Channel Network (SCN) to model atomic energies and forces. The SCN is a graph neural network where atom embeddings are spherical functions represented using spherical harmonics. We demonstrate state-of-the-art results on the large-scale Open Catalyst dataset.

---

### ForceNet: A Graph Neural Network for Large-Scale Quantum Calculations
**arXiv:** [2103.01436](https://arxiv.org/abs/2103.01436) (2021)

**Authors:** Weihua Hu, Muhammed Shuaibi, Abhishek Das, Siddharth Goyal, Anuroop Sriram, Jure Leskovec, Devi Parikh, C. Lawrence Zitnick

**Abstract:** By not imposing explicit physical constraints, we can flexibly design expressive models while maintaining computational efficiency. Physical constraints are implicitly imposed through physics-based data augmentation. ForceNet predicts atomic forces more accurately than state-of-the-art physics-based GNNs while being faster.

---

### Rotation Invariant Graph Neural Networks using Spin Convolutions
**arXiv:** [2106.09575](https://arxiv.org/abs/2106.09575) (2021)

**Authors:** Muhammed Shuaibi, Adeesh Kolluru, Abhishek Das, Aditya Grover, Anuroop Sriram, Zachary Ulissi, C. Lawrence Zitnick

**Abstract:** We introduce a novel approach to modeling angular information between atoms using per-edge local coordinate frames and spin convolutions. State-of-the-art results are demonstrated on the large-scale Open Catalyst 2020 dataset.

---

### Towards Training Billion Parameter Graph Neural Networks
**arXiv:** [2203.09697](https://arxiv.org/abs/2203.09697) (2022)

**Authors:** Anuroop Sriram, Abhishek Das, Brandon M. Wood, Siddharth Goyal, C. Lawrence Zitnick

**Abstract:** We introduce Graph Parallelism, a method to distribute input graphs across multiple GPUs, enabling training of very large GNNs with billions of parameters. On OC20, graph-parallelized models achieve 15% relative improvement on force MAE.
```

---

## Datasets

Large-scale open datasets powering the next generation of ML models for chemistry.

::::{grid} 1 1 2 2

:::{card} OMol25: Open Molecules 2025
:link: https://arxiv.org/abs/2505.08762

**2025** — 100M+ DFT calculations, 83 elements, molecules up to 350 atoms
:::

:::{card} OMat24: Open Materials 2024
:link: https://arxiv.org/abs/2410.12771

**2024** — 110M+ DFT calculations for inorganic materials
:::

:::{card} OMC25: Open Molecular Crystals 2025
:link: https://arxiv.org/abs/2508.02651

**2025** — 27M+ molecular crystal structures with DFT labels
:::

:::{card} ODAC25: Open DAC 2025
:link: https://arxiv.org/abs/2508.03162

**2025** — 60M DFT calculations for MOF sorbent discovery
:::

::::

```{admonition} All Dataset Papers
:class: dropdown

### The Open Polymers 2026 (OPoly26) Dataset
**arXiv:** [2512.23117](https://arxiv.org/abs/2512.23117) (2025)

**Authors:** Daniel S. Levine, Nicholas Liesen, Lauren Chua, James Diffenderfer, et al.

**Abstract:** We create the Open Polymers 2026 (OPoly26) dataset containing more than 6.57 million DFT calculations on polymer clusters comprising over 1.2 billion total atoms. OPoly26 captures chemical diversity including variations in monomer composition, degree of polymerization, chain architectures, and solvation environments.

---

### The Open Catalyst 2025 (OC25) Dataset for Solid-Liquid Interfaces
**arXiv:** [2509.17862](https://arxiv.org/abs/2509.17862) (2025)

**Authors:** Sushree Jagriti Sahoo, Mikael Maraschin, Daniel S. Levine, Zachary Ulissi, et al.

**Abstract:** We introduce OC25, consisting of 7,801,261 calculations across 1,511,270 unique explicit solvent environments. OC25 is the largest solid-liquid interface dataset available, spanning 88 elements with commonly used solvents/ions.

---

### The Open Molecules 2025 (OMol25) Dataset
**arXiv:** [2505.08762](https://arxiv.org/abs/2505.08762) (2025)

**Authors:** Daniel S. Levine, Muhammed Shuaibi, Evan Walter Clark Spotte-Smith, et al.

**Abstract:** OMol25 is composed of more than 100 million DFT calculations at the ωB97M-V/def2-TZVPD level. It uniquely blends 83 elements, diverse intra- and intermolecular interactions, explicit solvation, variable charge/spin, conformers, and reactive structures across ~83M unique molecular systems.

---

### Open Molecular Crystals 2025 (OMC25) Dataset
**arXiv:** [2508.02651](https://arxiv.org/abs/2508.02651) (2025)

**Authors:** Vahe Gharakhanyan, Luis Barroso-Luque, Yi Yang, Muhammed Shuaibi, et al.

**Abstract:** OMC25 contains over 27 million molecular crystal structures with 12 elements and up to 300 atoms per unit cell, generated from dispersion-inclusive DFT relaxation trajectories of over 230,000 structures.

---

### The Open DAC 2025 (ODAC25) Dataset
**arXiv:** [2508.03162](https://arxiv.org/abs/2508.03162) (2025)

**Authors:** Anuroop Sriram, Logan M. Brabson, Xiaohan Yu, Sihoon Choi, et al.

**Abstract:** ODAC25 comprises nearly 60 million DFT calculations for CO₂, H₂O, N₂, and O₂ adsorption in 15,000 MOFs. It introduces chemical diversity through functionalized MOFs, GCMC-derived placements, and synthetically generated frameworks.

---

### Open Materials 2024 (OMat24) Dataset
**arXiv:** [2410.12771](https://arxiv.org/abs/2410.12771) (2024)

**Authors:** Luis Barroso-Luque, Muhammed Shuaibi, Xiang Fu, Brandon M. Wood, et al.

**Abstract:** OMat24 contains over 110 million DFT calculations focused on structural and compositional diversity. EquiformerV2 models achieve state-of-the-art on Matbench Discovery with F1 score above 0.9.

---

### Open Catalyst Experiments 2024 (OCx24)
**arXiv:** [2411.11783](https://arxiv.org/abs/2411.11783) (2024)

**Authors:** Jehad Abed, Jiheon Kim, Muhammed Shuaibi, Brook Wander, et al.

**Abstract:** OCX24 bridges experiments and computational models with 572 synthesized samples and 441 gas diffusion electrodes evaluated for CO₂RR and HER. DFT-verified adsorption energies were calculated on ~20,000 materials requiring 685 million AI-accelerated relaxations.

---

### The Open Catalyst 2022 (OC22) Dataset
**arXiv:** [2206.08917](https://arxiv.org/abs/2206.08917) (2022)

**Authors:** Richard Tran, Janice Lan, Muhammed Shuaibi, Brandon M. Wood, et al.

**Abstract:** OC22 consists of 62,331 DFT relaxations (~9,854,504 single point calculations) across oxide materials, coverages, and adsorbates, critical for OER catalyst development.

---

### The Open DAC 2023 (ODAC23) Dataset
**arXiv:** [2311.00341](https://arxiv.org/abs/2311.00341) (2023)

**Authors:** Anuroop Sriram, Sihoon Choi, Xiaohan Yu, Logan M. Brabson, et al.

**Abstract:** ODAC23 consists of more than 38M DFT calculations on more than 8,400 MOF materials containing adsorbed CO₂ and/or H₂O for direct air capture applications.

---

### The Open Catalyst 2020 (OC20) Dataset
**arXiv:** [2010.09990](https://arxiv.org/abs/2010.09990) (2020)

**Authors:** Lowik Chanussot, Abhishek Das, Siddharth Goyal, Thibaut Lavril, et al.

**Abstract:** OC20 consists of 1,281,040 DFT relaxations (~264,890,000 single point evaluations) across materials, surfaces, and adsorbates. The foundational dataset for the Open Catalyst Project.
```

---

## Generative Models

Generating novel molecules, materials, and crystal structures.

::::{grid} 1 1 2 2

:::{card} ADiT: All-atom Diffusion Transformers
:link: https://arxiv.org/abs/2503.03965

**2025** — Unified generative model for molecules and materials
:::

:::{card} FlowMM: Riemannian Flow Matching for Materials
:link: https://arxiv.org/abs/2406.04713

**2024** — 3x more efficient at finding stable materials
:::

:::{card} FlowLLM: LLMs + Flow Matching for Crystals
:link: https://arxiv.org/abs/2410.23405

**2024** — 3x higher generation rate of stable materials
:::

:::{card} Space Group Conditional Flow Matching
:link: https://arxiv.org/abs/2509.23822

**2025** — Symmetry-aware crystal generation
:::

::::

```{admonition} More Generative Model Papers
:class: dropdown

### All-atom Diffusion Transformers (ADiT)
**arXiv:** [2503.03965](https://arxiv.org/abs/2503.03965) (2025)

**Authors:** Chaitanya K. Joshi, Xiang Fu, Yi-Lun Liao, Vahe Gharakhanyan, Benjamin Kurt Miller, Anuroop Sriram, Zachary W. Ulissi

**Abstract:** We introduce ADiT, a unified latent diffusion framework for jointly generating both periodic materials and non-periodic molecular systems. An autoencoder maps molecules and materials to a shared latent space, then diffusion generates new embeddings. ADiT achieves state-of-the-art results on MP20, QM9, and GEOM-DRUGS with significant speedups over equivariant diffusion models.

---

### Space Group Conditional Flow Matching
**arXiv:** [2509.23822](https://arxiv.org/abs/2509.23822) (2025)

**Authors:** Omri Puny, Yaron Lipman, Benjamin Kurt Miller

**Abstract:** We introduce a generative framework that samples highly-symmetric, stable crystals by conditioning on space groups and Wyckoff positions. Using efficient group averaging, we achieve state-of-the-art on crystal structure prediction and de novo generation benchmarks.

---

### FlowLLM: LLMs and Riemannian Flow Matching for Materials
**arXiv:** [2410.23405](https://arxiv.org/abs/2410.23405) (2024)

**Authors:** Anuroop Sriram, Benjamin Kurt Miller, Ricky T. Q. Chen, Brandon M. Wood

**Abstract:** FlowLLM combines fine-tuned LLMs with Riemannian flow matching. It outperforms state-of-the-art by 3x on stable material generation rate and ~50% on stable, unique, and novel crystals.

---

### FlowMM: Riemannian Flow Matching for Materials
**arXiv:** [2406.04713](https://arxiv.org/abs/2406.04713) (2024)

**Authors:** Benjamin Kurt Miller, Ricky T. Q. Chen, Anuroop Sriram, Brandon M Wood

**Abstract:** We generalize Riemannian Flow Matching to crystals with translation, rotation, permutation, and periodic boundary conditions. FlowMM is ~3x more efficient at finding stable materials compared to previous methods.

---

### Fine-Tuned Language Models Generate Stable Inorganic Materials
**arXiv:** [2402.04379](https://arxiv.org/abs/2402.04379) (2024)

**Authors:** Nate Gruver, Anuroop Sriram, Andrea Madotto, Andrew Gordon Wilson, C. Lawrence Zitnick, Zachary Ulissi

**Abstract:** Fine-tuned LLaMA-2 70B generates materials predicted to be metastable at about twice the rate (49% vs 28%) of CDVAE. Around 90% of sampled structures obey physical constraints, demonstrating LLMs' surprising suitability for atomistic data.

---

### FastCSP: Accelerated Molecular Crystal Structure Prediction
**arXiv:** [2508.02641](https://arxiv.org/abs/2508.02641) (2025)

**Authors:** Vahe Gharakhanyan, Yi Yang, Luis Barroso-Luque, Muhammed Shuaibi, et al.

**Abstract:** FastCSP combines random structure generation with UMA-powered relaxation and free energy calculations. Results for a single system can be obtained within hours on tens of GPUs, making high-throughput crystal structure prediction feasible.
```

---

## Sampling & Molecular Dynamics

Advanced methods for sampling molecular configurations and running simulations.

::::{grid} 1 1 2 2

:::{card} Adjoint Sampling: Highly Scalable Diffusion Samplers
:link: https://arxiv.org/abs/2504.11713

**2025** — First on-policy approach that scales to large problems
:::

:::{card} Adjoint Schrödinger Bridge Sampler
:link: https://arxiv.org/abs/2506.22565

**2025** — Kinetic-optimal transportation for molecular sampling
:::

:::{card} Enhancing Diffusion Sampling with Collective Variables
:link: https://arxiv.org/abs/2510.11923

**2025** — First reactive sampling with diffusion-based samplers
:::

::::

```{admonition} Sampling Paper Details
:class: dropdown

### Adjoint Sampling
**arXiv:** [2504.11713](https://arxiv.org/abs/2504.11713) (2025)

**Authors:** Aaron Havens, Benjamin Kurt Miller, Bing Yan, Carles Domingo-Enrich, Anuroop Sriram, Brandon Wood, Daniel Levine, et al.

**Abstract:** We introduce Adjoint Sampling, a highly scalable algorithm for learning diffusion processes that sample from unnormalized densities. It's the first on-policy approach allowing significantly more gradient updates than energy evaluations. We demonstrate amortized conformer generation across many molecular systems.

---

### Adjoint Schrödinger Bridge Sampler (ASBS)
**arXiv:** [2506.22565](https://arxiv.org/abs/2506.22565) (2025)

**Authors:** Guan-Horng Liu, Jaemoo Choi, Yongxin Chen, Benjamin Kurt Miller, Ricky T. Q. Chen

**Abstract:** ASBS employs simple matching-based objectives without needing target samples during training. Grounded in the Schrödinger Bridge for kinetic-optimal transportation, ASBS generalizes to arbitrary source distributions for sampling from molecular Boltzmann distributions.

---

### Enhancing Diffusion-Based Sampling with Molecular Collective Variables
**arXiv:** [2510.11923](https://arxiv.org/abs/2510.11923) (2025)

**Authors:** Juno Nam, Bálint Máté, Artur P. Toshev, Manasa Kaniselvan, Rafael Gómez-Bombarelli, et al.

**Abstract:** We introduce a sequential bias along collective variables to improve mode discovery and enable free energy estimation. We are the first to demonstrate reactive sampling using a diffusion-based sampler, capturing bond breaking and formation with universal interatomic potentials.
```

---

## Applications & Discovery

Practical applications accelerating catalyst and materials discovery.

::::{grid} 1 1 2 2

:::{card} CatTSunami: Accelerating Transition State Calculations
:link: https://arxiv.org/abs/2405.02078

**2024** — 1500x speedup for reaction network enumeration
:::

:::{card} AdsorbML: Efficient Adsorption Energy Calculations
:link: https://arxiv.org/abs/2211.16486

**2022** — 2000x speedup with 87% accuracy
:::

:::{card} Catlas: Automated Catalyst Discovery Framework
:link: https://arxiv.org/abs/2208.12717

**2022** — High-throughput screening for syngas conversion
:::

:::{card} GA-Accelerated Liquid Crystal Polymer Discovery
:link: https://arxiv.org/abs/2505.13477

**2025** — Genetic algorithms + first-principles for VR/AR materials
:::

::::

```{admonition} Application Paper Details
:class: dropdown

### CatTSunami: Accelerating Transition State Energy Calculations
**arXiv:** [2405.02078](https://arxiv.org/abs/2405.02078) (2024)

**Authors:** Brook Wander, Muhammed Shuaibi, John R. Kitchin, Zachary W. Ulissi, C. Lawrence Zitnick

**Abstract:** We show GNN potentials trained on OC20 find transition states within 0.1 eV of DFT 91% of the time with 28x speedup. We replicated a reaction network with 61 intermediates and 174 reactions at DFT resolution using just 12 GPU days (vs 52 GPU years), a 1500x speedup.

---

### AdsorbML: A Leap in Efficiency for Adsorption Energy Calculations
**arXiv:** [2211.16486](https://arxiv.org/abs/2211.16486) (2022)

**Authors:** Janice Lan, Aini Palizhati, Muhammed Shuaibi, Brandon M. Wood, et al.

**Abstract:** We demonstrate ML potentials can identify low energy adsorbate-surface configurations with 87.36% accuracy while achieving a 2000x speedup. We introduce the Open Catalyst Dense dataset with ~1,000 surfaces and 100,000 configurations.

---

### Catlas: Automated Framework for Catalyst Discovery
**arXiv:** [2208.12717](https://arxiv.org/abs/2208.12717) (2022)

**Authors:** Brook Wander, Kirby Broderick, Zachary W. Ulissi

**Abstract:** Catlas explores large design spaces using pre-trained ML models without upfront training. For syngas-to-oxygenates conversion, we explored 947 binary intermetallics, identifying 144 candidate materials including Pt-Ti, Pd-V, Ni-Nb, and Ti-Zn.

---

### Genetic Algorithm-Accelerated Discovery of Liquid Crystal Polymers
**arXiv:** [2505.13477](https://arxiv.org/abs/2505.13477) (2025)

**Authors:** Jianing Zhou, Yuge Huang, Arman Boromand, Keian Noori, et al.

**Abstract:** We integrate first-principles calculations with genetic algorithms to discover liquid crystal polymers with low visible absorption and high refractive index for VR/AR/MR technologies.

---

### Adapting OC20-trained EquiformerV2 for High-Entropy Materials
**arXiv:** [2403.09811](https://arxiv.org/abs/2403.09811) (2024)

**Authors:** Christian M. Clausen, Jan Rossmeisl, Zachary W. Ulissi

**Abstract:** We show that through energy filtering and few-shot fine-tuning, EquiformerV2 achieves state-of-the-art accuracy on high-entropy alloys (Ag-Ir-Pd-Pt-Ru), demonstrating that foundational knowledge from ordered structures extrapolates to disordered solid-solutions.
```

---

## Training Methods & Techniques

Innovations in model training, active learning, and pre-training strategies.

::::{grid} 1 1 2 2

:::{card} DeNS: Generalizing Denoising to Non-Equilibrium Structures
:link: https://arxiv.org/abs/2403.09549

**2024** — New SOTA on OC20 and OC22
:::

:::{card} JMP: Joint Multi-domain Pre-training
:link: https://arxiv.org/abs/2310.16802

**2023** — 59% average improvement over training from scratch
:::

:::{card} FINETUNA: Fine-tuning Accelerated Simulations
:link: https://arxiv.org/abs/2205.01223

**2022** — 91% reduction in DFT calculations
:::

::::

```{admonition} Training Methods Paper Details
:class: dropdown

### Generalizing Denoising to Non-Equilibrium Structures (DeNS)
**arXiv:** [2403.09549](https://arxiv.org/abs/2403.09549) (2024)

**Authors:** Yi-Lun Liao, Tess Smidt, Muhammed Shuaibi, Abhishek Das

**Abstract:** We propose denoising non-equilibrium structures as an auxiliary training task. By encoding forces to specify which structure we're denoising, DeNS achieves new state-of-the-art on OC20 and OC22 while significantly improving training efficiency on MD17.

---

### From Molecules to Materials: Joint Multi-domain Pre-training (JMP)
**arXiv:** [2310.16802](https://arxiv.org/abs/2310.16802) (2023)

**Authors:** Nima Shoghi, Adeesh Kolluru, John R. Kitchin, Zachary W. Ulissi, C. Lawrence Zitnick, Brandon M. Wood

**Abstract:** JMP simultaneously trains on multiple datasets (~120M systems from OC20, OC22, ANI-1x, Transition-1x) within a multi-task framework. It demonstrates 59% average improvement over training from scratch, matching or setting SOTA on 34 of 40 tasks.

---

### FINETUNA: Fine-tuning Accelerated Molecular Simulations
**arXiv:** [2205.01223](https://arxiv.org/abs/2205.01223) (2022)

**Authors:** Joseph Musielewicz, Xiaoxiao Wang, Tian Tian, Zachary Ulissi

**Abstract:** An online active learning framework that incorporates prior information from pre-trained models. It accelerates simulations by reducing DFT calculations by 91% while meeting accuracy thresholds 93% of the time.

---

### Robust and Scalable Uncertainty Estimation with Conformal Prediction
**arXiv:** [2208.08337](https://arxiv.org/abs/2208.08337) (2022)

**Authors:** Yuge Hu, Joseph Musielewicz, Zachary Ulissi, Andrew J. Medford

**Abstract:** We combine conformal prediction with latent space distances to estimate uncertainty of neural network force fields. The method is calibrated, sharp, and scalable to training datasets of 1 million images.

---

### Generalization of Graph-Based Active Learning Relaxation Strategies
**arXiv:** [2311.01987](https://arxiv.org/abs/2311.01987) (2023)

**Authors:** Xiaoxiao Wang, Joseph Musielewicz, Richard Tran, et al.

**Abstract:** We investigate Finetuna on out-of-domain systems: larger adsorbates, metal-oxides with spin polarization, and 3D structures like zeolites and MOFs. The framework reduces DFT calculations by 80% for alcohols/3D structures and 42% for oxides.

---

### Enabling Robust Offline Active Learning for ML Potentials
**arXiv:** [2008.10773](https://arxiv.org/abs/2008.10773) (2020)

**Authors:** Muhammed Shuaibi, Saurabh Sivakumar, Rui Qi Chen, Zachary W. Ulissi

**Abstract:** We demonstrate a Δ-machine learning approach enabling stable convergence in offline active learning by avoiding unphysical configurations, reducing first-principles calculations by 70-90%.
```

---

## Electronic Structure & Properties

Predicting electronic structure and chemical properties beyond energies and forces.

::::{grid} 1 1 2 2

:::{card} HELM: Hamiltonian-trained Electronic-structure Learning
:link: https://arxiv.org/abs/2510.00224

**2025** — Hamiltonian prediction for 100+ atom structures
:::

:::{card} Chemical Properties from GNN-Predicted Electron Densities
:link: https://arxiv.org/abs/2309.04811

**2023** — Atomic charges and dipole moments from electron density
:::

:::{card} XRD Loss Landscape Analysis
:link: https://arxiv.org/abs/2512.04036

**2025** — Insights for crystal structure from powder diffraction
:::

::::

```{admonition} Electronic Structure Paper Details
:class: dropdown

### HELM: Learning from Electronic Structure Across the Periodic Table
**arXiv:** [2510.00224](https://arxiv.org/abs/2510.00224) (2025)

**Authors:** Manasa Kaniselvan, Benjamin Kurt Miller, Meng Gao, Juno Nam, Daniel S. Levine

**Abstract:** We introduce HELM, a state-of-the-art Hamiltonian prediction model scaling to 100+ atom structures with 58 elements and large basis sets. We release the 'OMol_CSH_58k' dataset and demonstrate 'Hamiltonian pretraining' improves energy-prediction in low-data regimes.

---

### Chemical Properties from Graph Neural Network-Predicted Electron Densities
**arXiv:** [2309.04811](https://arxiv.org/abs/2309.04811) (2023)

**Authors:** Ethan M. Sunshine, Muhammed Shuaibi, Zachary W. Ulissi, John R. Kitchin

**Abstract:** We demonstrate using established physical methods to obtain chemical properties from model-predicted electron densities. Without training to predict charges, the model predicts atomic charges with an order of magnitude lower error than sum of atomic densities.

---

### The Loss Landscape of Powder X-Ray Diffraction-Based Structure Optimization
**arXiv:** [2512.04036](https://arxiv.org/abs/2512.04036) (2025)

**Authors:** Nofit Segal, Akshay Subramanian, Mingda Li, Benjamin Kurt Miller, Rafael Gomez-Bombarelli

**Abstract:** We study the powder XRD-to-structure mapping using gradient descent. Commonly used XRD similarity metrics result in highly non-convex landscapes. Constraining optimization to the ground-truth crystal family significantly improves structure recovery.

---

### A Physically-informed Graph-based Order Parameter
**arXiv:** [2106.08215](https://arxiv.org/abs/2106.08215) (2021)

**Authors:** James Chapman, Nir Goldman, Brandon Wood

**Abstract:** A universal, transferable graph-based order parameter for characterizing atomistic structures. Validated on liquid lithium up to 300 GPa, carbon phases including nanotubes, and diverse aluminum configurations. Outperforms all existing crystalline order parameters.
```

---

## Perspectives & Introductions

Overview papers and perspectives on the field.

```{admonition} Perspectives & Introductions
:class: dropdown

### Open Challenges in Developing Large Scale ML Models for Catalyst Discovery
**arXiv:** [2206.02005](https://arxiv.org/abs/2206.02005) (2022)

**Authors:** Adeesh Kolluru, Muhammed Shuaibi, Aini Palizhati, Nima Shoghi, et al.

**Abstract:** We discuss challenges and findings of developments on OC20, examining performance across materials and adsorbates. We cover energy-conservation, finding local minima, and augmentation of off-equilibrium data.

---

### An Introduction to Electrocatalyst Design using Machine Learning
**arXiv:** [2010.09435](https://arxiv.org/abs/2010.09435) (2020)

**Authors:** C. Lawrence Zitnick, Lowik Chanussot, Abhishek Das, Siddharth Goyal, et al.

**Abstract:** An introduction to challenges in finding electrocatalysts for renewable energy storage, how machine learning may be applied, and the use of the Open Catalyst Project OC20 dataset for model training.

---

### Computational Catalyst Discovery: Active Classification through Myopic Multiscale Sampling
**arXiv:** [2102.01528](https://arxiv.org/abs/2102.01528) (2021)

**Authors:** Kevin Tran, Willie Neiswanger, Kirby Broderick, Eric Xing, Jeff Schneider, Zachary W. Ulissi

**Abstract:** We present myopic multiscale sampling, which combines multiscale modeling with automated DFT selection. This active classification strategy achieves ~7-16x speedup in catalyst classification relative to random sampling.
```
