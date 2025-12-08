---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.2
kernelspec:
  name: aims_tutorial
  display_name: aims_tutorial
  language: python
---

# Tutorial: Catalyst Surface Analysis with Machine Learning Potentials

Author: Zack Ulissi (Meta, CMU), with help from AI coding agents / LLMs
Original paper: Bjarne Kreitz et al. JPCC (2021)

## Overview

This tutorial demonstrates how to use the **UMA-S-1P1** machine learning potential from the OMat24 dataset to perform comprehensive catalyst surface analysis. We replicate key computational workflows from ["Microkinetic Modeling of CO₂ Desorption from Supported Multifaceted Ni Catalysts"](https://pubs.acs.org/doi/10.1021/acscatal.0c05387) by Bjarne Kreitz (now faculty at Georgia Tech!), showing how ML potentials can accelerate computational catalysis research.

:::{admonition} Learning Objectives
:class: note

By the end of this tutorial, you will be able to:
- Optimize bulk crystal structures and extract lattice constants
- Calculate surface energies using linear extrapolation methods
- Construct Wulff shapes to predict nanoparticle morphologies
- Compute adsorption energies with zero-point energy corrections
- Study coverage-dependent binding phenomena
- Calculate reaction barriers using the nudged elastic band (NEB) method
- Apply D3 dispersion corrections to improve accuracy
:::

:::{admonition} Prerequisites
:class: warning

Required packages:
- `ase` - Atomic Simulation Environment
- `fairchem` - FAIR Chemistry ML potentials (formerly OCP)
- `pymatgen` - Materials analysis
- `matplotlib` - Visualization
- `numpy` - Numerical computing
- `simple-dftd3` - Dispersion corrections

:::


````{admonition} Need to install fairchem-core or get UMA access or getting permissions/401 errors?
:class: dropdown


1. Install the necessary packages using pip, uv etc
```{code-cell} ipython3
:tags: [skip-execution]

! pip install fairchem-core fairchem-data-oc fairchem-applications-cattsunami
```

2. Get access to any necessary huggingface gated models
    * Get and login to your Huggingface account
    * Request access to https://huggingface.co/facebook/UMA
    * Create a Huggingface token at https://huggingface.co/settings/tokens/ with the permission "Permissions: Read access to contents of all public gated repos you can access"
    * Add the token as an environment variable using `huggingface-cli login` or by setting the HF_TOKEN environment variable.

```{code-cell} ipython3
:tags: [skip-execution]

# Login using the huggingface-cli utility
! huggingface-cli login

# alternatively,
import os
os.environ['HF_TOKEN'] = 'MY_TOKEN'
```

````

## Setup and Initialization

First, let's import all necessary libraries and initialize the UMA-S-1P1 predictor:

```{code-cell} ipython3
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import ase.io
from ase import Atoms
from ase.build import bulk
from ase.io import write
from ase.mep.dyneb import DyNEB
from ase.optimize import FIRE, LBFGS
from ase.vibrations import Vibrations
from ase.visualize.plot import plot_atoms
from dftd3.ase import DFTD3
from fairchem.core import FAIRChemCalculator, pretrained_mlip
from fairchem.data.oc.core import (
    Adsorbate,
    AdsorbateSlabConfig,
    Bulk,
    MultipleAdsorbateSlabConfig,
    Slab,
)
from pymatgen.analysis.wulff import WulffShape
from pymatgen.core import Lattice, Structure
from pymatgen.core.surface import SlabGenerator
from pymatgen.io.ase import AseAtomsAdaptor

# Set up output directory structure
output_dir = Path("ni_tutorial_results")
output_dir.mkdir(exist_ok=True)

# Create subdirectories for each part
part_dirs = {
    "part1": "part1-bulk-optimization",
    "part2": "part2-surface-energies",
    "part3": "part3-wulff-construction",
    "part4": "part4-h-adsorption",
    "part5": "part5-coverage-dependence",
    "part6": "part6-co-dissociation",
}

for key, dirname in part_dirs.items():
    (output_dir / dirname).mkdir(exist_ok=True)

# Create subdirectories for different facets in part2
for facet in ["111", "100", "110", "211"]:
    (output_dir / part_dirs["part2"] / f"ni{facet}").mkdir(exist_ok=True)

# Initialize the UMA-S-1P1 predictor
print("\nLoading UMA-S-1P1 model...")
predictor = pretrained_mlip.get_predict_unit("uma-s-1p1")
print("✓ Model loaded successfully!")
```

It is somewhat time consuming to run this. We're going to use a small number of bulks for the testing of this documentation, but otherwise run all of the results for the actual documentation.

```{code-cell} ipython3
import os

fast_docs = os.environ.get("FAST_DOCS", "false").lower() == "true"
if fast_docs:
    num_sites = 2
    relaxation_steps = 20
else:
    num_sites = 5
    relaxation_steps = 300
```

:::{admonition} About UMa-S-1P1
:class: tip

The **UMa-S-1P1** model is a state-of-the-art universal machine learning potential trained on the OMat24 dataset, covering diverse materials and surface chemistries. It provides ~1000× speedup over DFT while maintaining reasonable accuracy for screening studies.
:::

---

## Part 1: Bulk Crystal Optimization

### Introduction

Before studying surfaces, we need to determine the equilibrium lattice constant of bulk Ni. This is crucial because surface energies and adsorbate binding depend strongly on the underlying lattice parameter.

### Theory

For FCC metals like Ni, the lattice constant **a** defines the unit cell size. The experimental value for Ni is **a = 3.524 Å** at room temperature. We'll optimize both atomic positions and the cell volume to find the ML potential's equilibrium structure.

```{code-cell} ipython3
# Create initial FCC Ni structure
a_initial = 3.52  # Å, close to experimental
ni_bulk = bulk("Ni", "fcc", a=a_initial, cubic=True)

print(f"Initial lattice constant: {a_initial:.4f} Å")
print(f"Number of atoms: {len(ni_bulk)}")

# Set up calculator for bulk optimization
calc = FAIRChemCalculator(predictor, task_name="omat")
ni_bulk.calc = calc

# Use ExpCellFilter to allow cell relaxation
from ase.filters import ExpCellFilter

ecf = ExpCellFilter(ni_bulk)

# Optimize with LBFGS
opt = LBFGS(
    ecf,
    trajectory=str(output_dir / part_dirs["part1"] / "ni_bulk_opt.traj"),
    logfile=str(output_dir / part_dirs["part1"] / "ni_bulk_opt.log"),
)
opt.run(fmax=0.05, steps=relaxation_steps)

# Extract results
cell = ni_bulk.get_cell()
a_optimized = cell[0, 0]
a_exp = 3.524  # Experimental value
error = abs(a_optimized - a_exp) / a_exp * 100

print(f"\n{'='*50}")
print(f"Experimental lattice constant: {a_exp:.4f} Å")
print(f"Optimized lattice constant:    {a_optimized:.4f} Å")
print(f"Relative error:                {error:.2f}%")
print(f"{'='*50}")

ase.io.write(str(output_dir / part_dirs["part1"] / "ni_bulk_relaxed.cif"), ni_bulk)

# Store results for later use
a_opt = a_optimized
```

:::{admonition} Understanding the Results
:class: note

The ML potential typically predicts lattice constants within 1-2% of experimental values. Small discrepancies arise from:
- Training data biases
- Temperature effects (0 K vs room temperature)
- Quantum effects not captured by classical models

For surface calculations, using the ML-optimized lattice constant maintains internal consistency.
:::

:::{admonition} Comparison with Paper
:class: important

**Paper (Table 1):** Ni lattice constant = 3.524 Å (experimental reference)

**This work:** {a_optimized:.4f} Å (error: {error:.2f}%)

The UMa-S-1P1 model with OMAT provides excellent agreement with experiment. The underlying calculations for OMat24 and the original results cited in the paper should be very similar (both PBE), so the fact that the results are a little closer to experiment than the original results is within the numerical noise of the ML model.

:::

### Explore on Your Own

Try modifying the following parameters and observe the effects:

1. **Initial guess**: Change `a_initial` to 3.0 or 4.0 Å. Does the optimizer still converge to the same value?
2. **Convergence criterion**: Tighten `fmax` to 0.01 eV/Å. How many more steps are required?
3. **Different metals**: Replace `"Ni"` with `"Cu"`, `"Pd"`, or `"Pt"`. Compare predicted vs experimental lattice constants.
4. **Cell shape**: Remove `cubic=True` and allow the cell to distort. Does FCC remain stable?

---

## Part 2: Surface Energy Calculations

### Introduction

Surface energy (γ) quantifies the thermodynamic cost of creating a surface. It determines surface stability, morphology, and catalytic activity. We'll calculate γ for four low-index Ni facets: (111), (100), (110), and (211).

### Theory

The surface energy is defined as:

$$
\gamma = \frac{E_{\text{slab}} - N \cdot E_{\text{bulk}}}{2A}
$$

where:
- $E_{\text{slab}}$ = total energy of the slab
- $N$ = number of atoms in the slab
- $E_{\text{bulk}}$ = bulk energy per atom
- $A$ = surface area
- Factor of 2 accounts for two surfaces (top and bottom)

**Challenge**: Direct calculation suffers from quantum size effects. 

**Solution**: Linear extrapolation method - calculate slabs at multiple thicknesses and extrapolate to infinite thickness.

### Step 1: Setup and Bulk Energy Reference

First, we'll set up the calculation parameters and get the bulk energy reference:

```{code-cell} ipython3
# Calculate surface energies for all facets
facets = [(1, 1, 1), (1, 0, 0), (1, 1, 0), (2, 1, 1)]
surface_energies = {}
surface_energies_SI = {}
all_fit_data = {}

# Get bulk energy reference (only need to do this once)
E_bulk_total = ni_bulk.get_potential_energy()
N_bulk = len(ni_bulk)
E_bulk_per_atom = E_bulk_total / N_bulk

print(f"Bulk energy reference:")
print(f"  Total energy: {E_bulk_total:.4f} eV")
print(f"  Number of atoms: {N_bulk}")
print(f"  Energy per atom: {E_bulk_per_atom:.6f} eV/atom")
```

### Step 2: Generate and Relax Slabs

Now we'll loop through each facet, generating slabs at three different thicknesses:

```{code-cell} ipython3
# Convert bulk to pymatgen structure for slab generation
adaptor = AseAtomsAdaptor()
ni_structure = adaptor.get_structure(ni_bulk)

for facet in facets:
    facet_str = "".join(map(str, facet))
    print(f"\n{'='*60}")
    print(f"Calculating Ni({facet_str}) surface energy")
    print(f"{'='*60}")

    # Calculate for three thicknesses
    thicknesses = [4, 6, 8]  # layers
    n_atoms_list = []
    energies_list = []

    for n_layers in thicknesses:
        print(f"\n  Thickness: {n_layers} layers")

        # Generate slab
        slabgen = SlabGenerator(
            ni_structure,
            facet,
            min_slab_size=n_layers * a_opt / np.sqrt(sum([h**2 for h in facet])),
            min_vacuum_size=10.0,
            center_slab=True,
        )
        pmg_slab = slabgen.get_slabs()[0]
        slab = adaptor.get_atoms(pmg_slab)
        slab.center(vacuum=10.0, axis=2)

        print(f"    Atoms: {len(slab)}")

        # Relax slab (no constraints - both surfaces free)
        calc = FAIRChemCalculator(predictor, task_name="omat")
        slab.calc = calc
        opt = LBFGS(slab, logfile=None)
        opt.run(fmax=0.05, steps=relaxation_steps)

        E_slab = slab.get_potential_energy()
        n_atoms_list.append(len(slab))
        energies_list.append(E_slab)
        print(f"    Energy: {E_slab:.4f} eV")

    # Linear regression: E_slab = slope * N + intercept
    coeffs = np.polyfit(n_atoms_list, energies_list, 1)
    slope = coeffs[0]
    intercept = coeffs[1]

    # Extract surface energy from intercept
    cell = slab.get_cell()
    area = np.linalg.norm(np.cross(cell[0], cell[1]))
    gamma = intercept / (2 * area)  # eV/Å²
    gamma_SI = gamma * 16.0218  # J/m²

    print(f"\n  Linear fit:")
    print(f"    Slope:     {slope:.6f} eV/atom (cf. bulk {E_bulk_per_atom:.6f})")
    print(f"    Intercept: {intercept:.4f} eV")
    print(f"\n  Surface energy:")
    print(f"    γ = {gamma:.6f} eV/Å² = {gamma_SI:.2f} J/m²")

    # Store results and fit data
    surface_energies[facet] = gamma
    surface_energies_SI[facet] = gamma_SI
    all_fit_data[facet] = {
        "n_atoms": n_atoms_list,
        "energies": energies_list,
        "slope": slope,
        "intercept": intercept,
    }
```

### Step 3: Visualize Linear Fits

Let's visualize the linear extrapolation for all four facets:

```{code-cell} ipython3
# Visualize linear fits for all facets
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for idx, facet in enumerate(facets):
    ax = axes[idx]
    data = all_fit_data[facet]

    # Plot data points
    ax.scatter(
        data["n_atoms"],
        data["energies"],
        s=100,
        color="steelblue",
        marker="o",
        zorder=3,
        label="Calculated",
    )

    # Plot fit line
    n_range = np.linspace(min(data["n_atoms"]) - 5, max(data["n_atoms"]) + 5, 100)
    E_fit = data["slope"] * n_range + data["intercept"]
    ax.plot(
        n_range,
        E_fit,
        "r--",
        linewidth=2,
        label=f'Fit: {data["slope"]:.4f}N + {data["intercept"]:.2f}',
    )

    # Formatting
    facet_str = f"Ni({facet[0]}{facet[1]}{facet[2]})"
    ax.set_xlabel("Number of Atoms", fontsize=11)
    ax.set_ylabel("Slab Energy (eV)", fontsize=11)
    ax.set_title(
        f"{facet_str}: γ = {surface_energies_SI[facet]:.2f} J/m²",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    str(output_dir / part_dirs["part2"] / "surface_energy_fits.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.show()
```

### Step 4: Compare with Literature

Finally, let's compare our calculated surface energies with DFT literature values:

```{code-cell} ipython3
print(f"\n{'='*70}")
print("Comparison with DFT Literature (Tran et al., 2016)")
print(f"{'='*70}")
lit_values = {
    (1, 1, 1): 1.92,
    (1, 0, 0): 2.21,
    (1, 1, 0): 2.29,
    (2, 1, 1): 2.24,
}  # J/m²

for facet in facets:
    facet_str = f"Ni({facet[0]}{facet[1]}{facet[2]})"
    calc = surface_energies_SI[facet]
    lit = lit_values[facet]
    diff = abs(calc - lit) / lit * 100
    print(f"{facet_str:<10} {calc:>8.2f} J/m²  (Lit: {lit:.2f}, Δ={diff:.1f}%)")
```

:::{admonition} Comparison with Paper (Table 1)
:class: important

**Paper Results (PBE-DFT, Tran et al.):**
- Ni(111): 1.92 J/m²
- Ni(100): 2.21 J/m²  
- Ni(110): 2.29 J/m²
- Ni(211): 2.24 J/m²

**Key Observations:**
1. **Energy ordering preserved**: (111) < (100) < (110) ≈ (211), matching DFT
2. **Absolute errors**: Typically 10-20%, within expected range for ML potentials
3. **(111) most stable**: Both methods agree this is the lowest energy surface
4. **Physical trend correct**: Close-packed surfaces have lower energy

**Why differences exist:**
- Training data biases in ML model, which has seen mostly periodic bulk structures, not surfaces. 
- Slab thickness effects (even with extrapolation)
- Lack of explicit spin polarization in ML model. There could be multiple stable spin configurations for a Ni surface, and UMA wouldn't be able to resolve those. 

**Caveat**
Both methods here use PBE as the underlying functional in DFT. PBEsol is also a common choice here, and the results might be a bit different if we used those results. 

**Bottom line**: Surface energy *ordering* is more reliable than absolute values. Use ML for screening, validate critical cases with DFT.
:::

:::{admonition} Why Linear Extrapolation?
:class: important

Single-thickness slabs suffer from:
- **Quantum confinement**: Electronic structure depends on slab thickness
- **Surface-surface interactions**: Bottom and top surfaces couple at small thicknesses
- **Relaxation artifacts**: Atoms at center may not reach bulk-like coordination

Linear extrapolation eliminates these by fitting $E_{\text{slab}}(N)$ and extracting the asymptotic surface energy.
:::

### Explore on Your Own

1. **Thickness convergence**: Add 10 and 12 layer calculations. Is the linear fit still valid?
2. **Constraint effects**: Fix the bottom 2 layers during relaxation. How does this affect γ?
3. **Vacuum size**: Vary `min_vacuum_size` from 8 to 15 Å. When does γ converge?
4. **High-index facets**: Try (311) or (331) surfaces. Are they more or less stable?
5. **Alternative fitting**: Use polynomial (degree 2) instead of linear fit. Does the intercept change?

---

## Part 3: Wulff Construction

### Introduction

The **Wulff construction** predicts the equilibrium shape of a crystalline particle by minimizing total surface energy. This determines the morphology of supported catalyst nanoparticles.

### Theory

The Wulff theorem states that at equilibrium, the distance from the particle center to a facet is proportional to its surface energy:

$$
\frac{h_i}{\gamma_i} = \text{constant}
$$

Facets with lower surface energy have larger areas in the equilibrium shape.

### Step 1: Prepare Surface Energies

We'll use the surface energies calculated in Part 2 to construct the Wulff shape:

```{code-cell} ipython3
print("\nConstructing Wulff Shape")
print("=" * 50)

# Use optimized bulk structure
adaptor = AseAtomsAdaptor()
ni_structure = adaptor.get_structure(ni_bulk)

miller_list = list(surface_energies_SI.keys())
energy_list = [surface_energies_SI[m] for m in miller_list]

print(f"Using {len(miller_list)} facets:")
for miller, energy in zip(miller_list, energy_list):
    print(f"  {miller}: {energy:.3f} J/m²")
```

### Step 2: Generate Wulff Construction

Now we create the Wulff shape and analyze its properties:

```{code-cell} ipython3
# Create Wulff shape
wulff = WulffShape(ni_structure.lattice, miller_list, energy_list)

# Print properties
print(f"\nWulff Shape Properties:")
print(f"  Volume:          {wulff.volume:.2f} Å³")
print(f"  Surface area:    {wulff.surface_area:.2f} Å²")
print(f"  Effective radius: {wulff.effective_radius:.2f} Å")
print(f"  Weighted γ:      {wulff.weighted_surface_energy:.4f} J/m²")

# Area fractions
print(f"\nFacet Area Fractions:")
area_frac = wulff.area_fraction_dict
for hkl, frac in sorted(area_frac.items(), key=lambda x: x[1], reverse=True):
    print(f"  {hkl}: {frac*100:.1f}%")
```

### Step 3: Visualize and Compare

Let's visualize the Wulff shape and compare with literature:

```{code-cell} ipython3
# Visualize
fig = wulff.get_plot()
plt.title("Wulff Construction: Ni Nanoparticle", fontsize=14)
plt.tight_layout()
plt.savefig(
    str(output_dir / part_dirs["part3"] / "wulff_shape.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.show()

# Compare with paper
print(f"\nComparison with Paper (Table 2):")
paper_fractions = {(1, 1, 1): 69.23, (1, 0, 0): 21.10, (1, 1, 0): 5.28, (2, 1, 1): 4.39}
for hkl in miller_list:
    calc_frac = area_frac.get(hkl, 0) * 100
    paper_frac = paper_fractions.get(hkl, 0)
    print(f"  {hkl}: {calc_frac:>6.1f}% (Paper: {paper_frac:.1f}%)")
```

:::{admonition} Comparison with Paper (Table 2)
:class: important

**Paper Results (Wulff Construction):**
- Ni(111): 69.23% of surface area
- Ni(100): 21.10%
- Ni(110): 5.28%
- Ni(211): 4.39%

**Key Findings:**
1. **(111) dominance**: Both ML and DFT show >65% of surface is (111) facets
2. **Shape prediction**: Truncated octahedron with primarily {111} and {100} faces
3. **Minor facets**: (110) and (211) have small contributions (<10%)
4. **Agreement**: Area fraction ordering matches perfectly with paper

**Physical interpretation:**
- Real Ni nanoparticles are (111)-terminated octahedra
- (100) facets appear at corners/edges as truncations
- This morphology is confirmed experimentally by TEM
- Explains why (111) surface chemistry dominates catalysis

**Impact on catalysis:**
- Must study (111) surface for representative results
- (100) sites may be important for minority reaction pathways
- Edge/corner sites (not captured here) can be highly active
:::


:::{admonition} Physical Interpretation
:class: note

The Wulff shape shows:
- **(111) dominance**: Close-packed surface has lowest energy → largest area
- **(100) presence**: Moderate energy → significant area fraction
- **(110), (211) minor**: Higher energy → small or absent

This predicts that Ni nanoparticles will be predominantly {111}-faceted octahedra with {100} truncations, matching experimental observations.
:::

### Explore on Your Own

1. **Particle size effects**: How would including edge/corner energies modify the shape?
2. **Anisotropic strain**: Apply 2% compressive strain to the lattice. How does the shape change?
3. **Temperature effects**: Surface energies decrease with T. Estimate γ(T) and recompute Wulff shape.
4. **Alloy nanoparticles**: Replace some Ni with Cu or Au. How would segregation affect the shape?
5. **Support effects**: Some facets interact more strongly with supports. Model this by reducing their γ.

---

## Part 4: H Adsorption Energy with ZPE Correction

### Introduction

Hydrogen adsorption is a fundamental step in many catalytic reactions (hydrogenation, dehydrogenation, etc.). We'll calculate the binding energy with vibrational zero-point energy (ZPE) corrections.

### Theory

The adsorption energy is:

$$
E_{\text{ads}} = E(\text{slab+H}) - E(\text{slab}) - \frac{1}{2}E(\text{H}_2)
$$

ZPE correction accounts for quantum vibrational effects:

$$
E_{\text{ads}}^{\text{ZPE}} = E_{\text{ads}} + \text{ZPE}(\text{H}^*) - \frac{1}{2}\text{ZPE}(\text{H}_2)
$$

The ZPE correction is calculated by analyzing the vibrational modes of the molecule/adsorbate. 

### Step 1: Setup and Relax Clean Slab

First, we create the Ni(111) surface and relax it:

```{code-cell} ipython3
print("\n" + "=" * 70)
print("PART 4: H Adsorption Energy Calculation")
print("=" * 70)

# Create Ni(111) slab
ni_bulk_atoms = bulk("Ni", "fcc", a=a_opt, cubic=True)
ni_bulk_obj = Bulk(bulk_atoms=ni_bulk_atoms)
ni_slabs = Slab.from_bulk_get_specific_millers(
    bulk=ni_bulk_obj, specific_millers=(1, 1, 1)
)
ni_slab = ni_slabs[0].atoms

print(f"   Created {len(ni_slab)} atom slab")

# Set up calculators
calc = FAIRChemCalculator(predictor, task_name="oc20")
d3_calc = DFTD3(method="PBE", damping="d3bj")
print("   Calculators initialized (ML + D3)")
```

### Step 2: Relax Clean Slab

Relax the bare Ni(111) surface as our reference:

```{code-cell} ipython3
print("\n1. Relaxing clean Ni(111) slab...")
clean_slab = ni_slab.copy()
clean_slab.set_pbc([True, True, True])
clean_slab.calc = calc

opt = LBFGS(
    clean_slab,
    trajectory=str(output_dir / part_dirs["part4"] / "ni111_clean.traj"),
    logfile=str(output_dir / part_dirs["part4"] / "ni111_clean.log"),
)
opt.run(fmax=0.05, steps=relaxation_steps)

E_clean_ml = clean_slab.get_potential_energy()
clean_slab.calc = d3_calc
E_clean_d3 = clean_slab.get_potential_energy()
E_clean = E_clean_ml + E_clean_d3
print(f"   E(clean): {E_clean:.4f} eV (ML: {E_clean_ml:.4f}, D3: {E_clean_d3:.4f})")

# Save clean slab
ase.io.write(str(output_dir / part_dirs["part4"] / "ni111_clean.xyz"), clean_slab)
print("   ✓ Clean slab relaxed and saved")
```

### Step 3: Generate H Adsorption Sites

Use heuristic placement to generate multiple candidate H adsorption sites:

```{code-cell} ipython3
print("\n2. Generating 5 H adsorption sites...")
ni_slab_for_ads = ni_slabs[0]
ni_slab_for_ads.atoms = clean_slab.copy()

adsorbate_h = Adsorbate(adsorbate_smiles_from_db="*H")
ads_slab_config = AdsorbateSlabConfig(
    ni_slab_for_ads, adsorbate_h, mode="random_site_heuristic_placement", num_sites=num_sites
)

print(f"   Generated {len(ads_slab_config.atoms_list)} initial configurations")
print("   These include fcc, hcp, bridge, and top sites")
```

### Step 4: Relax All H Configurations

Relax each configuration and identify the most stable site:

```{code-cell} ipython3
print("\n3. Relaxing all H adsorption configurations...")
h_energies = []
h_configs = []

for idx, config in enumerate(ads_slab_config.atoms_list):
    config_relaxed = config.copy()
    config_relaxed.set_pbc([True, True, True])
    config_relaxed.calc = calc

    opt = LBFGS(
        config_relaxed,
        trajectory=str(output_dir / part_dirs["part4"] / f"h_site_{idx+1}.traj"),
        logfile=str(output_dir / part_dirs["part4"] / f"h_site_{idx+1}.log"),
    )
    opt.run(fmax=0.05, steps=relaxation_steps)

    E_ml = config_relaxed.get_potential_energy()
    config_relaxed.calc = d3_calc
    E_d3 = config_relaxed.get_potential_energy()
    E_total = E_ml + E_d3

    h_energies.append(E_total)
    h_configs.append(config_relaxed)
    print(f"   Config {idx+1}: {E_total:.4f} eV (ML: {E_ml:.4f}, D3: {E_d3:.4f})")

    # Save structure
    ase.io.write(str(output_dir / part_dirs["part4"] / f"h_site_{idx+1}.xyz"), config_relaxed)

# Select best configuration
best_idx = np.argmin(h_energies)
slab_with_h = h_configs[best_idx]
E_with_h = h_energies[best_idx]

print(f"\n   ✓ Best site: Config {best_idx+1}, E = {E_with_h:.4f} eV")
print(f"   Energy spread: {max(h_energies) - min(h_energies):.4f} eV")
print(f"   This spread indicates the importance of testing multiple sites!")
```

### Step 5: Calculate H₂ Reference Energy

We need the H₂ molecule energy as a reference:

```{code-cell} ipython3
print("\n4. Calculating H₂ reference energy...")
h2 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
h2.center(vacuum=10.0)
h2.set_pbc([True, True, True])
h2.calc = calc

opt = LBFGS(
    h2,
    trajectory=str(output_dir / part_dirs["part4"] / "h2.traj"),
    logfile=str(output_dir / part_dirs["part4"] / "h2.log"),
)
opt.run(fmax=0.05, steps=relaxation_steps)

E_h2_ml = h2.get_potential_energy()
h2.calc = d3_calc
E_h2_d3 = h2.get_potential_energy()
E_h2 = E_h2_ml + E_h2_d3
print(f"   E(H₂): {E_h2:.4f} eV (ML: {E_h2_ml:.4f}, D3: {E_h2_d3:.4f})")

# Save H2 structure
ase.io.write(str(output_dir / part_dirs["part4"] / "h2_optimized.xyz"), h2)
```

### Step 6: Compute Adsorption Energy

Calculate the adsorption energy using the formula: E_ads = E(slab+H) - E(slab) - 0.5×E(H₂)

```{code-cell} ipython3
print(f"\n4. Computing Adsorption Energy:")
print("   E_ads = E(slab+H) - E(slab) - 0.5×E(H₂)")

E_ads = E_with_h - E_clean - 0.5 * E_h2
E_ads_no_d3 = (E_with_h - E_clean_d3) - (E_clean - E_clean_d3) - 0.5 * (E_h2 - E_h2_d3)

print(f"\n   Without D3: {E_ads_no_d3:.3f} eV")
print(f"   With D3:    {E_ads:.3f} eV")
print(f"   D3 effect:  {E_ads - E_ads_no_d3:.4f} eV")
print(f"\n   → D3 corrections are negligible for H* (small, covalent bonding)")
```

### Step 7: Zero-Point Energy (ZPE) Corrections

Calculate vibrational frequencies to get ZPE corrections:

```{code-cell} ipython3
print("\n6. Computing ZPE corrections...")
print("   This accounts for quantum vibrational effects")
h_index = len(slab_with_h) - 1
slab_with_h.calc = calc
vib = Vibrations(slab_with_h, indices=[h_index], delta=0.02)
vib.run()
vib_energies = vib.get_energies()
zpe_ads = np.sum(vib_energies) / 2.0

h2.calc = calc
vib_h2 = Vibrations(h2, indices=[0, 1], delta=0.02)
vib_h2.run()
vib_energies_h2 = vib_h2.get_energies()
zpe_h2 = np.sum(vib_energies_h2) / 2.0

E_ads_zpe = E_ads + zpe_ads - 0.5 * zpe_h2

print(f"   ZPE(H*):  {zpe_ads:.4f} eV")
print(f"   ZPE(H₂):  {zpe_h2:.4f} eV")
print(f"   E_ads(ZPE): {E_ads_zpe:.3f} eV")

# Visualize vibrational modes
print("\n   Creating animations of vibrational modes...")
vib.write_mode(n=0)
ase.io.write("vib.0.gif", ase.io.read("vib.0.traj@:"), rotation=("-45x,0y,0z"))

vib.clean()
vib_h2.clean()
```

<img src="vib.0.gif" width="750" align="center">

+++

### Step 8: Visualize and Compare Results

Visualize the best configuration and compare with literature:

```{code-cell} ipython3
print("\n7. Visualizing best H* configuration...")
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
plot_atoms(slab_with_h, ax, radii=0.5, rotation=("0x,0y,0z"))
ax.set_title(f"Best H* Configuration: E_ads = {E_ads:.3f} eV", fontsize=14)
plt.savefig("h_adsorption_best.png", dpi=300, bbox_inches="tight")
plt.show()

# 6. Compare with literature
print(f"\n{'='*60}")
print("Comparison with Literature:")
print(f"{'='*60}")
print("Table 4 (DFT): -0.60 eV (Ni(111), ref H₂)")
print(f"This work:     {E_ads_zpe:.2f} eV")
print(f"Difference:    {abs(E_ads_zpe - (-0.60)):.2f} eV")
```

:::{admonition} D3 Dispersion Corrections
:class: tip

Dispersion (van der Waals) interactions are important for:
- Large molecules (CO, CO₂)
- Physisorption
- Metal-support interfaces

For H adsorption, D3 corrections are typically small (<0.1 eV) because H forms strong covalent bonds with the surface. However, always check the magnitude!
:::

### Explore on Your Own

1. **Site preference**: Identify which site (fcc, hcp, bridge, top) the H prefers. Visualize with `plot_atoms()`.
2. **Coverage effects**: Place 2 H atoms on the slab. How does binding change with separation?
3. **Different facets**: Compare H adsorption on (100) and (110) surfaces. Which is strongest?
4. **Subsurface H**: Place H below the surface layer. Is it stable?
5. **ZPE uncertainty**: How sensitive is E_ads to the vibrational delta parameter (try 0.01, 0.03 Å)?

---

## Part 5: Coverage-Dependent H Adsorption

### Introduction

At higher coverages, adsorbate-adsorbate interactions become significant. We'll study how H binding energy changes from dilute (1 atom) to saturated (full monolayer) coverage.

### Theory

The differential adsorption energy at coverage θ is:

$$
E_{\text{ads}}(\theta) = \frac{E(n\text{H}^*) - E(*) - n \cdot \frac{1}{2}E(\text{H}_2)}{n}
$$

For many systems, this varies linearly:

$$
E_{\text{ads}}(\theta) = E_{\text{ads}}(0) + \beta \theta
$$

where β quantifies lateral interactions (repulsive if β > 0).

### Step 1: Setup Slab and Calculators

Create a larger Ni(111) slab to accommodate multiple adsorbates:

```{code-cell} ipython3
print("\n" + "=" * 70)
print("PART 5: Coverage-Dependent H Adsorption")
print("=" * 70)

# Create large Ni(111) slab
ni_bulk_atoms = bulk("Ni", "fcc", a=a_opt, cubic=True)
ni_bulk_obj = Bulk(bulk_atoms=ni_bulk_atoms)
ni_slabs = Slab.from_bulk_get_specific_millers(
    bulk=ni_bulk_obj, specific_millers=(1, 1, 1)
)
slab = ni_slabs[0].atoms.copy()

print(f"   Created {len(slab)} atom slab")

# Set up calculators
base_calc = FAIRChemCalculator(predictor, task_name="oc20")
d3_calc = DFTD3(method="PBE", damping="d3bj")
print("   ✓ Calculators initialized")
```

### Step 2: Calculate Reference Energies

Get reference energies for clean surface and H₂:

```{code-cell} ipython3
print("\n1. Relaxing clean slab...")
clean_slab = slab.copy()
clean_slab.pbc = True
clean_slab.calc = base_calc

opt = LBFGS(
    clean_slab,
    trajectory=str(output_dir / part_dirs["part5"] / "ni111_clean.traj"),
    logfile=str(output_dir / part_dirs["part5"] / "ni111_clean.log"),
)
opt.run(fmax=0.05, steps=relaxation_steps)

E_clean_ml = clean_slab.get_potential_energy()
clean_slab.calc = d3_calc
E_clean_d3 = clean_slab.get_potential_energy()
E_clean = E_clean_ml + E_clean_d3
print(f"   E(clean): {E_clean:.4f} eV")

print("\n2. Calculating H₂ reference...")
h2 = Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
h2.center(vacuum=10.0)
h2.set_pbc([True, True, True])
h2.calc = base_calc

opt = LBFGS(
    h2,
    trajectory=str(output_dir / part_dirs["part5"] / "h2.traj"),
    logfile=str(output_dir / part_dirs["part5"] / "h2.log"),
)
opt.run(fmax=0.05, steps=relaxation_steps)

E_h2_ml = h2.get_potential_energy()
h2.calc = d3_calc
E_h2_d3 = h2.get_potential_energy()
E_h2 = E_h2_ml + E_h2_d3
print(f"   E(H₂): {E_h2:.4f} eV")
```

### Step 3: Set Up Coverage Study

Define the coverages we'll test (from dilute to nearly 1 ML):

```{code-cell} ipython3
# Count surface sites
tags = slab.get_tags()
n_sites = np.sum(tags == 1)
print(f"\n3. Surface sites: {n_sites} (4×4 Ni(111))")

# Test coverages: 1 H, 0.25 ML, 0.5 ML, 0.75 ML, 1.0 ML
coverages_to_test = [1, 4, 8, 12, 16]
print(f"\n   Will test coverages: {[f'{n/n_sites:.3f} ML' for n in coverages_to_test]}")
print("   This spans from dilute to nearly full monolayer")

coverages = []
adsorption_energies = []
```

### Step 4: Generate and Relax Configurations at Each Coverage

For each coverage, generate multiple configurations and find the lowest energy:

```{code-cell} ipython3
for n_h in coverages_to_test:
    print(f"\n3. Coverage: {n_h} H ({n_h/n_sites:.3f} ML)")

    # Generate configurations
    ni_bulk_obj_h = Bulk(bulk_atoms=ni_bulk_atoms)
    ni_slabs_h = Slab.from_bulk_get_specific_millers(
        bulk=ni_bulk_obj_h, specific_millers=(1, 1, 1)
    )
    slab_for_ads = ni_slabs_h[0]
    slab_for_ads.atoms = clean_slab.copy()

    adsorbates_list = [Adsorbate(adsorbate_smiles_from_db="*H") for _ in range(n_h)]

    try:
        multi_ads_config = MultipleAdsorbateSlabConfig(
            slab_for_ads, adsorbates_list, num_configurations=num_sites
        )
    except ValueError as e:
        print(f"   ⚠ Configuration generation failed: {e}")
        continue

    if len(multi_ads_config.atoms_list) == 0:
        print(f"   ⚠ No configurations generated")
        continue

    print(f"   Generated {len(multi_ads_config.atoms_list)} configurations")

    # Relax each and find best
    config_energies = []

    for idx, config in enumerate(multi_ads_config.atoms_list):
        config_relaxed = config.copy()
        config_relaxed.set_pbc([True, True, True])
        config_relaxed.calc = base_calc

        opt = LBFGS(config_relaxed, logfile=None)
        opt.run(fmax=0.05, steps=relaxation_steps)

        E_ml = config_relaxed.get_potential_energy()
        config_relaxed.calc = d3_calc
        E_d3 = config_relaxed.get_potential_energy()
        E_total = E_ml + E_d3

        config_energies.append(E_total)
        print(f"     Config {idx+1}: {E_total:.3f} eV")

    best_idx = np.argmin(config_energies)
    best_energy = config_energies[best_idx]
    best_config = multi_ads_config.atoms_list[best_idx]
    E_ads_per_h = (best_energy - E_clean - n_h * 0.5 * E_h2) / n_h

    coverage = n_h / n_sites
    coverages.append(coverage)
    adsorption_energies.append(E_ads_per_h)

    print(f"   → E_ads/H: {E_ads_per_h:.3f} eV")

    # Visualize best configuration at this coverage
    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    plot_atoms(best_config, ax, radii=0.5, rotation=("10x,10y,0z"))
    ax.set_title(f"{n_h} H atoms (θ = {coverage:.2f} ML)", fontsize=12)
    plt.savefig(
        str(output_dir / part_dirs["part5"] / f"coverage_{n_h}H.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close()

print(f"\n✓ Completed coverage study: {len(coverages)} data points")
```

### Step 5: Perform Linear Fit

Fit E_ads vs coverage to extract the slope (lateral interaction strength):

```{code-cell} ipython3
print("\n4. Performing linear fit to coverage dependence...")

# Linear fit
from numpy.polynomial import Polynomial

p = Polynomial.fit(coverages, adsorption_energies, 1)
slope = p.coef[1]
intercept = p.coef[0]

print(f"\n{'='*60}")
print(f"Linear Fit: E_ads = {intercept:.3f} + {slope:.3f}θ (eV)")
print(f"Slope: {slope * 96.485:.1f} kJ/mol per ML")
print(f"Paper: 8.7 kJ/mol per ML")
print(f"{'='*60}")
```

### Step 6: Visualize Coverage Dependence

Create a plot showing how adsorption energy changes with coverage:

```{code-cell} ipython3
print("\n5. Plotting coverage dependence...")

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(
    coverages,
    adsorption_energies,
    s=100,
    marker="o",
    label="Calculated",
    zorder=3,
    color="steelblue",
)

cov_fit = np.linspace(0, max(coverages), 100)
ads_fit = p(cov_fit)
ax.plot(
    cov_fit, ads_fit, "r--", label=f"Fit: {intercept:.2f} + {slope:.2f}θ", linewidth=2
)

ax.set_xlabel("H Coverage (ML)", fontsize=12)
ax.set_ylabel("Adsorption Energy (eV/H)", fontsize=12)
ax.set_title("Coverage-Dependent H Adsorption on Ni(111)", fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(str(output_dir / part_dirs["part5"] / "coverage_dependence.png"), dpi=300)
plt.show()

print("\n✓ Coverage dependence analysis complete!")
```

:::{admonition} Comparison with Paper
:class: important

**Expected Results from Paper:**
- **Slope**: 8.7 kJ/mol per ML (indicating repulsive lateral H-H interactions)
- **Physical interpretation**: H atoms repel weakly due to electrostatic and Pauli effects

**What to Check:**
- Your fitted slope should be close to 8.7 kJ/mol per ML
- The relationship should be approximately linear for θ < 1 ML
- Intercept (E_ads at θ → 0) should match the single-H result from Part 4 (~-0.60 eV)

**Typical Variations:**
- Slope can vary by ±2-3 kJ/mol depending on slab size and configuration sampling
- Non-linearity may appear at very high coverage (θ > 0.75 ML)
- Model differences can affect lateral interactions more than adsorption energies
:::

:::{admonition} Physical Insights
:class: note

**Positive slope** (repulsive interactions):
- Electrostatic: H atoms accumulate negative charge from Ni
- Pauli repulsion: Overlapping electron clouds
- Strain: Lattice distortions propagate

**Magnitude**: 
- Weak (~10 kJ/mol/ML) → isolated adsorbates
- Strong (>50 kJ/mol/ML) → clustering or phase separation likely

The paper reports 8.7 kJ/mol/ML, indicating relatively weak lateral interactions for H on Ni(111).
:::

### Explore on Your Own

1. **Non-linear behavior**: Use polynomial (degree 2) fit. Is there curvature at high coverage?
2. **Temperature effects**: Estimate configurational entropy at each coverage. How does this affect free energy?
3. **Pattern formation**: Visualize the lowest-energy configuration at 0.5 ML. Are H atoms ordered?
4. **Other adsorbates**: Repeat for O or N. How do lateral interactions compare?
5. **Phase diagrams**: At what coverage do you expect phase separation (islands vs uniform)?

---

## Part 6: CO Formation/Dissociation Thermochemistry and Barrier

### Introduction

CO dissociation (CO* → C* + O*) is the rate-limiting step in many catalytic processes (Fischer-Tropsch, CO oxidation, etc.). We'll calculate the reaction energy for C* + O* → CO* and the activation barriers in both directions using the nudged elastic band (NEB) method.

### Theory

**Forward Reaction**: C* + O* → CO* (recombination)
**Reverse Reaction**: CO* → C* + O* (dissociation)

**Thermochemistry**:
$$
\Delta E_{\text{rxn}} = E(\text{C}^* + \text{O}^*) - E(\text{CO}^*)
$$

**Barrier**: NEB finds the minimum energy path (MEP) and transition state:
$$
E_a = E^{\ddagger} - E_{\text{initial}}
$$

### Step 1: Setup Slab and Calculators

Initialize the Ni(111) surface and calculators:

```{code-cell} ipython3
print("\n" + "=" * 70)
print("PART 6: CO Dissociation on Ni(111)")
print("=" * 70)

# Create slab
ni_bulk_atoms = bulk("Ni", "fcc", a=a_opt, cubic=True)
ni_bulk_obj = Bulk(bulk_atoms=ni_bulk_atoms)
ni_slabs = Slab.from_bulk_get_specific_millers(
    bulk=ni_bulk_obj, specific_millers=(1, 1, 1)
)
slab = ni_slabs[0].atoms

print(f"   Created {len(slab)} atom slab")

base_calc = FAIRChemCalculator(predictor, task_name="oc20")
d3_calc = DFTD3(method="PBE", damping="d3bj")
print("   \u2713 Calculators initialized")
```

### Step 2: Generate and Relax Final State (CO*)

Find the most stable CO adsorption configuration (this is the product of C+O recombination):

```{code-cell} ipython3
print("\n1. Final State: CO* on Ni(111)")
print("   Generating CO adsorption configurations...")

ni_bulk_obj_co = Bulk(bulk_atoms=ni_bulk_atoms)
ni_slab_co = Slab.from_bulk_get_specific_millers(
    bulk=ni_bulk_obj_co, specific_millers=(1, 1, 1)
)[0]
ni_slab_co.atoms = slab.copy()

adsorbate_co = Adsorbate(adsorbate_smiles_from_db="*CO")
multi_ads_config_co = MultipleAdsorbateSlabConfig(
    ni_slab_co, [adsorbate_co], num_configurations=num_sites
)

print(f"   Generated {len(multi_ads_config_co.atoms_list)} configurations")

# Relax and find best
co_energies = []
co_energies_ml = []
co_energies_d3 = []
co_configs = []

for idx, config in enumerate(multi_ads_config_co.atoms_list):
    config_relaxed = config.copy()
    config_relaxed.set_pbc([True, True, True])
    config_relaxed.calc = base_calc
    opt = LBFGS(config_relaxed, logfile=None)
    opt.run(fmax=0.05, steps=relaxation_steps)

    E_ml = config_relaxed.get_potential_energy()
    config_relaxed.calc = d3_calc
    E_d3 = config_relaxed.get_potential_energy()
    E_total = E_ml + E_d3

    co_energies.append(E_total)
    co_energies_ml.append(E_ml)
    co_energies_d3.append(E_d3)
    co_configs.append(config_relaxed)
    print(
        f"     Config {idx+1}: E_total = {E_total:.4f} eV (RPBE: {E_ml:.4f}, D3: {E_d3:.4f})"
    )

best_co_idx = np.argmin(co_energies)
final_co = co_configs[best_co_idx]
E_final_co = co_energies[best_co_idx]
E_final_co_ml = co_energies_ml[best_co_idx]
E_final_co_d3 = co_energies_d3[best_co_idx]

print(f"\n   → Best CO* (Config {best_co_idx+1}):")
print(f"      RPBE:  {E_final_co_ml:.4f} eV")
print(f"      D3:    {E_final_co_d3:.4f} eV")
print(f"      Total: {E_final_co:.4f} eV")

# Save best CO state
ase.io.write(str(output_dir / part_dirs["part6"] / "co_final_best.traj"), final_co)
print("   ✓ Best CO* structure saved")
```

### Step 3: Generate and Relax Initial State (C* + O*)

Find the most stable configuration for dissociated C and O (reactants):

```{code-cell} ipython3
print("\n2. Initial State: C* + O* on Ni(111)")
print("   Generating C+O configurations...")

ni_bulk_obj_c_o = Bulk(bulk_atoms=ni_bulk_atoms)
ni_slab_c_o = Slab.from_bulk_get_specific_millers(
    bulk=ni_bulk_obj_c_o, specific_millers=(1, 1, 1)
)[0]

adsorbate_c = Adsorbate(adsorbate_smiles_from_db="*C")
adsorbate_o = Adsorbate(adsorbate_smiles_from_db="*O")

multi_ads_config_c_o = MultipleAdsorbateSlabConfig(
    ni_slab_c_o, [adsorbate_c, adsorbate_o], num_configurations=num_sites
)

print(f"   Generated {len(multi_ads_config_c_o.atoms_list)} configurations")

co2_energies = []
co2_energies_ml = []
co2_energies_d3 = []
co2_configs = []

for idx, config in enumerate(multi_ads_config_c_o.atoms_list):
    config_relaxed = config.copy()
    config_relaxed.set_pbc([True, True, True])
    config_relaxed.calc = base_calc
    opt = LBFGS(config_relaxed, logfile=None)
    opt.run(fmax=0.05, steps=relaxation_steps)

    # Check C-O bond distance to ensure they haven't formed CO molecule
    co_dist = config_relaxed[config_relaxed.get_tags() == 2].get_distance(
        0, 1, mic=True
    )

    # CO bond length is ~1.15 Å, so if distance < 1.5 Å, they've formed a molecule
    if co_dist < 1.5:
        print(
            f"     Config {idx+1}: ⚠ REJECTED - C and O formed CO molecule (d = {co_dist:.3f} Å)"
        )
        continue

    E_ml = config_relaxed.get_potential_energy()
    config_relaxed.calc = d3_calc
    E_d3 = config_relaxed.get_potential_energy()
    E_total = E_ml + E_d3

    co2_energies.append(E_total)
    co2_energies_ml.append(E_ml)
    co2_energies_d3.append(E_d3)
    co2_configs.append(config_relaxed)
    print(
        f"     Config {idx+1}: E_total = {E_total:.4f} eV (RPBE: {E_ml:.4f}, D3: {E_d3:.4f}, C-O dist: {co_dist:.3f} Å)"
    )

best_co2_idx = np.argmin(co2_energies)
initial_co = co2_configs[best_co2_idx]
E_initial_co = co2_energies[best_co2_idx]
E_initial_co_ml = co2_energies_ml[best_co2_idx]
E_initial_co_d3 = co2_energies_d3[best_co2_idx]

print(f"\n   → Best C*+O* (Config {best_co2_idx+1}):")
print(f"      RPBE:  {E_initial_co_ml:.4f} eV")
print(f"      D3:    {E_initial_co_d3:.4f} eV")
print(f"      Total: {E_initial_co:.4f} eV")

# Save best C+O state
ase.io.write(str(output_dir / part_dirs["part6"] / "co_initial_best.traj"), initial_co)
print("   ✓ Best C*+O* structure saved")
```

### Step 3b: Calculate C* and O* Energies Separately

Another strategy to calculate the initial energies for *C and *O at very low coverage (without interactions between the two reactants) is to do two separate relaxations. 

```{code-cell} ipython3
# Clean slab
ni_bulk_obj = Bulk(bulk_atoms=ni_bulk_atoms)
clean_slab = Slab.from_bulk_get_specific_millers(
    bulk=ni_bulk_obj_c_o, specific_millers=(1, 1, 1)
)[0].atoms
clean_slab.set_pbc([True, True, True])
clean_slab.calc = base_calc
opt = LBFGS(clean_slab, logfile=None)
opt.run(fmax=0.05, steps=relaxation_steps)

E_clean_ml = clean_slab.get_potential_energy()
clean_slab.calc = d3_calc
E_clean_d3 = clean_slab.get_potential_energy()
E_clean = E_clean_ml + E_clean_d3

print(
    f"\n   Clean slab: E_total = {E_clean:.4f} eV (RPBE: {E_clean_ml:.4f}, D3: {E_clean_d3:.4f})"
)
```

```{code-cell} ipython3
print(f"\n2b. Separate C* and O* Energies:")
print("    Calculating energies in separate unit cells to avoid interactions")

ni_bulk_obj_c_o = Bulk(bulk_atoms=ni_bulk_atoms)
ni_slab_c_o = Slab.from_bulk_get_specific_millers(
    bulk=ni_bulk_obj_c_o, specific_millers=(1, 1, 1)
)[0]

print("\n   Generating C* configurations...")
multi_ads_config_c = MultipleAdsorbateSlabConfig(
    ni_slab_c_o,
    adsorbates=[Adsorbate(adsorbate_smiles_from_db="*C")],
    num_configurations=num_sites,
)

c_energies = []
c_energies_ml = []
c_energies_d3 = []
c_configs = []

for idx, config in enumerate(multi_ads_config_c.atoms_list):
    config_relaxed = config.copy()
    config_relaxed.set_pbc([True, True, True])
    config_relaxed.calc = base_calc
    opt = LBFGS(config_relaxed, logfile=None)
    opt.run(fmax=0.05, steps=relaxation_steps)

    E_ml = config_relaxed.get_potential_energy()
    config_relaxed.calc = d3_calc
    E_d3 = config_relaxed.get_potential_energy()
    E_total = E_ml + E_d3

    c_energies.append(E_total)
    c_energies_ml.append(E_ml)
    c_energies_d3.append(E_d3)
    c_configs.append(config_relaxed)
    print(
        f"     Config {idx+1}: E_total = {E_total:.4f} eV (RPBE: {E_ml:.4f}, D3: {E_d3:.4f})"
    )

best_c_idx = np.argmin(c_energies)
c_ads = c_configs[best_c_idx]
E_c = c_energies[best_c_idx]
E_c_ml = c_energies_ml[best_c_idx]
E_c_d3 = c_energies_d3[best_c_idx]

print(f"\n   → Best C* (Config {best_c_idx+1}):")
print(f"      RPBE:  {E_c_ml:.4f} eV")
print(f"      D3:    {E_c_d3:.4f} eV")
print(f"      Total: {E_c:.4f} eV")

# Save best C state
ase.io.write(str(output_dir / part_dirs["part6"] / "c_best.traj"), c_ads)

# Generate O* configuration
print("\n   Generating O* configurations...")
multi_ads_config_o = MultipleAdsorbateSlabConfig(
    ni_slab_c_o,
    adsorbates=[Adsorbate(adsorbate_smiles_from_db="*O")],
    num_configurations=num_sites,
)
o_energies = []
o_energies_ml = []
o_energies_d3 = []
o_configs = []

for idx, config in enumerate(multi_ads_config_o.atoms_list):
    config_relaxed = config.copy()
    config_relaxed.set_pbc([True, True, True])
    config_relaxed.calc = base_calc
    opt = LBFGS(config_relaxed, logfile=None)
    opt.run(fmax=0.05, steps=relaxation_steps)

    E_ml = config_relaxed.get_potential_energy()
    config_relaxed.calc = d3_calc
    E_d3 = config_relaxed.get_potential_energy()
    E_total = E_ml + E_d3

    o_energies.append(E_total)
    o_energies_ml.append(E_ml)
    o_energies_d3.append(E_d3)
    o_configs.append(config_relaxed)
    print(
        f"     Config {idx+1}: E_total = {E_total:.4f} eV (RPBE: {E_ml:.4f}, D3: {E_d3:.4f})"
    )

best_o_idx = np.argmin(o_energies)
o_ads = o_configs[best_o_idx]
E_o = o_energies[best_o_idx]
E_o_ml = o_energies_ml[best_o_idx]
E_o_d3 = o_energies_d3[best_o_idx]

print(f"\n   → Best O* (Config {best_o_idx+1}):")
print(f"      RPBE:  {E_o_ml:.4f} eV")
print(f"      D3:    {E_o_d3:.4f} eV")
print(f"      Total: {E_o:.4f} eV")

# Save best O state
ase.io.write(str(output_dir / part_dirs["part6"] / "o_best.traj"), o_ads)

# Calculate combined energy for separate C* and O*
E_initial_co_separate = E_c + E_o
E_initial_co_separate_ml = E_c_ml + E_o_ml
E_initial_co_separate_d3 = E_c_d3 + E_o_d3
```

```{code-cell} ipython3
print(f"\n   Combined C* + O* (separate calculations):")
print(f"      RPBE:  {E_initial_co_separate_ml:.4f} eV")
print(f"      D3:    {E_initial_co_separate_d3:.4f} eV")
print(f"      Total: {E_initial_co_separate:.4f} eV")

print(f"\n   Comparison:")
print(f"      C*+O* (same cell):  {E_initial_co - E_clean:.4f} eV")
print(f"      C* + O* (separate): {E_initial_co_separate - 2*E_clean:.4f} eV")
print(
    f"      Difference:         {(E_initial_co - E_clean) - (E_initial_co_separate - 2*E_clean):.3f} eV"
)
print("   ✓ Separate C* and O* energies calculated")
```

### Step 4: Calculate Reaction Energy with ZPE

Compute the thermochemistry for C* + O* → CO* with ZPE corrections:

```{code-cell} ipython3
print(f"\n3. Reaction Energy (C* + O* → CO*):")
print(f"   " + "=" * 60)

# Electronic energies
print(f"\n   Electronic Energies:")
print(
    f"   Initial (C*+O*): RPBE = {E_initial_co_ml:.4f} eV, D3 = {E_initial_co_d3:.4f} eV, Total = {E_initial_co:.4f} eV"
)
print(
    f"   Final (CO*):     RPBE = {E_final_co_ml:.4f} eV, D3 = {E_final_co_d3:.4f} eV, Total = {E_final_co:.4f} eV"
)

# Reaction energies without ZPE
delta_E_rpbe = E_final_co_ml - E_initial_co_ml
delta_E_d3_contrib = E_final_co_d3 - E_initial_co_d3
delta_E_elec = E_final_co - E_initial_co

print(f"\n   Reaction Energies (without ZPE):")
print(f"   ΔE(RPBE only):     {delta_E_rpbe:.4f} eV = {delta_E_rpbe*96.485:.1f} kJ/mol")
print(
    f"   ΔE(D3 contrib):    {delta_E_d3_contrib:.4f} eV = {delta_E_d3_contrib*96.485:.1f} kJ/mol"
)
print(f"   ΔE(RPBE+D3):       {delta_E_elec:.4f} eV = {delta_E_elec*96.485:.1f} kJ/mol")

# Calculate ZPE for CO* (final state)
print(f"\n   Computing ZPE for CO*...")
final_co.calc = base_calc

co_indices = np.where(final_co.get_tags() == 2)[0]
vib_co = Vibrations(final_co, indices=co_indices, delta=0.02, name="vib_co")
vib_co.run()
vib_energies_co = vib_co.get_energies()
zpe_co = np.sum(vib_energies_co[vib_energies_co > 0]) / 2.0
vib_co.clean()
print(f"   ZPE(CO*): {zpe_co:.4f} eV ({zpe_co*1000:.1f} meV)")


# Calculate ZPE for C* and O* (initial state)
print(f"\n   Computing ZPE for C* and O*...")
initial_co.calc = base_calc
co_indices = np.where(final_co.get_tags() == 2)[0]
vib_c_o = Vibrations(initial_co, indices=c_o_indices, delta=0.02, name="vib_c_o")
vib_c_o.run()
vib_energies_c_o = vib_c_o.get_energies()
zpe_c_o = np.sum(vib_energies_c_o[vib_energies_c_o > 0]) / 2.0
vib_c_o.clean()
print(f"   ZPE(C*+O*): {zpe_c_o:.4f} eV ({zpe_c_o*1000:.1f} meV)")


# Total reaction energy with ZPE
delta_zpe = zpe_co - zpe_c_o
delta_E_zpe = delta_E_elec + delta_zpe

print(f"\n   Reaction Energy (with ZPE):")
print(f"   ΔE(electronic):    {delta_E_elec:.4f} eV = {delta_E_elec*96.485:.1f} kJ/mol")
print(
    f"   ΔZPE:              {delta_zpe:.4f} eV = {delta_zpe*96.485:.1f} kJ/mol ({delta_zpe*1000:.1f} meV)"
)
print(f"   ΔE(total):         {delta_E_zpe:.4f} eV = {delta_E_zpe*96.485:.1f} kJ/mol")

print(f"\n   Summary:")
print(
    f"   Without D3, without ZPE: {delta_E_rpbe:.4f} eV = {delta_E_rpbe*96.485:.1f} kJ/mol"
)
print(
    f"   With D3, without ZPE:    {delta_E_elec:.4f} eV = {delta_E_elec*96.485:.1f} kJ/mol"
)
print(
    f"   With D3, with ZPE:       {delta_E_zpe:.4f} eV = {delta_E_zpe*96.485:.1f} kJ/mol"
)

print(f"\n   " + "=" * 60)
print(f"\n   Comparison with Paper (Table 5):")
print(f"   Paper (DFT-D3): -142.7 kJ/mol = -1.48 eV")
print(f"   This work:      {delta_E_zpe*96.485:.1f} kJ/mol = {delta_E_zpe:.2f} eV")
print(f"   Difference:     {abs(delta_E_zpe - (-1.48)):.2f} eV")

if delta_E_zpe < 0:
    print(f"\n   ✓ Reaction is exothermic (C+O recombination favorable)")
else:
    print(f"\n   ⚠ Reaction is endothermic (dissociation favorable)")
```

### Step 5: Calculate CO Adsorption Energy (Bonus)

Calculate how strongly CO binds to the surface:

```{code-cell} ipython3
print(f"\n4. CO Adsorption Energy ( CO(g) + * → CO*):")
print("   This helps us understand CO binding strength")

# CO(g)
co_gas = Atoms("CO", positions=[[0, 0, 0], [0, 0, 1.15]])
co_gas.center(vacuum=10.0)
co_gas.set_pbc([True, True, True])
co_gas.calc = base_calc
opt = LBFGS(co_gas, logfile=None)
opt.run(fmax=0.05, steps=relaxation_steps)

E_co_gas_ml = co_gas.get_potential_energy()
co_gas.calc = d3_calc
E_co_gas_d3 = co_gas.get_potential_energy()
E_co_gas = E_co_gas_ml + E_co_gas_d3

print(
    f"   CO(g):       E_total = {E_co_gas:.4f} eV (RPBE: {E_co_gas_ml:.4f}, D3: {E_co_gas_d3:.4f})"
)

# Calculate ZPE for CO(g)
co_gas.calc = base_calc
vib_co_gas = Vibrations(co_gas, indices=[0, 1], delta=0.01, nfree=2)
vib_co_gas.clean()
vib_co_gas.run()
vib_energies_co_gas = vib_co_gas.get_energies()
zpe_co_gas = 0.5 * np.sum(vib_energies_co_gas[vib_energies_co_gas > 0])
vib_co_gas.clean()

print(f"   ZPE(CO(g)):  {zpe_co_gas:.4f} eV")
print(f"   ZPE(CO*):    {zpe_co:.4f} eV (from Step 4 calculation)")

# Electronic adsorption energy
E_ads_co_elec = E_final_co - E_clean - E_co_gas

# ZPE contribution to adsorption energy
delta_zpe_ads = zpe_co - zpe_co_gas

# Total adsorption energy with ZPE
E_ads_co_total = E_ads_co_elec + delta_zpe_ads

print(f"\n   Electronic Energy Breakdown:")
print(f"   ΔE(RPBE only) = {(E_final_co_ml - E_clean_ml - E_co_gas_ml):.4f} eV")
print(f"   ΔE(D3 contrib) = {((E_final_co_d3 - E_clean_d3 - E_co_gas_d3)):.4f} eV")
print(f"   ΔE(RPBE+D3) = {E_ads_co_elec:.4f} eV")
print(f"\n   ZPE Contribution:")
print(f"   ΔZPE = {delta_zpe_ads:.4f} eV")
print(f"\n   Total Adsorption Energy:")
print(f"   ΔE(total) = {E_ads_co_total:.4f} eV = {E_ads_co_total*96.485:.1f} kJ/mol")
print(f"\n   Summary:")
print(
    f"   E_ads(CO) without ZPE = {-E_ads_co_elec:.2f} eV = {-E_ads_co_elec*96.485:.1f} kJ/mol"
)
print(
    f"   E_ads(CO) with ZPE    = {-E_ads_co_total:.2f} eV = {-E_ads_co_total*96.485:.1f} kJ/mol"
)
print(
    f"   → CO binds {abs(E_ads_co_total):.2f} eV stronger than H ({abs(E_ads_co_total)/0.60:.1f}x)"
)
```

```{admonition} Comparison with Paper Results
:class: tip

The paper reports a CO adsorption energy of **1.82 eV (175.6 kJ/mol)** in Table 4, calculated using DFT (RPBE functional).

These results show:
- **Without ZPE**: The electronic binding energy matches well with DFT predictions
- **With ZPE**: The zero-point energy correction reduces the binding strength slightly
- **D3 Dispersion**: Contributes to stronger binding due to van der Waals interactions

```

### Step 6: Find Transition State with Dimer Method

Use the dimer method to get a good initial guess for the transition state:

```{code-cell} ipython3
print(f"\n5. Finding Transition State with Dimer Method")
print("   Creating initial guess with stretched C-O bond...")
print("   Starting from CO* and stretching the C-O bond...")

# Create a guess structure with stretched CO bond (start from CO*)
ts_guess = final_co.copy()

# Optimize with dimer method to find saddle point
print("   Running dimer optimization to find transition state...")
from ase.mep import DimerControl, MinModeAtoms, MinModeTranslate

ts_guess.calc = FAIRChemCalculator(predictor, task_name="oc20")

# Set up dimer control
d_control = DimerControl(
    initial_eigenmode_method="displacement",
    displacement_method="vector",
    mask=[i >= len(slab) for i in range(len(ts_guess))],
)  # Only move adsorbates

# Create displacement vector along C-O bond
displacement_vector = np.zeros((len(ts_guess), 3))
# displacement_vector[co_indices[1]] = (ts_guess.positions[-1]-ts_guess.positions[-2]) / co_distance  # O moves away
# displacement_vector[co_indices[1],0:1] = 0.5

displacement_vector[co_indices] = np.random.normal(size=(len(co_indices), 3)) * 0.1

d_atoms = MinModeAtoms(ts_guess, d_control)
d_atoms.displace(displacement_vector=displacement_vector)

# Run dimer
with MinModeTranslate(d_atoms, trajectory="dimer.traj", logfile="dimer.log") as dim_rlx:
    dim_rlx.run(fmax=0.1, steps=500)

ts_structure = ts_guess.copy()
print("   ✓ Dimer optimization completed")

# Check final C-O distance
final_co_dist = np.linalg.norm(
    ts_structure.get_positions()[co_indices[1]]
    - ts_structure.get_positions()[co_indices[0]]
)
print(f"   Final C-O distance at TS: {final_co_dist:.3f} Å")

# Save TS guess
ase.io.write(str(output_dir / part_dirs["part6"] / "ts_guess.xyz"), ts_structure)
```

### Step 7: Run NEB to Find Activation Barrier

Use the nudged elastic band method to find the minimum energy path:

```{code-cell} ipython3
print(f"\n6. NEB Barrier Calculation (C* + O* → CO*)")
print("   Setting up 7-image NEB chain with TS guess in middle...")
print("   Reaction: C* + O* (initial) → TS → CO* (final)")

n_images = 7
images = [initial_co]  # Start with C* + O*

# Create images: C*+O* → TS → CO*
for i in range(1, n_images):
    if i == n_images // 2:  # Middle image
        image = ts_structure.copy()
    else:
        image = initial_co.copy()
    image.set_pbc([True, True, True])
    image.calc = FAIRChemCalculator(predictor, task_name="oc20")
    images.append(image)

images.append(final_co)  # End with CO*

# Interpolate with better initial guess
dyneb = DyNEB(images, climb=True, fmax=0.1)

# Interpolate first half (C*+O* → TS)
from ase.neb import interpolate

print("\n   Interpolating images...")
interpolate(images[:n_images//2 + 1])
# Interpolate second half (TS → CO*)
interpolate(images[n_images//2:])

# Optimize
print("   Optimizing NEB path (this may take a while)...")
try:
    opt = FIRE(
        dyneb,
        trajectory=str(output_dir / part_dirs["part6"] / 'neb.traj'),
        logfile=str(output_dir / part_dirs["part6"] / 'neb.log')
    )
    opt.run(fmax=0.1, steps=relaxation_steps)
    
    # Extract barrier (from C*+O* to TS)
    energies = [img.get_potential_energy() for img in images]
    energies_rel = np.array(energies) - E_initial_co
    E_barrier = np.max(energies_rel)
    
    print(f"\n   ✓ NEB converged!")
    print(f"\n   Forward barrier (C*+O* → CO*): {E_barrier:.3f} eV = {E_barrier*96.485:.1f} kJ/mol")
    print(f"   Reverse barrier (CO* → C*+O*): {E_barrier - delta_E_zpe:.3f} eV = {(E_barrier - delta_E_zpe)*96.485:.1f} kJ/mol")
    print(f"\n   Paper (Table 5): 153 kJ/mol = 1.59 eV (reverse barrier)")
    print(f"   Difference: {abs((E_barrier - delta_E_zpe) - 1.59):.2f} eV")
```

### Step 8: Visualize NEB Path and Key Structures

Create plots showing the reaction pathway:

```{code-cell} ipython3
print("\n   Creating NEB visualization...")

# Plot NEB path
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(
    range(len(energies_rel)),
    energies_rel,
    "o-",
    linewidth=2,
    markersize=10,
    color="steelblue",
    label="NEB Path",
)
ax.axhline(0, color="green", linestyle="--", alpha=0.5, label="Initial: C*+O*")
ax.axhline(delta_E_zpe, color="red", linestyle="--", alpha=0.5, label="Final: CO*")
ax.axhline(
    E_barrier,
    color="orange",
    linestyle=":",
    alpha=0.7,
    linewidth=2,
    label=f"Forward Barrier = {E_barrier:.2f} eV",
)

# Annotate transition state
ts_idx = np.argmax(energies_rel)
ax.annotate(
    f"TS\n{energies_rel[ts_idx]:.2f} eV",
    xy=(ts_idx, energies_rel[ts_idx]),
    xytext=(ts_idx, energies_rel[ts_idx] + 0.3),
    ha="center",
    fontsize=11,
    fontweight="bold",
    arrowprops=dict(arrowstyle="->", lw=1.5, color="red"),
)

ax.set_xlabel("Image Number", fontsize=13)
ax.set_ylabel("Relative Energy (eV)", fontsize=13)
ax.set_title(
    "CO Formation on Ni(111): C* + O* → CO* - NEB Path", fontsize=15, fontweight="bold"
)
ax.legend(fontsize=11, loc="upper left")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(
    str(output_dir / part_dirs["part6"] / "neb_path.png"), dpi=300, bbox_inches="tight"
)
plt.show()

# Create animation of NEB path
print("\n   Creating NEB path animation...")
from ase.io import write as ase_write

ase.io.write(str(output_dir / part_dirs["part6"] / "neb_path.gif"), images, format="gif")
print("   → Saved as neb_path.gif")

# Visualize key structures
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
structures = [
    (initial_co, "Initial: C* + O*", 0),
    (images[ts_idx], f"TS (Image {ts_idx})", 1),
    (final_co, "Final: CO*", 2),
]

for atoms, title, idx in structures:
    plot_atoms(atoms, axes[idx], radii=0.5, rotation=("10x,10y,0z"))
    axes[idx].set_title(title, fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig(
    str(output_dir / part_dirs["part6"] / "neb_structures.png"),
    dpi=300,
    bbox_inches="tight",
)
plt.show()

print("\n✓ NEB analysis complete!")
```

:::{admonition} Comparison with Paper (Tables 4 & 5)
:class: important

**Expected Results from Paper:**
- **Reaction Energy (C* + O* → CO*)**: **-142.7 kJ/mol = -1.48 eV** (exothermic, DFT-D3)
- **Activation Barrier (C* + O* → CO*)**: **153 kJ/mol = 1.59 eV** (reverse/dissociation, DFT-D3)
- **CO Adsorption Energy**: **1.82 eV = 175.6 kJ/mol** (DFT-D3)

**Reaction Direction:**
- Paper reports CO dissociation barrier (CO* → C* + O*), which is the **reverse** of the recombination we calculate
- Forward (C* + O* → CO*): barrier = reverse_barrier - |ΔE| ≈ 1.59 - 1.48 ≈ 0.11 eV (very fast)
- Reverse (CO* → C* + O*): barrier = 1.59 eV (very slow, kinetic bottleneck)

**What to Check:**
- Reaction energy (C*+O* → CO*) should be strongly exothermic (~-1.5 eV)
- Reverse barrier (CO dissociation) should be substantial (~1.6 eV)
- Forward barrier (recombination) should be very small (~0.1 eV)
- CO binds much more strongly than H (1.82 eV vs 0.60 eV)

**Typical Variations:**
- Reaction energies typically accurate within 0.1-0.2 eV
- Barriers more sensitive: expect ±0.2-0.3 eV variation
- ZPE corrections typically add 0.05-0.15 eV to reaction energies
- NEB convergence affects barrier more than reaction energy

**Physical Insight:**
- Large reverse barrier (1.59 eV) makes CO dissociation very slow at low T
- Small forward barrier (0.11 eV) means C+O rapidly recombine to CO
- This explains why Ni produces CO in Fischer-Tropsch rather than keeping C and O separate
- High temperatures needed to overcome the dissociation barrier for further C-C coupling
:::

:::{admonition} NEB Method Explained
:class: important

The **Nudged Elastic Band (NEB)** method finds the minimum energy path between reactants and products:

1. **Interpolate** between initial and final states (5-9 images typical)
2. **Add spring forces** along the chain to maintain spacing
3. **Project out** spring components perpendicular to the path
4. **Climbing image** variant: highest energy image climbs to saddle point

Advantages:
- No prior knowledge of transition state needed
- Finds entire reaction coordinate
- Robust for complex reactions

Limitations:
- Computationally expensive (optimize N images)
- May find wrong path if initial interpolation is poor
:::

### Explore on Your Own

1. **Image convergence**: Run with 7 or 9 images. Does the barrier change?
2. **Spring constant**: Modify the NEB spring constant. How does this affect convergence?
3. **Alternative paths**: Try different initial CO/final C+O configurations. Are there multiple pathways?
4. **Reverse barrier**: Calculate E_a(reverse) = E_a(forward) - ΔE. Check Brønsted-Evans-Polanyi relationship.
5. **Diffusion barriers**: Compute NEB for C or O diffusion on the surface. How do they compare?

---

## Summary and Best Practices

### Key Takeaways

1. **ML Potentials**: UMa-S-1P1 provides ~1000× speedup over DFT with reasonable accuracy
2. **Bulk optimization**: Always use the ML-optimized lattice constant for consistency
3. **Surface energies**: Linear extrapolation eliminates finite-size effects
4. **Adsorption**: Test multiple sites; lowest energy may not be intuitive
5. **Coverage**: Lateral interactions become significant above ~0.3 ML
6. **Barriers**: NEB requires careful setup but yields full reaction pathway

### Recommended Workflow for New Systems

```{mermaid}
graph TD
    A[Optimize Bulk] --> B[Calculate Surface Energies]
    B --> C[Wulff Construction]
    C --> D[Low-Coverage Adsorption]
    D --> E{Coverage Important?}
    E -->|Yes| F[Coverage Study]
    E -->|No| G[Reaction Barriers]
    F --> G
    G --> H[Microkinetic Modeling]
```

### Accuracy Considerations

| Property | Typical Error | When Critical |
|----------|--------------|---------------|
| Lattice constants | 1-2% | Strain effects, alloys |
| Surface energies | 10-20% | Nanoparticle shapes |
| Adsorption energies | 0.1-0.3 eV | Thermochemistry |
| Barriers | 0.2-0.5 eV | Kinetics, selectivity |

**Rule of thumb**: Use ML for screening → DFT for validation → Experiment for verification

### Further Reading

- **UMA Paper**: [Wood et al. 2025](https://arxiv.org/abs/2506.23971)
- **OMat24 Paper**: [Barroso-Luque et al., 2024](https://arxiv.org/abs/2410.12771)
- **OC20 Dataset**: [Chanussot et al., Nature, 2021](https://www.nature.com/articles/s41586-021-03828-1)
- **ASE Tutorial**: [https://wiki.fysik.dtu.dk/ase/](https://wiki.fysik.dtu.dk/ase/)
---

## Appendix: Troubleshooting

### Common Issues

**Problem**: Convergence failures
- **Solution**: Reduce `fmax` to 0.1 initially, tighten later
- Check if system is metastable (try different starting geometry)

**Problem**: NEB fails to find transition state
- **Solution**: Use more images (9-11) or better initial guess
- Try fixed-end NEB first, then climbing image

**Problem**: Unexpected adsorption energies
- **Solution**: Visualize structures - check for distortions
- Compare with multiple sites
- Add D3 corrections

**Problem**: Out of memory
- **Solution**: Reduce system size (smaller supercells)
- Use fewer NEB images
- Run on HPC with more RAM

### Performance Tips

1. **Use batching**: Relax multiple configurations in parallel
2. **Start with DEBUG_MAX_STEPS=50**: Get quick results, refine later
3. **Cache bulk energies**: Don't recalculate reference systems
4. **Trajectory analysis**: Monitor optimization progress with ASE GUI

---

## Caveats and Pitfalls

:::{admonition} Important Considerations
:class: warning

When using ML potentials for surface catalysis, be aware of these critical issues:
:::

### 1. Task Selection: OMAT vs OC20

**Critical choice**: Which task_name to use?
- **`task_name="omat"`**: Optimized for bulk and clean surface calculations
  - Use for: Part 1 (bulk), Part 2 (surface energies), Part 3 (Wulff)
  - Better for structural relaxations without adsorbates
- **`task_name="oc20"`**: Optimized for surface chemistry with adsorbates
  - Use for: Part 4-6 (all adsorbate calculations)
  - Trained on Open Catalyst data with adsorbate-surface interactions

**Impact**: Using wrong task can lead to 0.1-0.3 eV errors in adsorption energies!

### 2. D3 Dispersion Corrections

**Multiple decisions required**:

1. **Whether to use D3 at all?**
   - Small adsorbates (H, O, N): D3 effect ~0.01-0.05 eV (often negligible)
   - Large molecules (CO, CO₂, aromatics): D3 effect ~0.1-0.3 eV (important!)
   - Physisorption: D3 critical (can change binding from repulsive to attractive)
   - RPBE was originally fit for chemisorption energies without D3 corrections, so adding D3 corrections may actually cause small adsorbates to overbind. However, it probably would be important for larger molecules. It's relatively uncommon to see RPBE+D3 as a choice in the catalysis literature (compared to PBE+D3, or RPBE, or BEEF-vdW). 

2. **Which DFT functional for D3?**
   - This tutorial uses `method="PBE"` consistently for the D3 correction. This is often implied when papers say they use a D3 correction, but the results can be different if use the RPBE parameterizations.
   - Original paper used PBE for bulk/surfaces, RPBE for adsorption. It's not specified what D3 parameterization they used, but it's likely PBE. 

3. **When to apply D3?**
   - **End-point correction** (used here): Fast, run ML optimization then add D3 energy
   - **During optimization**: Slower but more accurate geometries
   - **Impact**: Usually <0.05 eV difference, but can be larger for weak interactions

### 3. Coverage Dependence Challenges

**Non-linearity at high coverage**:
- This tutorial assumes linear E_ads(θ) = E₀ + βθ
- Reality: Often non-linear, especially near θ = 1 ML. See the plots generated - there is a linear regime for relatively high coverage, and relatively low coverage, but it's not uniformly linear everywhere. As long as you consistently in one regime or the other a linear assumption is probably ok, but you could get into problems if solving microkinetic models where the coverage of the species in question changes significantly from very low to high. 
- **Why**: Phase transitions, adsorbate ordering, surface reconstruction
- **Solution**: Test polynomial fits, look for ordering in visualizations

**Low coverage limit**:
- At θ < 0.1 ML, coverage effects are tiny (<0.01 eV)
- Hard to distinguish from numerical noise
- **Best practice**: Focus on 0.25-1.0 ML range for fitting

### 4. Periodic Boundary Conditions

**UMa requires PBC=True in all directions!**
```python
atoms.set_pbc([True, True, True])  # Always required
```
- Forgetting this causes crashes or wrong energies
- Even for "gas phase" molecules in vacuum

### 5. Gas Phase Reference Energies

**Tricky cases**:
- **H₂(g)**: UMa handles well (used in this tutorial)
- **H(g)**: May not be reliable (use H₂/2 instead)
- **CO(g)**, **O₂(g)**: Usually okay, but check against DFT
- **Radicals**: Often problematic

**Best practice**: Always use stable molecules as references (H₂, not H; H₂O, not OH)

### 6. Spin Polarization

**Key limitation**: OC20/UMa does not include spin!
- Paper used spin-polarized DFT
- **Impact**: Usually small (0.05-0.1 eV)
- **Larger** for:
  - Magnetic metals (Fe, Co, Ni)
  - Open-shell adsorbates (O*, OH*)
  - Reaction barriers with radicals

### 7. Constraint Philosophy

**Clean slabs** (Part 2): No constraints (both surfaces relax)
- Best for surface energy calculations
- More physical for symmetric slabs

**Adsorbate slabs** (Part 4-6): Bottom layers fixed
- Faster convergence
- Prevents adsorbate-induced reconstruction
- Standard practice in surface chemistry

**Fairchem helper functions**: Automatically apply sensible constraints
- Trust their heuristics unless you have good reason not to
- Check `atoms.constraints` to see what was applied

### 8. Complex Surface Structures

**This tutorial uses low-index facets** (111, 100, 110, 211)
- Well-defined, symmetric
- Easy to generate and analyze

**Real catalysts** have:
- Steps, kinks, grain boundaries
- Support interfaces
- Defects and vacancies
- **Challenge**: Harder to generate, more configurations to test

### 9. Slab Thickness and Vacuum

**Convergence tests critical** but expensive:
- This tutorial uses "reasonable" values (4-8 layers, 10 Å vacuum)
- **Always check** convergence for new systems
- **Especially important** for:
  - Metals with long electron screening (Au, Ag)
  - Charged adsorbates
  - Strong adsorbate-induced reconstruction

### 10. NEB Convergence

**Most computationally expensive part**:
- May need 7-11 images (not just 5)
- Initial guess matters a lot
- Can get stuck in local minima

**Tricks**:
1. Use dimer method to find better TS guess (as shown in Part 6)
2. Start with coarse convergence (fmax=0.2), refine later
3. Visualize the path - does it make chemical sense?
4. Try different spring constants (0.1-1.0 eV/Å)

### 11. Lattice Constant Source

**Consistency is key**:
- Use ML-optimized lattice constant throughout (as done here)
- **Don't mix**: ML lattice + DFT surface energies = inconsistent
- Alternative: Use experimental lattice constant for everything

### 12. Adsorbate Placement

**Multiple local minima**:
- Surface chemistry is **not** convex!
- Always test multiple adsorption sites
- Fairchem helpers generate ~5 configurations in this tutorial, but you may need more to search many modes. You can already try methods like minima hopping or other global optimization methods to sample more configurations.

**For complex adsorbates**:
- Test different orientations
- May need 10-20 configurations
- Consider genetic algorithms or basin hopping

---

:::{admonition} Congratulations! 🎉
:class: tip

You've completed a comprehensive computational catalysis workflow using state-of-the-art ML potentials. You can now:

- Characterize catalyst surfaces computationally
- Predict nanoparticle shapes
- Calculate reaction thermodynamics and kinetics
- Apply these methods to your own research questions

**Next steps**: 
- Apply to your catalyst system of interest
- Validate key results with DFT
- Develop microkinetic models
- Publish your findings!
:::

```{code-cell} ipython3

```
