---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.1
kernelspec:
  name: python3
  language: python
  display_name: Python 3 (ipykernel)
---

Adsorption Energies
======================================================

Pre-trained ODAC models are versatile across various MOF-related tasks. To begin, we'll start with a fundamental application: calculating the adsorption energy for a single CO<sub>2</sub> molecule. This serves as an excellent and simple demonstration of what you can achieve with these datasets and models.

For predicting the adsorption energy of a single CO<sub>2</sub> molecule within a MOF structure, the adsorption energy ($E_{\mathrm{ads}}$) is defined as:

$$ E_{\mathrm{ads}} = E_{\mathrm{MOF+CO2}} - E_{\mathrm{MOF}} - E_{\mathrm{CO2}} \tag{1}$$

Each term on the right-hand side represents the energy of the relaxed state of the indicated chemical system. For a comprehensive understanding of our methodology for computing these adsorption energies, please refer to our [paper](https://doi.org/10.1021/acscentsci.3c01629).

## Loading Pre-trained Models

To leverage the ODAC pre-trained models, ensure you have fairchem version 2 installed; more details are available [here](../../core/fairchemv1_v2.html). You can install the required version using pip if you haven't already:

```{code-cell}
:tags: [skip-execution]

pip install fairchem-core
```

Once installed, a pre-trained model can be loaded using `FAIRChemCalculator`. In this example, we'll employ UMA to determine the CO<sub>2</sub> adsorption energies.

```{code-cell}
from fairchem.core import pretrained_mlip, FAIRChemCalculator

predictor = pretrained_mlip.get_predict_unit("uma-s-1", device = "cpu")
calc = FAIRChemCalculator(predictor, task_name = "odac")
```

## CO<sub>2</sub> Adsorption Energy in Mg-MOF-74

Let's apply our knowledge to Mg-MOF-74, a widely studied MOF known for its excellent CO<sub>2</sub> adsorption properties. Its structure comprises magnesium atomic complexes connected by a carboxylated and oxidized benzene ring, serving as an organic linker. Previous studies consistently report the CO<sub>2</sub> adsorption energy for Mg-MOF-74 to be around -0.40 eV [[1]](https://doi.org/10.1039/C4SC02064B) [[2]](https://doi.org/10.1039/C3SC51319J) [[3]](https://doi.org/10.1021/acs.jpcc.8b00938).

Our goal is to verify if we can achieve a similar value by performing a simple single-point calculation using UMA. In the ODAC23 dataset, all MOF structures are identified by their CSD (Cambridge Structural Database) code. For Mg-MOF-74, this code is **OPAGIX**. We've extracted a specific `OPAGIX+CO2` configuration from the dataset, which exhibits the lowest adsorption energy among its counterparts.

```{code-cell}
import matplotlib.pyplot as plt

from ase.io import read
from ase.visualize.plot import plot_atoms

mof_co2 = read('OPAGIX_w_CO2.cif')
mof = read('OPAGIX.cif')
co2 = read('co2.xyz')

fig, ax = plt.subplots(figsize = (5, 4.5), dpi = 250)
plot_atoms(mof_co2, ax)
ax.set_axis_off()
```

+++

The final step in calculating the adsorption energy involves connecting the `FAIRChemCalculator` to each relaxed structure: `OPAGIX+CO2`, `OPAGIX`, and `CO2`. The structures used here are already relaxed from ODAC23. For simplicity, we assume here that further relaxations can be neglected. We will show how to go beyond this assumption in the next section.

```{code-cell}
mof_co2.calc = calc
mof.calc = calc
co2.calc = calc

E_ads = mof_co2.get_potential_energy() - mof.get_potential_energy() - co2.get_potential_energy()

print(f'Adsorption energy of CO2 in Mg-MOF-74: {E_ads:.3f} eV')
```