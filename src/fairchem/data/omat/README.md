# OMat24 Dataset

> [!CAUTION]
> OMat24 DFT calculations are not compatible with Materials Project calculations. If you are using models trained on OMat24 only
> for such calculations, you can find a OMat24 specific calculations of reference unary compounds and MP2020-style
> anion and GGA/GGA+U mixing corrections in the [OMat24 Hugging Face repo](https://huggingface.co/datasets/facebook/OMAT24).
> Do not use MP2020 corrections or use the MP references compounds when using OMat24 trained models of OMat24 DFT calculations.
> Additional care must be taken when computing energy differences, such as formation and energy above hull and comparing with
> calculations in the Materials Project since DFT pseudopotentials are different and magnetic ground states may differ as well.

The OMat24 dataset is available for download from [this](https://huggingface.co/datasets/facebook/OMAT24) HuggingFace repo.

Pretrained eqV2 and eSEN models can be downloaded from HuggingFace [here](https://huggingface.co/facebook/OMAT24) and
UMA models [here](https://huggingface.co/facebook/UMA).

The VASP sets used to generate OMat24 data are implemented as `pymatgen` `VaspInputSets`. You can
generate OMat24 VASP inputs as follows,

```bash
:tags: [skip-execution]

pip install fairchem-data-omat
```

```python
from pymatgen.core import Structure, Lattice
from fairchem.data.omat.vasp.sets import OMat24StaticSet

lattice = Lattice.cubic(3.615)

structure = Structure.from_spacegroup(
    "Fm-3m", species=["Cu"], coords=[[0, 0, 0]], lattice=lattice
)

input_set = OMat24StaticSet(structure)
input_set.write_input("path/to/input-dir")
```

## Citing

If you use the OMat24 dataset or pretrained models in your work, consider citing the following,

```bibtex
@article{barroso_omat24,
  title={Open Materials 2024 (OMat24) Inorganic Materials Dataset and Models},
  author={Barroso-Luque, Luis and Muhammed, Shuaibi and Fu, Xiang and Wood, Brandon, Dzamba, Misko, and Gao, Meng and Rizvi, Ammar and  Zitnick, C. Lawrence and Ulissi, Zachary W.},
  journal={arXiv preprint arXiv:2410.12771},
  year={2024}
}
@article{schmidt_2023_machine,
  title={Machine-Learning-Assisted Determination of the Global Zero-Temperature Phase Diagram of Materials},
  author={Schmidt, Jonathan and Hoffmann, Noah and Wang, Hai-Chen and Borlido, Pedro and Carri{\c{c}}o, Pedro JMA and Cerqueira, Tiago FT and Botti, Silvana and Marques, Miguel AL},
  journal={Advanced Materials},
  volume={35},
  number={22},
  pages={2210788},
  year={2023},
  url={https://onlinelibrary.wiley.com/doi/full/10.1002/adma.202210788},
  publisher={Wiley Online Library}
}
```
