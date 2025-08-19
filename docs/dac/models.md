
# Pretrained models

## ODAC25 Models

As part of the ODAC25 release, we released two sets of models:
1. UMA models trained on a range of FAIR chemistry datasets including a subset of ODAC25, available at the [UMA HuggingFace site](https://huggingface.co/facebook/UMA)
2. eSEN models trained only on ODAC25, available at the [ODAC25 HuggingFace site](https://huggingface.co/facebook/ODAC25)

The UMA models were only trained on CO₂ and H₂O adsorbates, and are competitive with
the eSEN models for these adsorbates. Since the UMA models were not trained on 
N₂ and O₂, we strongly recommend using the eSEN models for these.

If you use the ODAC25 trained models, please cite the following paper:

```bib
@misc{sriram2025odac25,
    title={The Open DAC 2025 Dataset for Sorbent Discovery in Direct Air Capture}, 
    author={Anuroop Sriram and Logan M. Brabson and Xiaohan Yu and Sihoon Choi and Kareem Abdelmaqsoud and Elias Moubarak and Pim de Haan and Sindy Löwe and Johann Brehmer and John R. Kitchin and Max Welling and C. Lawrence Zitnick and Zachary Ulissi and Andrew J. Medford and David S. Sholl},
    year={2025},
    eprint={},
    archivePrefix={arXiv},
    primaryClass={},
    url={}, 
}
```

## ODAC23 Models [Deprecated]
* All config files for the ODAC23 models are available in the [`configs/odac`](https://github.com/facebookresearch/fairchem/tree/main/configs/odac) directory.

### S2EF models

| Model Name                   |Model	|Checkpoint	| Config |
|------------------------------|---	|---	|---	|
| SchNet-S2EF-ODAC             | SchNet               | [checkpoint](https://dl.fbaipublicfiles.com/dac/checkpoints_20231018/Schnet.pt) | [config](https://github.com/facebookresearch/fairchem/tree/main/configs/odac/s2ef/schnet.yml) |
| DimeNet++-S2EF-ODAC          | DimeNet++           | [checkpoint](https://dl.fbaipublicfiles.com/dac/checkpoints_20231018/DimenetPP.pt) | [config](https://github.com/facebookresearch/fairchem/tree/main/configs/odac/s2ef/dpp.yml) |
| PaiNN-S2EF-ODAC              | PaiNN               | [checkpoint](https://dl.fbaipublicfiles.com/dac/checkpoints_20231018/PaiNN.pt) | [config](https://github.com/facebookresearch/fairchem/tree/main/configs/odac/s2ef/painn.yml) |
| GemNet-OC-S2EF-ODAC          | GemNet-OC           | [checkpoint](https://dl.fbaipublicfiles.com/dac/checkpoints_20231018/Gemnet-OC.pt) | [config](https://github.com/facebookresearch/fairchem/tree/main/configs/odac/s2ef/gemnet-oc.yml) |
| eSCN-S2EF-ODAC               | eSCN                | [checkpoint](https://dl.fbaipublicfiles.com/dac/checkpoints_20231018/eSCN.pt) | [config](https://github.com/facebookresearch/fairchem/tree/main/configs/odac/s2ef/eSCN.yml) |
| EquiformerV2-S2EF-ODAC       | EquiformerV2        | [checkpoint](https://dl.fbaipublicfiles.com/dac/checkpoints_20231116/eqv2_31M.pt) | [config](https://github.com/facebookresearch/fairchem/tree/main/configs/odac/s2ef/eqv2_31M.yml) |
| EquiformerV2-Large-S2EF-ODAC | EquiformerV2 (Large) | [checkpoint](https://dl.fbaipublicfiles.com/dac/checkpoints_20231116/Equiformer_V2_Large.pt) | [config](https://github.com/facebookresearch/fairchem/tree/main/configs/odac/s2ef/eqv2_153M.yml) |

### IS2RE Direct models

| Model Name              | Model        |Checkpoint	| Config |
|-------------------------|--------------|---	| --- |
| Gemnet-OC-IS2RE-ODAC    | Gemnet-OC    | [checkpoint](https://dl.fbaipublicfiles.com/dac/checkpoints_20231018/Gemnet-OC_Direct.pt) | [config](https://github.com/facebookresearch/fairchem/tree/main/configs/odac/is2re/gemnet-oc.yml) |
| eSCN-IS2RE-ODAC         | eSCN         | [checkpoint](https://dl.fbaipublicfiles.com/dac/checkpoints_20231018/eSCN_Direct.pt) | [config](https://github.com/facebookresearch/fairchem/tree/main/configs/odac/is2re/eSCN.yml) |
| EquiformerV2-IS2RE-ODAC | EquiformerV2 | [checkpoint](https://dl.fbaipublicfiles.com/dac/checkpoints_20231116/Equiformer_V2_Direct.pt) | [config](https://github.com/facebookresearch/fairchem/tree/main/configs/odac/is2re/eqv2_31M.yml) |

The models in the table above were trained to predict relaxed energy directly. Relaxed energies can also be predicted by running structural relaxations using the S2EF models from the previous section.

### IS2RS

The IS2RS is solved by running structural relaxations using the S2EF models from the prior section.

The Open DAC 2023 (ODAC23) dataset is licensed under a [Creative Commons Attribution 4.0 License](https://creativecommons.org/licenses/by/4.0/legalcode).

Please consider citing the following paper in any research manuscript using the ODAC23 dataset:

```bibtex
@article{odac23_dataset,
    author = {Anuroop Sriram and Sihoon Choi and Xiaohan Yu and Logan M. Brabson and Abhishek Das and Zachary Ulissi and Matt Uyttendaele and Andrew J. Medford and David S. Sholl},
    title = {The Open DAC 2023 Dataset and Challenges for Sorbent Discovery in Direct Air Capture},
    year = {2023},
    journal={arXiv preprint arXiv:2311.00341},
}
```
