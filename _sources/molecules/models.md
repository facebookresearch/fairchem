# Pretrained models

**2025 recommendation:** We suggest using the [UMA model](../core/uma), trained on all of the FAIR chemistry datasets before using one of the checkpoints below. The UMA model has a number of nice features over the previous checkpoints
1. It is state-of-the-art in out-of-domain prediction accuracy
2. The UMA small model is an energy conserving and smooth checkpoint, so should work much better for vibrational calculations, molecular dynamics, etc.
3. The UMA model is most likely to be updated in the future.

## Baseline models in the OMol25 paper
As part of the OMol25 release, we released two sets of models:
1. [preferred] UMA models trained on a range of FAIR chemistry datasets, available at [HuggingFace](https://huggingface.co/facebook/UMA)
2. eSEN models trained only on OMol25, available at [HuggingFace](https://huggingface.co/facebook/OMol25/tree/main)

The UMA models will continue to be updated regularly and we expect those to remain the default and performant option for the forseeable future. The OMol25-only eSEN models are provided mostly as a base-line for models trained only on OMol25.

## Citing

If you use the OMol25-trained eSEN models, please cite the following paper.

```bib
@misc{levine2025openmolecules2025omol25,
      title={The Open Molecules 2025 (OMol25) Dataset, Evaluations, and Models},
      author={Daniel S. Levine and Muhammed Shuaibi and Evan Walter Clark Spotte-Smith and Michael G. Taylor and Muhammad R. Hasyim and Kyle Michel and Ilyes Batatia and Gábor Csányi and Misko Dzamba and Peter Eastman and Nathan C. Frey and Xiang Fu and Vahe Gharakhanyan and Aditi S. Krishnapriyan and Joshua A. Rackers and Sanjeev Raja and Ammar Rizvi and Andrew S. Rosen and Zachary Ulissi and Santiago Vargas and C. Lawrence Zitnick and Samuel M. Blau and Brandon M. Wood},
      year={2025},
      eprint={2505.08762},
      archivePrefix={arXiv},
      primaryClass={physics.chem-ph},
      url={https://arxiv.org/abs/2505.08762},
}
```

## Baseline models in the OMC25 paper
As part of the OMC25 release, we released eSEN model trained only on OMC25, available at [HuggingFace](https://huggingface.co/facebook/OMC25). [preferred] UMA models trained on a range of FAIR chemistry datasets are available at [HuggingFace](https://huggingface.co/facebook/UMA).

## Citing

We encourage users to cite this paper when using the OMC25 dataset or pretrained models for molecular crystals in their research.

```bibtex
@misc{gharakhanyan2025openmolecularcrystals2025omc25dataset,
      title={Open Molecular Crystals 2025 (OMC25) Dataset and Models},
      author={Vahe Gharakhanyan and Luis Barroso-Luque and Yi Yang and Muhammed Shuaibi and Kyle Michel and Daniel S. Levine and Misko Dzamba and Xiang Fu and Meng Gao and Xingyu Liu and Haoran Ni and Keian Noori and Brandon M. Wood and Matt Uyttendaele and Arman Boromand and C. Lawrence Zitnick and Noa Marom and Zachary W. Ulissi and Anuroop Sriram},
      year={2025},
      eprint={2508.02651},
      archivePrefix={arXiv},
      primaryClass={physics.chem-ph},
      url={https://arxiv.org/abs/2508.02651},
}
```

## License

All models require users to agree to the FAIR Chemistry License as part of the HuggingFace model gating process.
