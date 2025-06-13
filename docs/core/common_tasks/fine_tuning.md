# Fine-tuning

This repo provides a number of methods to quickly fine-tune a model using a custom ASE LMDB dataset. Training and Fine-tuning in the fairchem repo uses the fairchem cli tool which uses (Hydra yaml)[https://hydra.cc/] configs. Training dataset must be in the ASE-lmdb format. For UMA models, we provide a simple script to help generate ASE-lmdb datasets from a variety of input formats as such (cifs, traj, extxyz etc) as well as a finetuning yaml config that can be directly used for finetuning.

## Generating training/fine-tuning datasets
