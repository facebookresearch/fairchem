# Leaderboard

:::{tip} Community Leaderboard
Submit your model predictions for evaluation on the [fairchem_leaderboard](https://huggingface.co/spaces/facebook/fairchem_leaderboard) hosted on HuggingFace.
:::

The FAIR Chemistry Leaderboard provides a centralized platform for evaluating machine learning interatomic potentials (MLIPs) across different chemical domains. Researchers can submit predictions and compare their models against existing benchmarks.

The leaderboard currently supports the following datasets:

- **[OMol25](omol.md)**: Organic molecules - S2EF evaluation and downstream chemistry tasks (ligand pocket, conformers, spin gap, etc.)
- **[OC20](oc20.md)**: Heterogeneous catalysis - S2EF and IS2RE evaluation across in-domain and out-of-domain test splits.

## How to Submit

1. Generate prediction files for the appropriate task (see dataset-specific pages for details).
2. Go to the [fairchem_leaderboard](https://huggingface.co/spaces/facebook/fairchem_leaderboard) on HuggingFace.
3. Sign in with your HuggingFace account.
4. Fill in the submission metadata (model name, organization, contact info, etc.).
5. Select the evaluation type that matches your prediction file.
6. Upload your file and click Submit.
7. Wait for the evaluation to complete and see the success message.

:::{warning}
Submission limits: Users are limited to **5 successful submissions per month** for each evaluation type.
:::

### Need Help?
- [GitHub Issues](https://github.com/facebookresearch/fairchem)
- [Discussion Forum](https://huggingface.co/spaces/facebook/fairchem_leaderboard/discussions)
