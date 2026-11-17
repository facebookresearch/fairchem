# Open Catalyst Project

## Overview

The Open Catalyst Project was a collaborative research effort between
Fundamental AI Research (FAIR) at Meta and Carnegie Mellon University's (CMU)
Department of Chemical Engineering. The aim was to use AI to model and discover
new catalysts for use in renewable energy storage to help in addressing climate
change.

Scalable and cost-effective solutions to renewable energy storage are essential
to addressing the world's rising energy needs while reducing climate change. As
we increase our reliance on renewable energy sources such as wind and solar,
which produce intermittent power, storage is needed to transfer power from times
of peak generation to peak demand. This may require the storage of power for
hours, days, or months. One solution that offers the potential of scaling to
nation-sized grids is the conversion of renewable energy to other fuels, such as
hydrogen. To be widely adopted, this process requires cost-effective solutions to
running chemical reactions.

An open challenge is finding low-cost catalysts to drive these reactions at high
rates. Through the use of quantum mechanical simulations (density functional
theory), new catalyst structures can be tested and evaluated. Unfortunately, the
high computational cost of these simulations limits the number of structures
that may be tested. The use of AI or machine learning may provide a method to
efficiently approximate these calculations, leading to new approaches in finding
effective catalysts.

To enable the broader research community to participate in the project, we
released the [Open Catalyst 2020 (OC20)](datasets/oc20.md) and
[Open Catalyst 2022 (OC22)](datasets/oc22.md) datasets for training ML models.
Together, these releases contained 1.3 million molecular relaxations with results
from over 260 million DFT calculations. In addition to the data, baseline models
and code were open-sourced on
[our GitHub page](https://github.com/facebookresearch/fairchem). Results are
available on the
[leaderboard](https://huggingface.co/spaces/facebook/fairchem_leaderboard).

## Research papers and resources

Our research papers described the baselines, evaluation metrics, and datasets:

- [Overview Paper](https://arxiv.org/abs/2010.09435)
- [OC20 Dataset Paper](https://arxiv.org/abs/2010.09990)
- [OC22 Dataset Paper](https://arxiv.org/abs/2206.08917)

The datasets and benchmark results are available through the following resources:

- [Download Datasets](datasets/summary.md)
- [View Leaderboard](https://huggingface.co/spaces/facebook/fairchem_leaderboard)
