#!/bin/bash
pip install -e packages/fairchem-core[docs,adsorbml,quacc] -e packages/fairchem-data-oc[dev] -e packages/fairchem-applications-cattsunami jupytext

# Convert all .md docs to ipynb for easy viewing in vscode later!
find ./docs -name '*.md' -exec jupytext --to ipynb {} \;
