#!/usr/bin/env bash
# Launch a fairchem job on the dm1 / fair_amaia_cw (CoreWeave H100) cluster with
# LIVE (online) wandb logging to the internal server.
#
# Why this wrapper: the login/compute env sets http_proxy=10.0.2.2:... (a proxy
# for PUBLIC internet that compute nodes can't reach). wandb then routes to the
# internal meta-fair.wandb.io THROUGH that proxy and wandb.init times out. Fixing
# it needs the proxy cleared for the wandb host in the job env, and WANDB_MODE
# left at its online default. Credentials/URL come from ~/.profile
# (WANDB_BASE_URL, WANDB_API_KEY), inherited into the submitit job via --export=ALL.
#
# Usage:
#   scripts/fairchem_dm1_online.sh -c configs/.../foo.yaml key=value ...
#
# Requires job.debug=False (debug disables the logger) and a job.logger block
# with project/entity/group (the esen configs already set these).
set -u
export CKPT_HOME=${CKPT_HOME:-/checkpoint/amaia/explore/lvj}
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}

# 1) drop the unreachable public-internet proxy so wandb reaches the internal host
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY
# 2) belt-and-suspenders: even if something re-sets the proxy on the node, make
#    requests/wandb bypass it for the internal wandb host
export NO_PROXY="meta-fair.wandb.io,${NO_PROXY:-}"
export no_proxy="meta-fair.wandb.io,${no_proxy:-}"
# 3) ensure ONLINE (never inherit an offline override)
unset WANDB_MODE

source /home/lvj/fairchem/.venv/bin/activate
exec fairchem "$@"
