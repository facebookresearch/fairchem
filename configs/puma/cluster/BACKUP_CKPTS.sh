#!/bin/bash

PUMA_DIR="/checkpoint/ocp/shared/puma"
aws s3 sync ${PUMA_DIR} s3://opencatalysisdata/fm/checkpoints/puma
