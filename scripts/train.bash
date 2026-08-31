#!/bin/bash
# Example: launch training. Run from the repo root: bash scripts/train.bash

cd "$(dirname "$0")/../src"

python run.py \
    --config ../configs/my_run.yaml \
    --checkpoint-epochs 5 \
    --accumulation-steps 4

# To resume from a checkpoint instead:
# python run.py \
#     --config ../configs/my_run.yaml \
#     --resume \
#     --resume-checkpoint-dir /path/to/run/checkpoints
