#!/bin/bash
# Example: run inference with a trained checkpoint. Run from the repo root:
# bash scripts/predict.bash

cd "$(dirname "$0")/../src"

python predict.py \
    --model /path/to/run/best_model.pth \
    --input /path/to/data/model_input.zarr \
    --output /path/to/run/preds/pred.zarr \
    --config /path/to/run/config.yaml \
    --start-date '2021-01-01' \
    --end-date '2023-12-31'
