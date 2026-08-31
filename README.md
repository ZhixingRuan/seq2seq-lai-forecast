# seq2seq-lai-forecast

A sequence-to-sequence ConvLSTM model for forecasting daily, gridded, 1-km Leaf Area
Index (LAI) up to 30 days ahead from meteorological forcing and historical LAI
sequences. This is the code accompanying:

> Ruan, Z. and Lu, L. (2026). *A Sequence-to-Sequence ConvLSTM Approach for Leaf Area
> Index Forecasting over the South-Central United States.*
> arXiv:[2608.00879](https://arxiv.org/abs/2608.00879) [physics.ao-ph]

Most machine-learning work on LAI has targeted estimation at point or regional scales.
The aim here is a prognostic, spatially continuous forecast — LAI fields that can be
carried forward in time and used as input to coupled land–atmosphere and Earth system
models, on subseasonal-to-seasonal (S2S) timescales. 

This repo covers the model and the pipeline that feeds it — the architecture, the
training and inference code, and the preprocessing that builds a model-ready zarr from
raw sources. The ablation sweeps, verification metrics, and analysis reported in the paper are not included here.

## Model

`ConvLSTMSeq2SeqDOY` / `ConvLSTMSeq2SeqDOY2Stacked` in [src/model.py](src/model.py)
implement a two-layer stacked ConvLSTM encoder–decoder. The encoder consumes a rolling
window of gridded daily weather (minimum and maximum temperature, precipitation), land
cover, and past LAI; the decoder rolls the forecast forward day by day over the lead
time. Day-of-year is supplied as sine/cosine encodings so the model has an explicit
seasonal reference, and training uses a masked MSE loss ([src/loss.py](src/loss.py)) so
that grid cells with missing or interpolated LAI do not contribute gradients.

Default configuration is `kernel_size=3`, `hidden_dim=16`; see
[configs/config.example.yaml](configs/config.example.yaml) for the full set.

## Repository layout

```
configs/
└── config.example.yaml         # base template — copy and edit paths before use
src/
├── model.py                    # ConvLSTM encoder-decoder architecture
├── data.py                     # SequenceDataset — windowing/patching of the zarr input
├── loss.py                     # masked MSE loss
├── config.py                   # pydantic config schema, loaded from configs/*.yaml
├── train.py                    # training loop
├── predict.py                  # inference
├── run.py                      # training entry point (typer CLI)
├── data_additional_laiflag.py  # helper used by predict.py to substitute the
│                               #   LAI-flag mask at inference time
├── tensorboard_summary.py      # RMSE tracking used by train.py
├── util.py, log.py             # small shared utilities
└── preprocess/                 # builds a model-ready zarr from raw LAI,
                                #   weather (GHCN-D), and land-cover sources
    ├── lai_aggregate.py                # MODIS LAI 500 m -> 1 km LAEA-projected grid
    ├── lc_aggregate.py                 # MODIS land cover -> target grid
    ├── lc_preprocess.py                # group land cover into PFT classes
    ├── ghcnd_station_analysis.py       # station metadata parsing (parse_stations)
    ├── ghcnd_byyear_extract.py         # extract station obs within the domain, by year
    ├── interpolate_barnes_km.py        # station tmax/tmin -> grid (Barnes interpolation)
    ├── interpolate_IDW_km.py           # station precip -> grid (IDW interpolation)
    ├── weather_var_process_km.py       # normalize weather vars, add rolling precip sums
    ├── lai_process_km.py               # LAI nan-fill + normalize + coord-normalize
    └── create_model_ready_dataset.py   # combine LAI + weather + land cover -> input zarr
```

## Setup

```bash
conda env create -f environment.yml
conda activate ml_lai
```

(Drop the `pytorch-cuda` line in [environment.yml](environment.yml) for a CPU-only
install.)

## Data

This repo does not include raw data or trained checkpoints.

The training and prediction pipeline expects a model-ready zarr store with the
feature/target variables listed in
[configs/config.example.yaml](configs/config.example.yaml) (e.g. `tmin_norm`,
`tmax_norm`, `prcp_norm`, `LAI_norm`, plus their `*_flag` masks) on a shared
`(time, lat, lon)` grid — see [src/data.py](src/data.py) for exactly how a window is
built from it. LAI carries a three-state flag: `1` for a valid retrieval, `2` for a
carry-forward fill, `0` for invalid. The masked loss uses these to exclude unreliable
cells from training.

[src/preprocess/](src/preprocess/) has the scripts used to build that zarr from raw
sources:

- **LAI** — MODIS MCD15A3H, quality-screened and aggregated from 500 m to the 1-km
  LAEA-projected target grid
- **Weather** — GHCN-D daily station observations, interpolated to the grid with Barnes
  analysis for temperature and inverse-distance weighting for precipitation
- **Land cover** — MODIS land cover, grouped into plant functional type classes

`create_model_ready_dataset.py` is the assembly step that combines the three into the
final input zarr. These were written as one-off pipeline scripts with input/output paths
hardcoded as constants near the top of each file rather than exposed as CLI flags; those
paths have been replaced with `/path/to/...` placeholders — edit them directly before
running.

## Configuration

Copy the example config and point it at your own data/output paths:

```bash
cp configs/config.example.yaml configs/my_run.yaml
# edit dataset.data.path and root_dir in configs/my_run.yaml
```

Config parsing is a plain `yaml.safe_load` (see [src/config.py](src/config.py)), so
paths are not shell-expanded — edit them directly.

## Training

```bash
bash scripts/train.bash
```

which runs (see [scripts/train.bash](scripts/train.bash)):

```bash
python run.py --config ../configs/my_run.yaml \
    --checkpoint-epochs 5 \
    --accumulation-steps 4
```

Add `--resume` (with `--resume-checkpoint-dir`) to resume from a checkpoint — see the
commented-out example at the bottom of the script.

Training in the paper was run on NVIDIA A100/H100 GPUs under Slurm; gradient
accumulation (`--accumulation-steps`) is there to keep the effective batch size
reasonable when a full spatial window does not fit in memory.

## Prediction

```bash
bash scripts/predict.bash
```

Edit the `--model`/`--input`/`--output`/`--config` paths in
[scripts/predict.bash](scripts/predict.bash) to point at your trained run first; it
otherwise wraps:

```bash
python predict.py \
    --model /path/to/run/best_model.pth \
    --input /path/to/model_input.zarr \
    --output /path/to/preds/pred.zarr \
    --config /path/to/run/config.yaml
```

## License

MIT — see [LICENSE](LICENSE).