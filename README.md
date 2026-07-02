# Photovoltaic Power Forecasting Code

This folder contains a cleaned, submission-ready version of the forecasting code used in the manuscript experiments. The original notebooks and source files were not modified.

## Folder Layout

```text
submission_code/
  configs/default.json          Default experiment settings.
  data_vmd/                     Processed seven-site VMD input data.
  results/                      Existing prediction CSV files from the notebook runs.
  scripts/train.py              Re-train one model or all models.
  scripts/evaluate_results.py   Compute metrics from prediction CSV files.
  scripts/plot_results.py       Generate publication-style diagnostic plots.
  src/pv_forecasting/           Reusable data, model, training, and metric code.
```

## Environment

Python 3.9 or later is recommended. Install dependencies with:

```bash
pip install -r requirements.txt
```

The training code automatically uses CUDA when a compatible GPU is available.

## Reproducing Experiments

Run from this folder:

```bash
python scripts/train.py --model all --epochs 20
```

Train a single model:

```bash
python scripts/train.py --model gcn-informer --epochs 20
python scripts/train.py --model gat-informer --epochs 20
python scripts/train.py --model gcn-lstm --epochs 20
python scripts/train.py --model informer --epochs 20
python scripts/train.py --model lstm --epochs 20
```

Evaluate saved prediction files:

```bash
python scripts/evaluate_results.py --results-dir results
```

Generate plots:

```bash
python scripts/plot_results.py --results-dir results --output-dir figures
```

## Output Format

Each prediction CSV uses the same schema:

```text
y_true,q0.1,q0.5,q0.9
```

The columns correspond to observed aggregated active power and the 0.1, 0.5, and 0.9 quantile forecasts.

## Notes

The `data_vmd` directory contains processed input data with English file names for submission convenience. The large raw source data are intentionally excluded from this folder.
