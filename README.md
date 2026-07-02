# VMD-GCN-Informer Photovoltaic Cluster Forecasting

This repository contains the submission-oriented code for the manuscript:

**Probabilistic Power Forecasting for Photovoltaic Plant Clusters Using VMD-GCN-Informer**

The code is organized to match the manuscript pipeline: VMD-enhanced input features, Pearson-correlation graph construction, GCN/GAT spatial modeling, Informer or LSTM temporal modeling, and quantile regression for probabilistic forecasting.

## Repository Layout

```text
configs/default.json              Default manuscript-style settings.
data_vmd/                         Processed seven-site VMD input data.
results/reported/                 Values reported in the manuscript tables.
results/notebook_exports/         Prediction CSV files exported from earlier notebooks.
figures/                          Diagnostic plots generated from notebook exports.
scripts/train.py                  Re-train one model or all models.
scripts/evaluate_results.py       Evaluate prediction CSV files.
scripts/plot_results.py           Generate diagnostic plots from prediction CSV files.
src/pv_forecasting/               Reusable data, model, training, and metric code.
```

## Environment

Python 3.9 or later is recommended.

```bash
pip install -r requirements.txt
```

CUDA is used automatically when a compatible GPU is available.

## Manuscript Settings

The default configuration follows the manuscript:

- Seven PV sites from the DKASC Yulara dataset.
- VMD-processed input files in `data_vmd/`.
- Chronological train/test split: first 80% of days for training and remaining 20% for testing.
- Sequence length: 24 samples.
- Sampling interval: 5 minutes.
- Pearson correlation graph constructed from training-set active power.
- Graph threshold: 0.6.
- Hidden dimension: 32.
- GCN layers: 2.
- Optimizer: Adam.
- Learning rate: 0.001.
- Epochs: 50.
- Quantiles: 0.1, 0.5, and 0.9.

## Reported Results

The manuscript tables are stored separately from raw prediction exports:

```text
results/reported/table3_model_comparison.csv
results/reported/table4_multi_horizon_mae.csv
results/reported/table5_seasonal_metrics.csv
```

This separation is intentional. The files in `results/notebook_exports/` are prediction CSV files exported from earlier notebook runs and are kept for auditability. They are not treated as the authoritative manuscript tables.

## Re-training

Run from the repository root:

```bash
python scripts/train.py --model gcn-informer
```

Train all implemented models:

```bash
python scripts/train.py --model all
```

New prediction files are written to `results/generated_predictions/` by default, so they do not overwrite the reported manuscript tables.

## Evaluation and Plotting

Evaluate prediction CSV files:

```bash
python scripts/evaluate_results.py --results-dir results/notebook_exports
```

Generate diagnostic plots:

```bash
python scripts/plot_results.py --results-dir results/notebook_exports --output-dir figures
```

Prediction CSV files use:

```text
y_true,q0.1,q0.5,q0.9
```

where `q0.1`, `q0.5`, and `q0.9` are the 10th, 50th, and 90th percentile forecasts for aggregated PV cluster active power.
