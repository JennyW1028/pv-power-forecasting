# Manuscript-to-Code Map

This document maps the manuscript sections and tables to the organized code in this repository.

## Methodology

| Manuscript item | Repository implementation |
| --- | --- |
| VMD preprocessing | Processed files in `data_vmd/`; VMD components are included as `Active_Power_VMD1`, `Active_Power_VMD2`, and `Active_Power_VMD3`. |
| Seven-site PV cluster | `src/pv_forecasting/data.py::load_site_frames` loads seven CSV files from `data_vmd/`. |
| 80%/20% chronological split | `src/pv_forecasting/data.py::split_by_date`. |
| Pearson graph with threshold 0.6 | `src/pv_forecasting/data.py::build_correlation_adjacency`. |
| GCN-Informer | `src/pv_forecasting/models.py::GCNInformer`. |
| GAT-Informer baseline | `src/pv_forecasting/models.py::GATInformer`. |
| GCN-LSTM baseline | `src/pv_forecasting/models.py::GCNLSTM`. |
| LSTM baseline | `src/pv_forecasting/models.py::QuantileLSTM`. |
| Informer baseline | `src/pv_forecasting/models.py::QuantileInformer`. |
| Quantile regression loss | `src/pv_forecasting/training.py::quantile_loss`. |
| MAE, RMSE, PICP, RI, PINAW/MPIW | `src/pv_forecasting/metrics.py`. |

## Manuscript Tables

| Manuscript table | Repository file |
| --- | --- |
| Table 3. Metrics for different methods | `results/reported/table3_model_comparison.csv` |
| Table 4. MAE for different methods with various step sizes | `results/reported/table4_multi_horizon_mae.csv` |
| Table 5. Metrics for GCN-Informer of different seasons | `results/reported/table5_seasonal_metrics.csv` |

## Notes

Earlier notebook-exported prediction files are stored under `results/notebook_exports/`. They are retained for traceability, but the manuscript-reported numerical tables are stored under `results/reported/`.
