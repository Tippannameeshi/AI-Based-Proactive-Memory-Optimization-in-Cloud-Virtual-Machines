# Memory Spike Preprocessing (Capstone)

This project prepares VM telemetry data for memory spike detection and sequence models (e.g., LSTM). It loads raw VM telemetry, engineers time-series features, creates a `memory_spike` target, scales features, and exports train/val/test sequences plus artifacts for downstream modeling.

## What’s Inside

- `Memory_Preprocessing.ipynb` — end-to-end preprocessing workflow.
- `vmCloud_data.csv` — raw VM telemetry dataset.
- `preprocessed_data.csv` — scaled, feature-engineered dataset.
- `X_train.npy`, `X_val.npy`, `X_test.npy` — sequence inputs.
- `y_train.npy`, `y_val.npy`, `y_test.npy` — sequence labels.
- `scaler.pkl`, `label_encoders.pkl` — fitted preprocessing artifacts.
- Plots and diagrams:
- `Architecture_Diagram.png`
- `boxplot_before.png`
- `correlation_heatmap.png`
- `memory_spike_distribution.png`

## Data Columns (from notebook output)

The notebook expects columns such as:

- `vm_id`, `timestamp`
- `cpu_usage`, `memory_usage`, `network_traffic`, `power_consumption`
- `num_executed_instructions`, `execution_time`, `energy_efficiency`
- `task_type`, `task_priority`, `task_status`

## Preprocessing Summary

- Drops `vm_id` as a label-only field.
- Label-encodes categorical columns (`vm_id`, `timestamp`, task fields).
- Removes outliers (IQR-based) for key numeric columns.
- Adds rolling stats for `memory_usage` and `cpu_usage`.
- Adds lag features and rate-of-change features.
- Creates interaction features (`memory_cpu_product`, `memory_cpu_ratio`).
- Defines `memory_spike` as `memory_usage` > 80th percentile.
- Scales numeric features to `[0, 1]` using `MinMaxScaler`.
- Builds fixed-length sequences (window size = 10).
- Splits sequences into 70% train / 15% val / 15% test.

## How To Run

1. Open `Memory_Preprocessing.ipynb`.
2. Run all cells in order.
3. Outputs are written to the project root.

## Outputs

- `preprocessed_data.csv` — full scaled dataset.
- `X_*.npy`, `y_*.npy` — sequence data and labels.
- `scaler.pkl`, `label_encoders.pkl` — preprocessing artifacts.

## Notes

- The `memory_spike` threshold is computed as the 80th percentile of `memory_usage` in the dataset.
- The notebook saves plots for EDA and feature correlation.

## Next Step Ideas

- Train an LSTM or Temporal CNN using `X_train.npy` / `y_train.npy`.
- Evaluate with ROC-AUC and PR-AUC due to class imbalance.
- Consider experimenting with a higher spike percentile for stricter labeling.
