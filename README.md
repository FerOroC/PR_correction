# GNSS Pseudorange Error Correction

ML framework for correcting GNSS pseudorange errors (multipath / NLOS) using simGNSS simulated urban driving data.


## Data

Place CSV files under `data/data_v1/` and `data/data_v2/` (not included in this repo).
Expected naming convention: `sat_data_V1A1_<day>Jan_<scenario>_multipath_processed.csv`

Link to data: 
---

## Models

### Non-sequential (epoch-wise) — `utils.py`

`PrNet` | Deep MLP, each satellite independently 
`transformer` | Cross-satellite self-attention 
`Fcnn_Lstm` | Per-satellite FCNN + LSTM over randomly-packed visible set 

### Sequential (sliding window) — `temporal_utils.py`

`LSTM` | Per-satellite independent LSTM over window 
`MAMBA` | State-space sequence model (optional dep.) 
`transformer_enhanced` | LSTM + attention + BiLSTM 
`1DUnet` | 1-D U-Net across the time dimension 
`DiffGNSS` | Coarse MLP init + DDIM diffusion refinement 

---

## Running experiments

```bash
python main.py --model <MODEL> --dataset <v1|v2> [options]
```

## Output

For each test file pattern a separate CSV is written:

```
results_<MODEL>_<DATASET>_<TEST_PATTERN>.csv
```

Columns: `index`, `time_epoch`, `sat_id`, `actual_target_value`, `model_output`.

MAE and RMSE over valid satellites are printed to stdout per test file.

---

## Evaluation 

Jupyter notebook provided for pseudorange correction and WLS position estimation, as well as horizontal error scores.

## Configuration

Environment.yml provided.
All default hyperparameters live in `config.json`.
CLI flags override any value at runtime.

**General defaults:** 500 epochs, lr = 1e-4, StepLR(step\_size=100, γ=0.5), Adam, L1 loss, batch size 512.

