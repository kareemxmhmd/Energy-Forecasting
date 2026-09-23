Roadmap: Energy Forecasting Neural Network

Phase 1 — Data Loader (data_loader.py)

Load raw CSV
Combine Date + Time → single datetime column, set as index
Cast all numeric columns to float (they load as strings/object due to missing markers)
Print shape, dtypes, and date range as a sanity check

Phase 2 — Preprocessing (preprocess.py)

Handle missing values (0-fill, as you decided) — print before/after counts
Check and drop duplicate rows/timestamps — print count dropped
Resample to your chosen frequency (per-minute/hourly/daily — pending your answer above)
Sort by datetime index to guarantee chronological order

Phase 3 — Feature Engineering (build_features.py)

Time-based features: hour, day of week, month, is_weekend
Lag features: power at t-1, t-24 (or equivalent for your resolution), etc.
Rolling statistics: rolling mean/std over a window (e.g. last 24 readings)
Scale features (MinMaxScaler or StandardScaler) — fit only on training data, save the scaler
Create sequences/windows for the neural net input (X = past N steps, y = next step or next N steps)

Phase 4 — Train/Val/Test Split

Chronological split (no shuffling) — e.g. 70/15/15 by time
Confirm no leakage: scaler fit before split boundary only, lag features don't cross the split incorrectly

Phase 5 — Model Training (train.py)

Build TensorFlow model — likely LSTM or GRU given the time-series/sequential nature
Try at least 2 architectures (e.g., simple LSTM vs stacked LSTM, or LSTM vs GRU) since your standard is to compare multiple models and pick the best
Train with early stopping + validation loss monitoring
Save the best model, the scaler, and the feature column order

Phase 6 — Evaluation (evaluate.py)

Load saved model + scaler
Predict on test set, inverse-transform to real units
Compute RMSE, MAE, MAPE
Plot predicted vs actual (time series line plot) to visually sanity-check
Sanity-check against a naive baseline (e.g., "predict = previous value") — if your model doesn't beat that, something's wrong

Phase 7 — (Optional, if you want it deployable later)

FastAPI server exposing /predict — following your usual backend standard
Frontend if this needs a UI, following your black/white/dark-blue design rules