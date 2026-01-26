# Meow: Financial Time-Series Prediction

## Overview

Meow is a modular framework for financial time-series prediction based on high-frequency trading data stored in HDF5 format.
It supports data cleaning, feature engineering, model training, and evaluation in an end-to-end pipeline.

## Project Structure

```
.
├── meow.py                # Main pipeline (train & evaluate)
├── datawash.py            # Data cleaning
├── dl.py                  # Data loader
├── feat.py                # Feature engineering
├── mdl.py                 # LightGBM model
├── trained_LSTM.py        # LSTM model (optional)
├── eval.py                # Evaluation metrics
├── tradingcalendar.py     # Trading calendar
├── log.py                 # Logging utility
├── archive/               # HDF5 data directory
├── cache/                 # Saved models
└── resources/calendar     # Trading days list
```

## Requirements

- Python ≥ 3.8
- numpy
- pandas
- scikit-learn
- lightgbm
- tensorflow (optional, for LSTM)

Install dependencies:

```
pip install numpy pandas scikit-learn lightgbm tensorflow joblib
```

## Data

Input data should be stored as daily `.h5` files in the `archive/` directory.
Each file represents one trading day.

## Usage

Run the full pipeline:

```
python meow.py
```

This will:

1. Clean raw data
2. Load trading-day data
3. Generate features
4. Train the model
5. Evaluate prediction performance

Example:

```
engine = MeowEngine(h5dir="archive", cacheDir="cache")
engine.fit(20230601, 20231130)
engine.eval(20231201, 20231229)
```

## Features

The system generates technical and microstructure features including:

- Order book imbalance
- Trade imbalance
- Lagged returns
- RSI, MACD, Bollinger Bands
- Volatility and volume ratios

Target variable: `fret12`
Prediction output: `forecast`

## Models

- **LightGBM (default):** tree-based regression with grid search
- **LSTM (optional):** deep learning time-series model (`trained_LSTM.py`)

## Evaluation

Performance metrics:

- Pearson correlation
- R² score
- Mean Squared Error (MSE)

## Logging

Custom logger with timestamp and file-line tracking is provided in `log.py`.

