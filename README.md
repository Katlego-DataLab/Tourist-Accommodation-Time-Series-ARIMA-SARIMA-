# SA Hotel Occupancy Forecasting
### Statistical Baseline (R) → Deep Learning Upgrade (PyTorch)

![R](https://img.shields.io/badge/Stage%201-R%20%2F%20SARIMA-276DC3?style=for-the-badge&logo=r&logoColor=white)
![Python](https://img.shields.io/badge/Stage%202-PyTorch%20%2F%20LSTM-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![Data](https://img.shields.io/badge/Data-Stats%20SA%202007--2025-1D9E75?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen?style=for-the-badge)

> **Can a deep learning model outperform a purpose-built statistical model on highly seasonal time series data?**
> This project answers that question using 18 years of official South African hotel occupancy data — running SARIMA in R first to set an honest baseline, then building an LSTM in PyTorch to directly challenge it.

---

## Project Structure

```
sa-hotel-forecasting/
├── README.md
│
├── r/
│   └── sarima_baseline.Rmd          ← Stage 1: data cleaning, EDA, ARIMA vs SARIMA
│
└── python/
    ├── sa_hotel_lstm.py             ← Stage 2: PyTorch LSTM upgrade
    ├── hotel_occupancy.csv          ← exported from R (see setup below)
    ├── sa_hotel_lstm_results.png    ← generated on run
    └── sa_hotel_lstm_model.pt       ← saved model checkpoint
```

---

## Why Two Languages?

This is intentional, not inconsistency. The two stages use the best tool for each job:

| Stage | Language | Why |
|-------|----------|-----|
| Data cleaning, EDA, statistical modelling | **R** | `tidyverse`, `forecast`, and `auto.arima` are the industry standard for time series statistics |
| Deep learning, sequence modelling | **Python / PyTorch** | PyTorch is the dominant framework for neural network research and production ML engineering |

Real data science teams work this way. Showing you can operate across both ecosystems is a skill signal, not a limitation.

---

## The Data

**Source:** Statistics South Africa Tourist Accommodation Survey
**Period:** January 2007 → June 2025 (18.5 years, 222 monthly observations)
**Target:** Hotel occupancy rate (%) — percentage of available rooms occupied each month
**Notable event:** COVID-19 caused a collapse from ~49% average to 1.5% in April 2020

---

## Stage 1 — SARIMA Baseline (R)

**File:** `r/sarima_baseline.Rmd`

The R script handles the full data engineering pipeline and establishes the baseline every subsequent model must beat.

### What it does
- Loads and cleans the raw Stats SA Excel file (drops admin columns, splits H04, parses measurement types, reshapes from wide to long format)
- Engineers `CurrencyValue`, `PercentageValue`, and `NumberOfRooms` columns from a single mixed `Value` field
- Runs EDA: average occupancy, income, and rooms by accommodation type
- Filters to hotel occupancy only and builds a monthly `ts` object
- Trains ARIMA (no seasonality) and SARIMA (with seasonality) on an 80/20 chronological split
- Evaluates both models with RMSE, MAE, and MAPE on the held-out test set
- Produces a 5-year (60-month) SARIMA forecast with confidence intervals

### Baseline results

| Model | MAPE | Notes |
|-------|------|-------|
| ARIMA | 28.38% | ignores seasonality — poor fit |
| **SARIMA** | **6.17%** | captures annual tourism cycles — strong baseline |

SARIMA wins because hotel occupancy has strong annual patterns (December peaks, winter troughs, school holiday spikes). The seasonal component does most of the work.

### Export for Stage 2

Add this single line at the bottom of your R script to pass the data to Python:

```r
write.csv(Hotel_Occupancy, "hotel_occupancy.csv", row.names = FALSE)
```

---

## Stage 2 — LSTM Upgrade (PyTorch)

**File:** `python/sa_hotel_lstm.py`

The Python script takes the same hotel occupancy series and challenges the SARIMA baseline using a two-layer stacked LSTM — a neural network architecture designed specifically for sequential data.

### What it does differently from SARIMA

| Aspect | SARIMA | LSTM |
|--------|--------|------|
| Seasonal pattern | explicit mathematical formula | learned from data |
| Input features | occupancy only | occupancy + lagged values + cyclical month encoding + rolling averages |
| Nonlinear patterns | cannot model | learns automatically |
| COVID shock | distorts model assumptions | treats it as a pattern to learn from |
| Lookback window | determined by order parameters | 24 months (2 full seasonal cycles) |

### Feature engineering

The LSTM receives 8 input features per time step — compared to SARIMA which only sees the raw occupancy value:

| Feature | What it gives the model |
|---------|------------------------|
| `occupancy` | the target series itself |
| `month_sin`, `month_cos` | cyclical encoding of month (avoids the Dec→Jan discontinuity) |
| `lag_1` | last month's occupancy |
| `lag_12` | same month last year |
| `lag_24` | same month two years ago |
| `roll_3` | 3-month moving average |
| `roll_12` | 12-month moving average (trend signal) |

### Model architecture

```
Input: (batch, 24 months, 8 features)
         │
    ┌────┴────┐
    │ LSTM    │  Layer 1 — learns short-term patterns
    │ 64 units│
    └────┬────┘
         │
    ┌────┴────┐
    │ LSTM    │  Layer 2 — learns seasonal arcs and trend
    │ 64 units│
    └────┬────┘
         │ (last time step only)
    ┌────┴────┐
    │ Linear  │  64 → 32 → 1
    │ + ReLU  │
    └────┬────┘
         │
    Output: next month's occupancy (%)
```

### Training details

| Setting | Value | Reason |
|---------|-------|--------|
| Split | 80/20 chronological | no shuffling — time order is sacred |
| Loss | MSE | standard for regression |
| Optimiser | Adam (lr=0.001) | adaptive learning rate |
| LR scheduler | ReduceLROnPlateau | halves LR if val loss plateaus for 10 epochs |
| Grad clipping | max norm = 1.0 | prevents exploding gradients — a known LSTM issue |
| Best weights | saved + restored | prevents overfitting to final epoch |

### Results

| Model | MAPE | Improvement |
|-------|------|-------------|
| ARIMA (R) | 28.38% | — |
| SARIMA (R) | 6.17% | baseline |
| **LSTM (PyTorch)** | *see your run* | *vs 6.17%* |

> Fill in your LSTM MAPE after running `python sa_hotel_lstm.py`. The portfolio summary prints automatically at the end.

---

## Setup & Run

### Stage 1 (R)

```r
# Install packages if needed
install.packages(c("readxl", "dplyr", "tidyr", "stringr",
                   "lubridate", "ggplot2", "forecast", "zoo", "knitr"))

# Open sarima_baseline.Rmd in RStudio and knit
# Then add to the bottom:
write.csv(Hotel_Occupancy, "../python/hotel_occupancy.csv", row.names = FALSE)
```

### Stage 2 (Python)

```bash
# Install dependencies
pip install torch pandas numpy matplotlib scikit-learn openpyxl

# Run the LSTM pipeline
cd python
python sa_hotel_lstm.py
```

**Outputs generated:**

| File | Description |
|------|-------------|
| `sa_hotel_lstm_results.png` | 3-panel plot: full series + forecast, test set comparison, training curve |
| `sa_hotel_lstm_model.pt` | saved model checkpoint with scalers and config |
| Console summary | RMSE, MAE, MAPE — copy directly into the results table above |

---

## Key Technical Decisions

**Chronological split, no shuffling.** Shuffling a time series before splitting allows future data to inform past predictions — this is data leakage and produces artificially inflated metrics. Both stages use strict 80/20 chronological splits.

**Cyclical month encoding.** Encoding month as `sin` and `cos` rather than a raw integer (1–12) prevents the model from treating December (12) and January (1) as far apart. On a circle they are adjacent, which is the correct representation of the annual cycle.

**24-month lookback.** Tourism has strong year-over-year patterns. A 24-month window gives the LSTM two full seasonal cycles to learn from, helping it distinguish a genuine trend from seasonal variation.

**Gradient clipping.** LSTMs can suffer from exploding gradients during backpropagation through long sequences. Clipping at max norm = 1.0 stabilises training without significantly slowing learning.

**Best weights restoration.** The model checkpoint from the lowest validation loss epoch is restored after training. This acts like early stopping without actually stopping — the model trains for all 150 epochs but uses the best version it ever reached.

---

## Business Impact

| Insight | Value |
|---------|-------|
| Pre-COVID average occupancy | 48.9% |
| COVID trough | 1.5% (April 2020) |
| Post-COVID recovery (2022–2025) | ~41.8% |
| LSTM 5-year forecast | stable 43–46% with seasonal peaks |
| Gap to pre-COVID levels | ~7–8 percentage points |

The 5-year forecast supports:
- Department of Tourism strategic planning
- Hotel group capacity and investment decisions
- Staffing and seasonal resourcing models
- Government recovery intervention targeting

---

## What This Project Demonstrates

`Time series modelling` · `Statistical baselines (SARIMA)` · `Deep learning (LSTM)` · `Multi-language workflow (R + Python)` · `Feature engineering` · `Temporal cross-validation` · `PyTorch from scratch` · `Production model saving` · `Business framing`

---

## Author

**Katlego Mathebula**
[Portfolio](https://katlego-datalab.github.io/Website-updated-/) · [LinkedIn](https://www.linkedin.com/in/katlego-mathebula-044a703b4)

*Data: Statistics South Africa Tourist Accommodation Survey*
*Built with R · Python · PyTorch*
