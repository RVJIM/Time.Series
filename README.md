# 📉 Time Series Analysis — EMH Testing & ARIMA Forecasting

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![statsmodels](https://img.shields.io/badge/statsmodels-0.14+-lightgrey)
![pandas](https://img.shields.io/badge/pandas-2.0+-150458?logo=pandas&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-green)

> **Regression and Time Series Models** — Homework 2  
> University of Padova | 2024  
> Authors: Cristian Angelini, **Riccardo Caruso** (Lead Developer), Francesco Salvagnin

---

## 📌 Overview

This project performs a comprehensive time series analysis of **Japanese financial and macroeconomic data**, testing the **Efficient Market Hypothesis (EMH)** through a battery of statistical tests and building **ARIMA/ARMA forecasting models** to predict future price movements.

The analysis covers both **daily and monthly** timeframes across equity and economic indicators.

---

## 📂 Repository Structure

```
Time.Series/
│
├── Completo.py              # Main script — full pipeline (all 7 assignments)
├── fun.py                   # Core utility functions
├── f.py                     # Additional helper functions
│
├── equities.xlsx            # Financial data (5 Japanese equities + Nikkei)
├── economics.xlsx           # Macroeconomic data (CPI, IPI, JGB 10Y, Unemployment)
│
├── Daily/                   # Output plots — daily timeframe analysis
├── Monthly/                 # Output plots — monthly timeframe analysis
├── Economics/               # Output plots — macroeconomic analysis
├── Jarque Bera/             # Normality test output plots
└── originali/               # Raw original data files
```

---

## 📊 Dataset

**Financial Data** — sourced from Yahoo Finance
| Asset | Type | Frequency |
|---|---|---|
| Nikkei 225 | Market Index | Daily & Monthly |
| Nippon Telecommunication | Large-cap Equity | Daily & Monthly |
| Sony | Large-cap Equity | Daily & Monthly |
| Keyence | Large-cap Equity | Daily & Monthly |
| Toyota | Large-cap Equity | Daily & Monthly |
| Fast Retailing | Large-cap Equity | Daily & Monthly |

**Macroeconomic Data** — sourced from FRED
| Indicator | Frequency |
|---|---|
| CPI (Consumer Price Index) | Monthly |
| Industrial Production Index | Monthly |
| Japan 10Y Government Bond Yield | Monthly |
| Unemployment Rate | Monthly |

**Sample period:** December 2012 – December 2023

---

## 🔬 Methodology

### Assignment 2 — Data Transformation & Return Distribution
- Log-price transformation for financial series (variance stabilization)
- Logarithmic returns: `r_t = 100 * (log(P_t) - log(P_{t-1}))`
- Percentage changes for macroeconomic indicators
- **Jarque-Bera test** for normality → confirmed leptokurtic, negatively skewed distributions consistent with financial theory

### Assignment 3 — Stationarity Testing
- **Augmented Dickey-Fuller (ADF)** test with **BIC-driven model selection** (constant / trend / none)
- All log-price series confirmed as **I(1)** — unit root in levels, stationary in first differences
- Economics series: CPI and JGB Yield stationary at level; Industrial Production and Unemployment Rate stationary after first differencing

### Assignment 4 — ARMA Model Identification
- **ACF and PACF** visual inspection for order identification
- **BIC grid search** across all (p, d, q) combinations
- First-differenced financial returns → **ARIMA(0,1,0) White Noise**, consistent with EMH (Fama, 1970)
- **Ljung-Box test** on residuals confirming no serial autocorrelation

### Assignment 5 — ARIMA Forecasting
- Dynamic out-of-sample forecasts for all series
- Financial returns: forecast collapses to unconditional mean (EMH confirmed)
- Exceptions: NIKKEI Daily → **ARMA(1,0)**; Keyence Monthly → **ARMA(1,0)**
- Economic series: heterogeneous specifications (CPI → ARIMA(1,0,1); Unemployment → ARIMA(1,1,3))

### Assignment 6 — ARIMA vs. Random Walk
- Head-to-head comparison of ARIMA forecasts against naive **Random Walk benchmark**
- ARIMA outperforms RW for non-stationary series, especially at daily frequency
- Monthly models less accurate due to longer forecasting horizon (2 years vs. 6 months for daily)

### Assignment 7 — Volatility Modelling (Squared Log Returns)
- ARMA modelling on **squared log returns** as a volatility proxy
- Results: ARMA(1,1) for most series; AR(4) for Nippon Tel; ARMA(1,2) for Nikkei
- Squared returns exhibit autocorrelation → volatility is **predictable** even when returns are not
- Findings motivate future implementation of **GARCH/ARCH** models

---

## ⚙️ Setup & Usage

### Requirements
```bash
pip install pandas numpy matplotlib statsmodels openpyxl scipy
```

### Run the full pipeline
```bash
python Completo.py
```

All output plots are saved automatically to the corresponding subdirectories (`Daily/`, `Monthly/`, `Economics/`, `Jarque Bera/`).

---

## 📈 Key Results

| Finding | Result |
|---|---|
| Return distribution | Non-normal, leptokurtic, negatively skewed (all series) |
| Financial log-prices | I(1) — unit root confirmed |
| Best ARIMA (most financial) | ARIMA(0,1,0) — White Noise |
| EMH | Supported for returns; not supported for volatility |
| Volatility predictability | Confirmed via squared log returns ARMA models |
| ARIMA vs. Random Walk | ARIMA superior, especially at daily frequency |

---

## 📚 References

- Fama, E. F. (1970). *Efficient Capital Markets: A Review of Theory and Empirical Work.* Journal of Finance.
- Box, G. E. P., & Jenkins, G. M. (1976). *Time Series Analysis: Forecasting and Control.*
- Jarque, C. M., & Bera, A. K. (1980). *Efficient tests for normality, homoscedasticity and serial independence of regression residuals.*
- Dickey, D. A., & Fuller, W. A. (1979). *Distribution of the estimators for autoregressive time series with a unit root.*

---

## 👥 Authors

| Name | Contribution |
|---|---|
| **Riccardo Caruso** | Lead Developer — Full codebase (all 7 assignments) + Partial report (assignments 4, 5, 7) |
| Cristian Angelini | Partial code (assignments 1–3) + Report writing (all assignments) |
| Francesco Salvagnin | Partial code (assignments 1–3) + Report writing (all assignments) |
