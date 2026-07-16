# Electricity Demand Forecasting — German Grid Case Study

A single, reproducible notebook that forecasts German electricity demand two years
ahead at a weekly resolution and tests whether any model can beat a seasonal-naive
baseline. The workflow runs step by step, from the raw hourly file to a final
model comparison.

## Aim

Forecast the last 104 weeks of German demand and compare benchmark, statistical,
tree-based and neural methods against the seasonal naive, judging each by absolute
error and a skill score rather than by which model looks most sophisticated.

## Data

- **Load:** hourly German load (`DE_load_actual_entsoe_transparency`) from Open
  Power System Data, January 2015 to October 2020, binned to weekly and daily
  series (megawatts).
- **Temperature:** daily Berlin temperatures from the Open-Meteo archive, reduced
  to weekly minimum and maximum. Because the test window uses observed
  temperature, the temperature-driven forecasts are reported as conditional.
- **Calendar:** a working-day share per week and a post-2020 indicator.

## Running the notebook

Open `Lakshmi_Manthri_24085661.ipynb` in Google Colab (a GPU runtime helps
the neural step) and run the cells in order.

- The load CSV is read locally if present, otherwise downloaded from OPSD.
- The SARIMA grid search fits the full mandated range and takes a while; the
  neural step is the other slow cell.
- `holidays` and other packages install from `requirements.txt`; on Colab most are
  preinstalled.

## Steps in the notebook

1. Load the OPSD hourly file.
2. Repair short gaps and bin to daily and weekly series.
3. Exploratory analysis: line plots, a seasonal subseries view, and autocorrelation.
4. Stationarity testing with the ADF test under two deterministic specifications.
5. Benchmark forecasts (mean, naive, seasonal naive, drift) with shared scoring.
6. SARIMA order search over the full AIC grid, refined on a validation split.
7. SARIMA residual diagnostics (Ljung-Box, skewness, kurtosis).
8. SARIMA two-year forecast with confidence intervals.
9. SARIMAX with temperature and calendar regressors (conditional).
10. A feature-based residual-learning hybrid: ExtraTrees on the seasonal anomaly.
11. An LSTM trained on the hourly data, tuned and rolled forward block by block.
12. Error growth as a function of the forecast horizon.
13. A final comparison figure and metrics table.

## Models

Mean, naive, seasonal naive and drift benchmarks; SARIMA (orders from the full AIC
grid, confirmed by validation RMSE); SARIMAX with temperature and calendar
regressors; an ExtraTrees hybrid on the seasonal anomaly; and an hourly LSTM.

## Evaluation

Every model is scored on the same 104-week window with the mean absolute error
(MAE), the median absolute error (MedAE), and a skill score against the seasonal
naive (zero matches the baseline, negative is worse). SARIMA is additionally
reported with RMSE.

## Avoiding data leakage

- The seasonal expectation and any fitted transforms use training data only.
- The LSTM scaler is fitted on training hours only, and its forecast is produced
  block by block without consuming test values.
- The hybrid uses no target lags; its only future input is observed temperature,
  which is declared as a conditional forecast.
- The train/test split is the final 104 weeks in time order (no shuffling).

## Outputs

Running the notebook writes to an `exports/` folder next to it:

```
exports/metrics_overview.csv     # MAE, MedAE, skill per model
exports/all_models_overlay.png   # all forecasts vs actual
```

## Requirements

See `requirements.txt`. On Colab most packages are preinstalled.

## Files

```
Lakshmi_Manthri_24085661.ipynb   # the analysis
Lakshmi_Manthri_24085661_Report.docx                            # final report
requirements.txt                       # dependencies
time_series_60min_singleindex.csv      # load data (optional local copy)
```
