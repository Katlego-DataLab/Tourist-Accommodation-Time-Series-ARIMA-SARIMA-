# Tourist-Accommodation-Time-Series-ARIMA-SARIMA-
This study analyses South Africa's tourist accommodation sector (2007-2024) using stats SA data and SARIMA modelling. Results show uneven post-Covid recovery, with hotels most resilient. SARIMA outperformed ARIMA, achieving lower RMSE and MAPE, and delivering reliable forecasts to support data-driven tourist policy and planning.
Author: Katlego Mathebula

Time-series analysis of South African tourism accommodation using ARIMA and SARIMA models. This project forecasts occupancy rates, revenue, and room counts across different accommodation types, providing insights for tourism planning and decision-making.

1. Key Features

Cleaned and transformed tourism accommodation data (2007–2024)

Explored occupancy rates, revenue, and number of rooms by accommodation type

Built ARIMA and SARIMA models for hotel occupancy

Generated 5-year SARIMA forecasts with confidence intervals

Evaluated models using RMSE, MAE, MAPE

2. Visualizations

Occupancy rate trends by accommodation type

Revenue (currency) trends over time

Train-test comparison for time series

Forecast vs actual occupancy rates

5-year SARIMA forecast with 80% & 95% confidence intervals

3. Libraries Used

readxl, dplyr, tidyr, stringr, lubridate, ggplot2, scales, forecast, tseries, knitr

4.  Repository Contents

Tourist accomodation From 2007.xlsx – Original dataset

Tourism_Occupancy_Forecasting.Rmd – Analysis script

Tourism_Occupancy_Forecasting.html – HTML report

Tourism_Occupancy_Forecasting.pdf – PDF report

5. Notes

SARIMA captures seasonal patterns in hotel occupancy rates

Forecasts provide actionable insights for tourism operators and policymakers
