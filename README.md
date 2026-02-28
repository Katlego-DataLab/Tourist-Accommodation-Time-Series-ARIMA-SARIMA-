# Forecasting Tourism Accommodation Trends in South Africa

### ARIMA & SARIMA Time-Series Modeling (2007–2024)

**Author:** Katlego Mathebula
**Tools Used:** R, ARIMA, SARIMA, Forecasting, Data Cleaning, Visualization
**Project Type:** End-to-End Time Series Analytics Project


##  Project Overview

This project analyzes **tourism accommodation trends in South Africa (2007–2024)** and builds predictive time-series models using **ARIMA and SARIMA** to forecast **hotel occupancy rates** for the next 5 years.

The project includes:

- Data cleaning and transformation
- Feature engineering
- Exploratory data analysis (EDA)
- Trend visualization
- Train-test model evaluation
- Forecast generation with confidence intervals
- Business interpretation of results

This project demonstrates **real-world forecasting capability** applicable to tourism, hospitality, government planning, and investment strategy.


##  Business Problem

The tourism and hospitality industry plays a major role in South Africa’s economy. However:

- Occupancy rates fluctuate due to seasonality
- Revenue planning becomes difficult without forecasting
- Investment decisions require forward-looking insights
- Government and businesses need demand projections

### Key Business Questions:

1. How have accommodation revenues and occupancy rates changed over time?
2. Which accommodation types perform best?
3. Can we forecast future hotel occupancy rates?
4. What do future trends imply for revenue planning and strategy?

This project solves these problems using **statistical time-series modeling**.


## Business Value & Impact

### 1. Revenue Planning

Forecasting occupancy rates helps hotels:

- Predict future demand
- Adjust pricing strategies
- Optimize staffing levels
- Plan seasonal promotions

### 2. Investment Decisions

Investors can:

- Identify growth trends
- Assess market stability
- Evaluate long-term profitability

### 3. Government & Policy Planning

Tourism authorities can:

- Monitor industry performance
- Plan infrastructure investments
- Evaluate economic contribution

### 4. Risk Management

Forecasting reduces uncertainty by:

- Identifying seasonal patterns
- Detecting trend shifts
- Quantifying future ranges (confidence intervals)


##  Dataset Description

**Time Period:** 2007–2024
**Data Source:** Tourism accommodation statistics (Excel dataset)

The dataset includes:

- Income types
- Accommodation categories:

  - Hotels
  - Guest houses & guest farms
  - Caravan parks & camping sites
  - Other accommodation
  - Total industry
- Monthly time-series data
- Revenue values
- Occupancy rates (%)

##  Data Cleaning & Transformation

The dataset required extensive preprocessing:

### Steps Performed: Removed irrelevant columns
- Split combined categorical fields
- Standardized text formatting
-  Extracted currency indicators
- Converted wide-format monthly data to long format
-  Created proper date variables
- Converted values to numeric format
- Engineered separate variables for:

- Currency value
- Occupancy rate
- Number of rooms

This ensures the dataset is analysis-ready and structured for time-series modeling.

##  Exploratory Data Analysis (EDA)

The project includes:

- Trend analysis of mean revenue values
- Occupancy rate comparisons across accommodation types
- Visual insights into industry performance
- Time-based trend visualization

### Key Insights Explored:

- Which accommodation type has the highest occupancy?
- How has occupancy evolved?
- Is there visible seasonality?
- Are trends increasing, decreasing, or stable?

##  Time Series Modeling

To forecast hotel occupancy rates, two models were used:

### 1. ARIMA (AutoRegressive Integrated Moving Average)

- Captures trend
- Handles non-seasonal patterns
- Baseline forecasting model

### 2. SARIMA (Seasonal ARIMA)

- Captures trend + seasonality
- Best suited for monthly tourism data
- More realistic for the hospitality industry forecasting

##  Model Evaluation

The dataset was split into:

- 80% Training Data
- 20% Testing Data

Models were evaluated using:

- **RMSE (Root Mean Squared Error)**
- **MAE (Mean Absolute Error)**
- **MAPE (Mean Absolute Percentage Error)**

Lower values indicate better performance.

This ensures the model is not overfit and performs well on unseen data.

##  Forecasting Results

The final SARIMA model was used to generate:

###  5-Year Forecast (60 Months)

Outputs include:

- Forecasted occupancy rates
- 80% Confidence Interval
- 95% Confidence Interval

This provides:

- Expected trend direction
- Upper and lower risk bounds
- Business uncertainty range


##  Visualizations Included

- Revenue trends by accommodation type
- Mean occupancy comparison (bar chart)
- Train vs test split visualization
- ARIMA vs SARIMA comparison
- Full 5-year forecast with confidence intervals

All plots are created using **ggplot2** with professional styling.

##  Key Technical Skills Demonstrated

- Data wrangling with `dplyr`
- Reshaping data with `tidyr`
- Date manipulation with `lubridate`
- Time series creation using `ts()`
- ARIMA & SARIMA modeling (`forecast` package)
- Model selection using `auto.arima()`
- Forecast evaluation
- Statistical validation
- Confidence interval interpretation
- Business-oriented analytics reporting

##  Results Interpretation (Business Perspective)

The forecast helps answer:

- Is hotel demand expected to grow?
- Are we entering a decline period?
- What is the expected occupancy range?
- Should capacity expansion be considered?

The confidence intervals allow stakeholders to:

- Prepare for best-case scenarios
- Plan for worst-case scenarios
- Make data-driven strategic decisions


## Project Structure

```
📁 Project Folder
 ├── Tourist accomodation From 2007.xlsx
 ├── Forecasting_Tourism.Rmd
 ├── README.md
 └── Output Figures
```

---

## How to Run This Project

1. Clone the repository
2. Install required R packages:

```r
install.packages(c("readxl","dplyr","tidyr","lubridate",
                   "ggplot2","forecast","tseries","knitr"))
```

3. Open the `.Rmd` file in RStudio
4. Click **Knit** to generate the HTML/PDF report


##  Why This Project Stands Out

- Real economic dataset
- End-to-end data pipeline
- Advanced time-series modeling
- Business interpretation included
- Confidence interval forecasting
- Model comparison
- Professional visualization
- Industry-relevant use case

This is not just a coding project — it is a **decision-support analytics project**.

## Future Improvements

- Add Prophet modeling
- Include external variables (GDP, exchange rate, inflation)
- Build an interactive dashboard (Shiny)
- Compare additional models (ETS, LSTM)
- Deploy forecast as a web app
- Automate monthly updates







