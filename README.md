# 🏨 Tourist Accommodation in South Africa — Time Series Analysis & Forecasting

> **Forecasting hotel occupancy rates using SARIMA modelling on 18+ years of Statistics South Africa survey data.**

---

## 📌 Project Overview

South Africa's tourism sector is a critical economic pillar — yet the COVID-19 pandemic shattered its accommodation industry almost overnight. This project uses **official Statistics South Africa Tourist Accommodation Survey data (2007–2025)** to analyse performance trends, quantify the pandemic's impact, and forecast hotel occupancy rates through 2030 using SARIMA time series modelling.

| 📋 Detail | Value |
|---|---|
| **Data Source** | Statistics South Africa — Tourist Accommodation Survey |
| **Time Span** | January 2007 → June 2025 *(18.5 years)* |
| **Total Observations** | 16,650 records |
| **Accommodation Types** | 5 *(Hotels, Guest Houses & Guest Farms, Caravan Parks & Camping Sites, Other, Total Industry)* |
| **Metric Categories** | 8 *(Occupancy, Income, Rooms, Revenue, etc.)* |
| **Primary Model** | SARIMA (Seasonal ARIMA) |
| **Language** | R |

---

## 🧹 Data Cleaning & Preparation Pipeline

The raw dataset required significant restructuring before any analysis could begin. Below is a step-by-step breakdown of every transformation applied.

### Step 1 — Column Removal
Irrelevant metadata columns (`H01`, `H02`, `H03`, `H15`, `H16`, `H25`, `_LABEL_`) were dropped. These contained administrative identifiers with no analytical value.

### Step 2 — Column Splitting
The `H04` column combined two pieces of information — **Income Type** and **Accommodation Type** — in a single string separated by ` - `. This was split into two clean columns using `tidyr::separate()`.

```r
separate(H04,
         into = c("IncomeType", "AccommodationType"),
         sep = " - ",
         extra = "merge")
```

### Step 3 — Unit/Measure Type Parsing
The `H17` column contained free-text measurement descriptions (e.g., *"R million"*, *"Percentage"*, *"Thousand"*). Rather than keeping this as unstructured text, three new binary flag columns were engineered:

| New Column | Captures |
|---|---|
| `Currency` | Values in *R million* or *Rands* |
| `Rate` | Values expressed as *Percentage* |
| `NumberCount` | Counts expressed in *Thousands* |

### Step 4 — Wide to Long Pivot
The raw data stored monthly values in **separate wide columns** (e.g., `MO200701`, `MO200702`, …). Using `tidyr::pivot_longer()`, all 222 monthly columns were collapsed into two columns: `MonthYear` (string) and `Value` (numeric).

```r
pivot_longer(
  cols = starts_with("MO"),
  names_to = "MonthYear",
  values_to = "Value"
)
```

This transformed the dataset from a wide format into a **tidy long format**, enabling time series analysis and ggplot visualisation.

### Step 5 — Date Parsing
The `MonthYear` string (e.g., `"MO200701"`) was parsed into a proper `Date` object:
- Characters 3–4 extracted as `Month`
- Characters 5–8 extracted as `Year`
- Combined into `TimePeriod` using `as.Date(paste0(Year, "-", Month, "-01"))`

### Step 6 — Value Disaggregation
The single `Value` column was split into **three purpose-specific numeric columns**, using `ifelse()` on the measurement flags:

| Column | Populated When |
|---|---|
| `CurrencyValue` | Record is a monetary income value |
| `PercentageValue` | Record is an occupancy rate |
| `NumberOfRooms` | Record is a room/unit count |

This separation prevents mixing incompatible units in downstream calculations and makes filtering trivial (e.g., `filter(PercentageValue > 0)` instantly isolates all occupancy records).

---

## 📊 Exploratory Analysis — Key Findings

### 🏨 Hotel Occupancy: Before vs. During vs. After COVID-19

| Period | Avg. Hotel Occupancy Rate |
|---|---|
| **Pre-COVID (2007–2019)** | **48.9%** |
| **During COVID (2020)** | **18.5%** *(min: 1.5%)* |
| **Post-COVID Recovery (2022–2025)** | **41.8%** |

> The pandemic caused a **-62% collapse** in hotel occupancy in 2020. By 2022–2025, recovery reached ~85% of pre-COVID levels, driven primarily by domestic tourism.

### 🏠 Average Occupancy Rate by Accommodation Type

| Accommodation Type | Avg. Occupancy (%) |
|---|---|
| 🥇 Hotels | **44.5%** |
| 🥈 Total Industry | **41.0%** |
| 🥉 Other Accommodation | **39.1%** |
| Guest Houses & Guest Farms | **38.3%** |
| Caravan Parks & Camping Sites | **24.9%** |

### 💰 Average Monthly Income by Accommodation Type (R million)

| Accommodation Type | Avg. Monthly Income |
|---|---|
| Total Industry | **R 1,598.4m** |
| Hotels | **R 1,205.3m** |
| Other Accommodation | **R 592.6m** |
| Guest Houses & Guest Farms | **R 218.1m** |
| Caravan Parks & Camping Sites | **R 93.3m** |

> Hotels generate **12.9× more income** than caravan parks on average, commanding the largest share of accommodation revenue in South Africa.

---

## 🤖 Modelling — ARIMA vs. SARIMA

The hotel occupancy time series was split **80/20 (train/test)** — approximately 177 months for training and 45 months for testing.

| Metric | ARIMA | SARIMA |
|---|---|---|
| RMSE | Higher | Lower ✅ |
| MAE | Higher | Lower ✅ |
| **MAPE** | **28.38%** | **6.17%** ✅ |

**SARIMA outperformed ARIMA by 78% in MAPE**, confirming that capturing seasonality is essential for accommodation data — which exhibits strong annual cycles tied to school holidays, summer travel, and December peaks.

### 📈 5-Year Forecast (2025–2030)
The final SARIMA model trained on the full dataset projects:
- **Stable occupancy between 43–46%** through 2030
- Narrow confidence intervals indicating high forecast reliability
- Continued gradual recovery trajectory with preserved seasonal patterns

---

## 📉 Visualisations

> All charts below are produced by the R code in this repository.

### Figure 1 — Average Income by Accommodation Type
![Average Income by Accommodation Type](plots/avg_income_by_type.png)

### Figure 2 — Average Occupancy Rate by Accommodation Type
![Average Occupancy Rate](plots/avg_occupancy_by_type.png)

### Figure 3 — Hotel Occupancy Time Series (2007–2025)
![Hotel Occupancy Time Series](plots/hotel_occupancy_timeseries.png)

### Figure 4 — ARIMA vs SARIMA Forecast Comparison
![Forecast Comparison](plots/arima_vs_sarima_forecast.png)

### Figure 5 — SARIMA 5-Year Forecast (2025–2030)
![5-Year SARIMA Forecast](plots/sarima_5yr_forecast.png)

### Figure 6 — Average Rooms by Accommodation Type
![Average Rooms](plots/avg_rooms_by_type.png)

---

## 💼 Business Impact

| Insight | Impact |
|---|---|
| **COVID-19 quantified** | Occupancy collapsed from 48.9% to 1.5% — the most severe disruption in 18 years of data |
| **Recovery benchmarked** | Post-2022 occupancy at 41.8% gives tourism operators a concrete baseline for investment planning |
| **Forecast precision** | MAPE of 6.17% means the SARIMA model is accurate enough for **budget planning, staffing models, and capacity decisions** |
| **Segment differentiation** | Hotels consistently outperform guest houses by ~6 percentage points — informing where marketing and infrastructure investment yields the highest return |
| **Policy evidence** | Narrow forecast confidence intervals support Department of Tourism interventions targeting the remaining 7–8% recovery gap to pre-COVID levels |

---

## 🗂️ Repository Structure

```
📦 tourist-accommodation-sa/
├── 📄 README.md
├── 📊 Tourist_Accommodation_Analysis.R    ← Main analysis script
├── 📁 data/
│   ├── Tourist_A_long_clean.csv           ← Cleaned long-format dataset
│   └── Tourist accomodation From 2007 SEP.xlsx  ← Raw source data
├── 📁 plots/                              ← Generated chart outputs
│   ├── avg_income_by_type.png
│   ├── avg_occupancy_by_type.png
│   ├── avg_rooms_by_type.png
│   ├── hotel_occupancy_timeseries.png
│   ├── arima_vs_sarima_forecast.png
│   └── sarima_5yr_forecast.png
└── 📄 FINAL_TOURIST_ACCOMMODATION_Report.docx  ← Full research report
```

---

## ▶️ How to Run

### Prerequisites

```r
install.packages(c(
  "readxl", "dplyr", "tidyr", "stringr",
  "lubridate", "ggplot2", "forecast", "zoo", "knitr"
))
```

### Run the Analysis

```r
# Clone the repo, open R, set working directory, then:
source("Tourist_Accommodation_Analysis.R")
```

All plots are saved automatically to the `plots/` folder and displayed in the RStudio viewer.

---

## 📊 Full Visualisation Code (with `ggsave` Export)

The code below extends the original analysis with **additional visuals** and automatically exports all charts as `.png` files.

```r
# ============================================================
# EXTENDED VISUALISATION SCRIPT
# Tourist Accommodation South Africa — Full Plot Suite
# ============================================================

library(readxl); library(dplyr); library(tidyr); library(stringr)
library(lubridate); library(ggplot2); library(forecast); library(zoo)

# ── 0. Setup ────────────────────────────────────────────────
dir.create("plots", showWarnings = FALSE)

theme_sa <- theme_minimal(base_size = 13) +
  theme(
    plot.title    = element_text(face = "bold", size = 15, hjust = 0.5),
    plot.subtitle = element_text(hjust = 0.5, colour = "grey40"),
    axis.text.x   = element_text(angle = 45, hjust = 1),
    plot.caption  = element_text(colour = "grey50", size = 9)
  )

# ── 1. Load Cleaned Data ────────────────────────────────────
Tourist_A_long_clean <- read.csv("data/Tourist_A_long_clean.csv",
                                  stringsAsFactors = FALSE) %>%
  mutate(TimePeriod = as.Date(TimePeriod))

# ── 2. FIGURE 1: Average Income by Accommodation Type ───────
avg_income <- Tourist_A_long_clean %>%
  filter(CurrencyValue > 0) %>%
  group_by(AccommodationType) %>%
  summarise(AvgIncome = mean(CurrencyValue, na.rm = TRUE)) %>%
  arrange(desc(AvgIncome))

p1 <- ggplot(avg_income,
             aes(x = reorder(AccommodationType, AvgIncome),
                 y = AvgIncome, fill = AvgIncome)) +
  geom_col(show.legend = FALSE) +
  geom_text(aes(label = paste0("R ", round(AvgIncome, 1), "m")),
            hjust = -0.1, size = 3.8) +
  scale_fill_gradient(low = "#81c784", high = "#1b5e20") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.2))) +
  coord_flip() +
  labs(
    title    = "Average Monthly Income by Accommodation Type",
    subtitle = "Statistics South Africa Tourist Accommodation Survey · 2007–2025",
    x = NULL, y = "Average Income (R million)",
    caption  = "Source: Stats SA | Analysis: Mathebula S.K."
  ) +
  theme_sa

ggsave("plots/avg_income_by_type.png", p1, width = 10, height = 5, dpi = 150)
print(p1)

# ── 3. FIGURE 2: Average Occupancy Rate by Accommodation Type
avg_occupancy <- Tourist_A_long_clean %>%
  filter(PercentageValue > 0) %>%
  group_by(AccommodationType) %>%
  summarise(AvgOccupancy = mean(PercentageValue, na.rm = TRUE)) %>%
  arrange(desc(AvgOccupancy))

p2 <- ggplot(avg_occupancy,
             aes(x = reorder(AccommodationType, AvgOccupancy),
                 y = AvgOccupancy, fill = AvgOccupancy)) +
  geom_col(show.legend = FALSE) +
  geom_text(aes(label = paste0(round(AvgOccupancy, 1), "%")),
            hjust = -0.15, size = 3.8) +
  scale_fill_gradient(low = "#64b5f6", high = "#0d47a1") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.2))) +
  coord_flip() +
  labs(
    title    = "Average Occupancy Rate by Accommodation Type",
    subtitle = "Statistics South Africa Tourist Accommodation Survey · 2007–2025",
    x = NULL, y = "Average Occupancy Rate (%)",
    caption  = "Source: Stats SA | Analysis: Mathebula S.K."
  ) +
  theme_sa

ggsave("plots/avg_occupancy_by_type.png", p2, width = 10, height = 5, dpi = 150)
print(p2)

# ── 4. FIGURE 3: Average Rooms by Accommodation Type ────────
avg_rooms <- Tourist_A_long_clean %>%
  filter(NumberOfRooms > 0) %>%
  group_by(AccommodationType) %>%
  summarise(AvgRooms = mean(NumberOfRooms, na.rm = TRUE)) %>%
  arrange(desc(AvgRooms))

p3 <- ggplot(avg_rooms,
             aes(x = reorder(AccommodationType, AvgRooms),
                 y = AvgRooms, fill = AvgRooms)) +
  geom_col(show.legend = FALSE) +
  geom_text(aes(label = paste0(round(AvgRooms, 1), "k")),
            hjust = -0.15, size = 3.8) +
  scale_fill_gradient(low = "#ce93d8", high = "#4a148c") +
  scale_y_continuous(expand = expansion(mult = c(0, 0.2))) +
  coord_flip() +
  labs(
    title    = "Average Number of Stay Units by Accommodation Type",
    subtitle = "Statistics South Africa Tourist Accommodation Survey · 2007–2025",
    x = NULL, y = "Average Stay Units (Thousands)",
    caption  = "Source: Stats SA | Analysis: Mathebula S.K."
  ) +
  theme_sa

ggsave("plots/avg_rooms_by_type.png", p3, width = 10, height = 5, dpi = 150)
print(p3)

# ── 5. FIGURE 4: Hotel Occupancy Time Series (annotated) ────
Hotel_Occupancy <- Tourist_A_long_clean %>%
  filter(
    grepl("hotel", AccommodationType, ignore.case = TRUE),
    grepl("occupancy",  IncomeType,         ignore.case = TRUE),
    PercentageValue > 0
  ) %>%
  arrange(TimePeriod)

p4 <- ggplot(Hotel_Occupancy, aes(x = TimePeriod, y = PercentageValue)) +
  geom_line(colour = "#1565c0", linewidth = 0.8) +
  geom_smooth(method = "loess", se = FALSE, colour = "#e53935",
              linetype = "dashed", linewidth = 0.8) +
  annotate("rect",
           xmin = as.Date("2020-03-01"), xmax = as.Date("2021-12-01"),
           ymin = -Inf, ymax = Inf,
           fill = "#ef9a9a", alpha = 0.3) +
  annotate("text", x = as.Date("2020-09-01"), y = 58,
           label = "COVID-19\nLockdowns", size = 3.2,
           colour = "#c62828", fontface = "bold") +
  scale_y_continuous(limits = c(0, 70),
                     labels = function(x) paste0(x, "%")) +
  scale_x_date(date_breaks = "2 years", date_labels = "%Y") +
  labs(
    title    = "Hotel Occupancy Rate · South Africa (2007–2025)",
    subtitle = "Red dashed line = LOESS trend · Shaded area = COVID-19 disruption period",
    x = "Year", y = "Occupancy Rate (%)",
    caption  = "Source: Stats SA | Analysis: Mathebula S.K."
  ) +
  theme_sa

ggsave("plots/hotel_occupancy_timeseries.png", p4,
       width = 12, height = 5, dpi = 150)
print(p4)

# ── 6. Build Time Series & Models ───────────────────────────
hotel_ts <- ts(
  Hotel_Occupancy$PercentageValue,
  start     = c(year(min(Hotel_Occupancy$TimePeriod)),
                month(min(Hotel_Occupancy$TimePeriod))),
  frequency = 12
)

train_size   <- round(length(hotel_ts) * 0.8)
train_ts     <- window(hotel_ts, end   = time(hotel_ts)[train_size])
test_ts      <- window(hotel_ts, start = time(hotel_ts)[train_size + 1])

arima_model  <- auto.arima(train_ts, seasonal = FALSE)
sarima_model <- auto.arima(train_ts, seasonal = TRUE)

arima_fc  <- forecast(arima_model,  h = length(test_ts))
sarima_fc <- forecast(sarima_model, h = length(test_ts))

# ── 7. FIGURE 5: ARIMA vs SARIMA on Test Data ───────────────
test_dates <- seq(
  from = as.Date(paste0(
    floor(time(test_ts)[1]), "-",
    round((time(test_ts)[1] %% 1) * 12 + 1), "-01")),
  by = "month", length.out = length(test_ts)
)

comp_df <- data.frame(
  Date   = test_dates,
  Actual = as.numeric(test_ts),
  ARIMA  = as.numeric(arima_fc$mean),
  SARIMA = as.numeric(sarima_fc$mean)
) %>%
  tidyr::pivot_longer(cols = -Date, names_to = "Series", values_to = "Value")

comp_df$Series <- factor(comp_df$Series,
                          levels = c("Actual", "SARIMA", "ARIMA"))

p5 <- ggplot(comp_df, aes(x = Date, y = Value,
                           colour = Series, linetype = Series)) +
  geom_line(linewidth = 1) +
  scale_colour_manual(values = c(Actual = "#212121",
                                  SARIMA = "#1565c0",
                                  ARIMA  = "#e53935")) +
  scale_linetype_manual(values = c(Actual = "solid",
                                    SARIMA = "solid",
                                    ARIMA  = "dashed")) +
  scale_y_continuous(labels = function(x) paste0(x, "%")) +
  labs(
    title    = "Model Comparison: Actual vs ARIMA vs SARIMA",
    subtitle = "Test set evaluation · SARIMA MAPE = 6.17% · ARIMA MAPE = 28.38%",
    x = "Date", y = "Occupancy Rate (%)", colour = NULL, linetype = NULL,
    caption  = "Source: Stats SA | Analysis: Mathebula S.K."
  ) +
  theme_sa +
  theme(legend.position = "top")

ggsave("plots/arima_vs_sarima_forecast.png", p5,
       width = 12, height = 5, dpi = 150)
print(p5)

# ── 8. FIGURE 6: 5-Year SARIMA Forecast ─────────────────────
final_sarima <- auto.arima(hotel_ts, seasonal = TRUE)
fc_5yr       <- forecast(final_sarima, h = 60)

p6 <- autoplot(fc_5yr) +
  autolayer(hotel_ts, series = "Historical Data", colour = "#1565c0") +
  scale_y_continuous(labels = function(x) paste0(x, "%")) +
  scale_x_continuous(breaks = seq(2007, 2031, by = 2)) +
  labs(
    title    = "Hotel Occupancy Forecast · South Africa (2025–2030)",
    subtitle = "SARIMA model · 80% & 95% prediction intervals shown",
    x = "Year", y = "Occupancy Rate (%)",
    caption  = "Source: Stats SA | Analysis: Mathebula S.K."
  ) +
  theme_sa +
  theme(legend.position = "top")

ggsave("plots/sarima_5yr_forecast.png", p6, width = 12, height = 5, dpi = 150)
print(p6)

# ── 9. BONUS: Pre vs Post COVID Comparison ──────────────────
covid_df <- Hotel_Occupancy %>%
  mutate(
    Period = case_when(
      TimePeriod < as.Date("2020-03-01") ~ "Pre-COVID\n(2007–Feb 2020)",
      TimePeriod < as.Date("2022-01-01") ~ "COVID Period\n(Mar 2020–Dec 2021)",
      TRUE                               ~ "Post-COVID\n(2022–2025)"
    ),
    Period = factor(Period, levels = c(
      "Pre-COVID\n(2007–Feb 2020)",
      "COVID Period\n(Mar 2020–Dec 2021)",
      "Post-COVID\n(2022–2025)"
    ))
  )

p7 <- ggplot(covid_df, aes(x = Period, y = PercentageValue, fill = Period)) +
  geom_boxplot(alpha = 0.8, outlier.colour = "grey40", outlier.size = 1.5) +
  stat_summary(fun = mean, geom = "point", shape = 23,
               size = 4, fill = "white", colour = "black") +
  scale_fill_manual(values = c("#42a5f5", "#ef5350", "#66bb6a")) +
  scale_y_continuous(labels = function(x) paste0(x, "%")) +
  labs(
    title    = "Hotel Occupancy Distribution: Pre vs During vs Post COVID-19",
    subtitle = "Diamond = mean · Box = interquartile range",
    x = NULL, y = "Occupancy Rate (%)",
    caption  = "Source: Stats SA | Analysis: Mathebula S.K."
  ) +
  theme_sa +
  theme(legend.position = "none")

ggsave("plots/covid_period_comparison.png", p7, width = 10, height = 5, dpi = 150)
print(p7)

message("✅ All 7 plots saved to /plots/")
```

---

## 📚 References

- Statistics South Africa. (2024). *Tourist Accommodation Survey*. [stats.gov.za](https://www.statssa.gov.za)
- UNWTO. (2014). *International Recommendations for Tourism Statistics*
- Hyndman, R.J. & Athanasopoulos, G. (2021). *Forecasting: Principles and Practice* (3rd ed.)
- Department of Tourism, South Africa. (2021). *National Tourism Recovery Plan*

---

## 👩‍💻 Author

**Mathebula Salphina Katlego**  
Diploma in Mathematical Sciences — Cape Peninsula University of Technology  
Module: Mathematical Sciences Project 3 (MSP360S) · October 2025  
Supervisor: Mr. M. Nombela

---

*Built with R · Stats SA data · SARIMA modelling*
