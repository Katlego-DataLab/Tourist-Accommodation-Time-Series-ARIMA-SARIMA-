# 1. LOAD REQUIRED LIBRARIES

library(readxl)
library(dplyr)
library(tidyr)
library(stringr)
library(lubridate)
library(ggplot2)
library(forecast)
library(zoo)
library(knitr)


# 2. LOAD DATA 


Tourist_A <- read_excel("Tourist accomodation From 2007 SEP.xlsx")
view(Tourist_A)

# 3.  INITIAL DATA CLEANING


Tourist_A_clean <- Tourist_A %>%
  
  # Remove unnecessary columns 
  select(-any_of(c("H01","H02","H03","H15","H16","H25","_LABEL_"))) %>%
  
  # Split H04 column
  separate(H04,
           into = c("IncomeType", "AccommodationType"),
           sep = " - ",
           extra = "merge") %>%
  
  mutate(
    IncomeType = str_trim(IncomeType),
    AccommodationType = str_trim(AccommodationType),
    
    Currency = ifelse(str_detect(H17, "R\\s?million|Rands"),
                      str_extract(H17, "R\\s?million|Rands"), NA),
    
    Rate = ifelse(str_detect(H17, "Percentage|%"),
                  "Percentage", NA),
    
    NumberCount = ifelse(str_detect(H17, "Thousand"),
                         "Thousand", NA)
  ) %>%
  
  select(-H17)


# 4. PIVOT DATA TO LONG FORMAT


monthly_cols <- grep("^MO\\d{6}$",
                     names(Tourist_A_clean),
                     value = TRUE)

Tourist_A_long <- Tourist_A_clean %>%
  
  select(IncomeType, AccommodationType,
         Currency, Rate, NumberCount,
         all_of(monthly_cols)) %>%
  
  pivot_longer(
    cols = starts_with("MO"),
    names_to = "MonthYear",
    values_to = "Value"
  ) %>%
  
  mutate(
    Month = substr(MonthYear, 3, 4),
    Year  = substr(MonthYear, 5, 8),
    TimePeriod = as.Date(paste0(Year, "-", Month, "-01")),
    Value = as.numeric(Value)
  ) %>%
  
  select(-MonthYear, -Month, -Year)


# 5.  CREATE FINAL CLEAN DATASET

Tourist_A_long_clean <- Tourist_A_long %>%
  
  mutate(
    CurrencyValue   = ifelse(!is.na(Currency), Value, 0),
    PercentageValue = ifelse(!is.na(Rate), Value, 0),
    NumberOfRooms   = ifelse(!is.na(NumberCount), Value, 0),
    NumberType = coalesce(Currency, Rate, NumberCount)
  ) %>%
  
  select(IncomeType, AccommodationType, NumberType,
         TimePeriod, CurrencyValue,
         PercentageValue, NumberOfRooms)


# 6.  FILTER HOTEL OCCUPANCY DATA

Hotel_Occupancy <- Tourist_A_long_clean %>%
  
  filter(
    grepl("hotel", AccommodationType, ignore.case = TRUE),
    grepl("occupancy", IncomeType, ignore.case = TRUE),
    PercentageValue > 0
  ) %>%
  
  arrange(TimePeriod)


# 7.  CREATE TIME SERIES OBJECT


start_year  <- year(min(Hotel_Occupancy$TimePeriod))
start_month <- month(min(Hotel_Occupancy$TimePeriod))

hotel_ts <- ts(
  Hotel_Occupancy$PercentageValue,
  start = c(start_year, start_month),
  frequency = 12
)


# 8.  TRAIN-TEST SPLIT (80/20)

train_size <- round(length(hotel_ts) * 0.8)

train_ts <- window(hotel_ts,
                   end = time(hotel_ts)[train_size])

test_ts <- window(hotel_ts,
                  start = time(hotel_ts)[train_size + 1])


# 9.  FIT ARIMA & SARIMA MODELS


arima_model  <- auto.arima(train_ts, seasonal = FALSE)
sarima_model <- auto.arima(train_ts, seasonal = TRUE)

arima_forecast  <- forecast(arima_model,  h = length(test_ts))
sarima_forecast <- forecast(sarima_model, h = length(test_ts))


# 10.  MODEL ACCURACY COMPARISON

arima_acc  <- accuracy(arima_forecast,  test_ts)
sarima_acc <- accuracy(sarima_forecast, test_ts)

accuracy_table <- data.frame(
  Model = c("ARIMA", "SARIMA"),
  RMSE  = c(arima_acc[2,"RMSE"], sarima_acc[2,"RMSE"]),
  MAE   = c(arima_acc[2,"MAE"],  sarima_acc[2,"MAE"]),
  MAPE  = c(arima_acc[2,"MAPE"], sarima_acc[2,"MAPE"])
)

kable(accuracy_table,
      caption = "Model Comparison (Lower is Better)")


# 11. FIT FINAL SARIMA MODEL (FULL DATA)

final_sarima <- auto.arima(hotel_ts, seasonal = TRUE)

forecast_5yr <- forecast(final_sarima, h = 60)


# 12. PLOT 5-YEAR FORECAST


autoplot(forecast_5yr) +
  autolayer(hotel_ts, series = "Actual Data") +
  labs(
    title = "Hotel Occupancy Rate Forecast (Next 5 Years)",
    x = "Year",
    y = "Occupancy Rate (%)"
  ) +
  theme_minimal()

# 13. BAR GRAPHS FOR ALL ACCOMMODATION TYPES

# 13.1 Average Occupancy Rate by Accommodation Type

avg_occupancy <- Tourist_A_long_clean %>%
  filter(PercentageValue > 0) %>%
  group_by(AccommodationType) %>%
  summarise(AvgOccupancy = mean(PercentageValue, na.rm = TRUE)) %>%
  arrange(desc(AvgOccupancy))

ggplot(avg_occupancy, 
       aes(x = reorder(AccommodationType, AvgOccupancy), 
           y = AvgOccupancy)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(
    title = "Average Occupancy Rate by Accommodation Type",
    x = "Accommodation Type",
    y = "Average Occupancy (%)"
  ) +
  theme_minimal()



# 13.2 Average Income (Currency) by Accommodation Type

avg_income <- Tourist_A_long_clean %>%
  filter(CurrencyValue > 0) %>%
  group_by(AccommodationType) %>%
  summarise(AvgIncome = mean(CurrencyValue, na.rm = TRUE)) %>%
  arrange(desc(AvgIncome))

ggplot(avg_income, 
       aes(x = reorder(AccommodationType, AvgIncome), 
           y = AvgIncome)) +
  geom_bar(stat = "identity", fill = "darkgreen") +
  coord_flip() +
  labs(
    title = "Average Income by Accommodation Type",
    x = "Accommodation Type",
    y = "Average Income (R million)"
  ) +
  theme_minimal()


# 13.3 Average Number of Rooms by Accommodation Type

avg_rooms <- Tourist_A_long_clean %>%
  filter(NumberOfRooms > 0) %>%
  group_by(AccommodationType) %>%
  summarise(AvgRooms = mean(NumberOfRooms, na.rm = TRUE)) %>%
  arrange(desc(AvgRooms))

ggplot(avg_rooms, 
       aes(x = reorder(AccommodationType, AvgRooms), 
           y = AvgRooms)) +
  geom_bar(stat = "identity", fill = "purple") +
  coord_flip() +
  labs(
    title = "Average Number of Rooms by Accommodation Type",
    x = "Accommodation Type",
    y = "Average Number of Rooms (Thousands)"
  ) +
  theme_minimal()

## END OF CODE ##