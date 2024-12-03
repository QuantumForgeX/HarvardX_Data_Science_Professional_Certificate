# HarvardX_Data_Science_Professional_Certificate

# Flight Price Prediction Project

## Overview
This project aims to predict flight ticket prices using machine learning algorithms. Leveraging the dataset `Flight Price Prediction` by Shubham Bathwal, which contains detailed flight information, we applied exploratory data analysis (EDA), feature engineering, and multiple machine learning models to analyze and predict flight prices. The report explores various aspects of the dataset, highlights key insights, and evaluates the performance of three models: **Multivariate Linear Regression**, **Random Forest**, and **XGBoost**.

## Key Features
- **Exploratory Data Analysis (EDA):**
  - Visualizations of flight price distributions, relationships between categorical and numerical features, and average prices across airlines.
  - Analysis of trends such as the impact of flight duration, number of stops, and days left until departure on prices.
- **Machine Learning Models:**
  - **Multivariate Linear Regression**: Provides a baseline prediction model.
  - **Random Forest**: Captures non-linear relationships between features.
  - **XGBoost**: A state-of-the-art algorithm for regression tasks, delivering the best performance.
- **Performance Metrics:**
  - RMSE (Root Mean Square Error), MAE (Mean Absolute Error), and R² are used to evaluate model accuracy.

## Dataset
The dataset `Clean_Dataset.csv` contains flight-related

information with the following columns:

- **airline**: The airline operating the flight.
- **source_city**: The city of departure.
- **destination_city**: The city of arrival.
- **departure_time**: Time of flight departure (e.g., Morning, Afternoon).
- **arrival_time**: Time of flight arrival (e.g., Evening, Night).
- **class**: Ticket class (e.g., Economy, Business).
- **stops**: The number of stops during the flight (e.g., zero, one).
- **duration**: Total flight duration (in minutes).
- **days_left**: Days left until departure.
- **price**: Flight ticket price (target variable).

The dataset is cleaned and preprocessed to handle categorical variables, missing values, and feature engineering to optimize model performance.

## Results
The results of the model evaluations are as follows:

| Model                          | RMSE     | MAE      | R²       |
|--------------------------------|----------|----------|----------|
| Multivariate Linear Regression | 6964.262 | 4656.313 | 0.9059   |
| Random Forest                  | 9824.884 | 7481.765 | 0.9132   |
| XGBoost                        | 5350.473 | 3183.110 | 0.9446   |

The **XGBoost model** outperformed the others with the lowest prediction error and the highest explanatory power, making it the most suitable model for this dataset.

## File Structure
- `FlightPricePrediction.Rmd`: The main R Markdown report, containing the detailed analysis, EDA, and modeling.
- `FlightPricePrediction.pdf`: A rendered PDF version of the report for easy viewing.
- `FlightPricePrediction.R`: The R script containing the code used for data preprocessing, analysis, and modeling.
- `Clean_Dataset.csv`: The dataset used in this project, included for reproducibility.

## How to Run the Project
1. Clone this repository:
   ```bash
   git clone https://github.com/YourUsername/FlightPricePrediction.git
   ```
2. Open `FlightPricePrediction.Rmd` in RStudio.
3. Ensure all required packages are installed:
   ```R
   install.packages(c("tidyverse", "caret", "randomForest", "xgboost", "corrplot", "naniar"))
   ```
4. Knit the `FlightPricePrediction.Rmd` file to generate the report.

Alternatively, run the analysis directly using the `FlightPricePrediction.R` script.

## Conclusion
This project demonstrates how machine learning can effectively predict flight prices based on various features. **XGBoost** proved to be the best-performing model, but other algorithms like **Random Forest** and **Linear Regression** provide valuable baselines. Future work could focus on further feature engineering, hyperparameter tuning, and incorporating additional data sources to enhance prediction accuracy.

## License
This project is licensed under the MIT License. You are free to use, modify, and distribute this project, provided proper attribution is given.
