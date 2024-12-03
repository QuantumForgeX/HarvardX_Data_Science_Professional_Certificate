# title: "Flight Price Prediction Project"
# author: "Kagan Heper"
# date: "December 2024"
# output: pdf_document

packages <- c("tidyverse", "caret", "randomForest", "xgboost")

for (pkg in packages) {
  if (!require(pkg, character.only = TRUE)) {
    install.packages(pkg, dependencies = TRUE)
    library(pkg, character.only = TRUE)
  }
}

# URL of the dataset
url <- "https://raw.githubusercontent.com/QuantumForgeX/HarvardX_Data_Science_Professional_Certificate/24fa4dda6efd0f9ce6e7df7f95b1e2c087a4a67b/Clean_Dataset.csv"

# Load the dataset directly into R
dataset <- read.csv(url, stringsAsFactors = FALSE)

# Replace commas in 'price' and convert to numeric
dataset$price <- as.numeric(gsub(",", "", dataset$price))

# Convert categorical variables to factors
categorical_cols <- c("airline", "source_city", "destination_city", 
                      "departure_time", "arrival_time", "class", "stops")
dataset[categorical_cols] <- lapply(dataset[categorical_cols], as.factor)

head(dataset)

# Summary of the dataset
summary(dataset)

# Distribution of Flight Prices
ggplot(dataset, aes(x = price)) +
  geom_histogram(fill = "steelblue", color = "white", bins = 30) +
  theme_minimal() +
  labs(title = "Distribution of Flight Prices", x = "Price", y = "Count")

# Average Prices by Airline
ggplot(dataset %>% group_by(airline) %>% summarise(avg_price = mean(price, na.rm = TRUE)), 
       aes(x = reorder(airline, avg_price), y = avg_price, fill = airline)) +
  geom_bar(stat = "identity", show.legend = FALSE) +
  theme_minimal() +
  labs(title = "Average Flight Prices by Airline", x = "Airline", y = "Average Price") +
  coord_flip()

# Number of Flights by Source City
ggplot(dataset, aes(x = source_city)) +
  geom_bar(fill = "steelblue") +
  theme_minimal() +
  labs(title = "Number of Flights by Source City", x = "Source City", y = "Count")

# Number of Flights by Departure Time
ggplot(dataset, aes(x = departure_time)) +
  geom_bar(fill = "steelblue") +
  theme_minimal() +
  labs(title = "Number of Flights by Departure Time", x = "Departure Time", y = "Count") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Flight Prices by Class
ggplot(dataset, aes(x = class, y = price, fill = class)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Flight Prices by Class", x = "Class", y = "Price")

# Flight Prices by Stops
ggplot(dataset, aes(x = stops, y = price, fill = stops)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Flight Prices by Stops", x = "Stops", y = "Price")

# Distribution of Flight Duration
ggplot(dataset, aes(x = duration)) +
  geom_histogram(fill = "steelblue", color = "white", bins = 30) +
  theme_minimal() +
  labs(title = "Distribution of Flight Duration (Hours)", x = "Duration (Hours)", y = "Count")

# Distribution of Days Left
ggplot(dataset, aes(x = days_left)) +
  geom_histogram(fill = "steelblue", color = "white", bins = 30) +
  theme_minimal() +
  labs(title = "Distribution of Days Left", x = "Days Left", y = "Count")

# Flight Prices by Source City
ggplot(dataset, aes(x = source_city, y = price, fill = source_city)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Flight Prices by Source City", x = "Source City", y = "Price") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Flight Prices by Destination City
ggplot(dataset, aes(x = destination_city, y = price, fill = destination_city)) +
  geom_boxplot() +
  theme_minimal() +
  labs(title = "Flight Prices by Destination City", x = "Destination City", y = "Price") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Flights by Departure Time
ggplot(dataset, aes(x = departure_time, fill = departure_time)) +
  geom_bar() +
  theme_minimal() +
  labs(title = "Flights by Departure Time", x = "Departure Time", y = "Count") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Flights by Arrival Time
ggplot(dataset, aes(x = arrival_time, fill = arrival_time)) +
  geom_bar() +
  theme_minimal() +
  labs(title = "Flights by Arrival Time", x = "Arrival Time", y = "Count") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Relationship Between Flight Duration and Price
ggplot(dataset, aes(x = duration, y = price, color = airline)) +
  geom_point(alpha = 0.6) +
  theme_minimal() +
  labs(title = "Relationship Between Flight Duration and Price", x = "Duration (Hours)", 
       y = "Price", color = "Airline")

# Relationship Between Days Left and Price
ggplot(dataset, aes(x = days_left, y = price)) +
  geom_point(alpha = 0.6, color = "steelblue") +
  theme_minimal() +
  labs(title = "Relationship Between Days Left and Price", x = "Days Left", y = "Price")

# Flights by Source City and Departure Time
ggplot(dataset, aes(x = source_city, fill = departure_time)) +
  geom_bar(position = "dodge", color = "white") +
  theme_minimal() +
  labs(title = "Flights by Source City and Departure Time", x = "Source City", y = "Count", 
       fill = "Departure Time") +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

# Splitting Data
set.seed(123)
train_index <- createDataPartition(dataset$price, p = 0.8, list = FALSE)
train_set <- dataset[train_index, ]
test_set <- dataset[-train_index, ]

# Linear Regression
# Individual models are fitted
fit_source_city <- lm(price ~ source_city, data = train_set)
fit_departure_time <- lm(price ~ departure_time, data = train_set)
fit_stops <- lm(price ~ stops, data = train_set)
fit_arrival_time <- lm(price ~ arrival_time, data = train_set)
fit_destination_city <- lm(price ~ destination_city, data = train_set)
fit_class <- lm(price ~ class, data = train_set)
fit_duration <- lm(price ~ duration, data = train_set)
fit_days_left <- lm(price ~ days_left, data = train_set)

# Multivariate Linear Regression
# Fit multivariate linear regression
fit_multivariate <- lm(price ~ source_city + departure_time + stops + arrival_time + 
                         destination_city + class + duration + days_left, data = train_set)
y_hat_multivariate <- predict(fit_multivariate, test_set)

# Evaluate performance
mult_rmse <- RMSE(y_hat_multivariate, test_set$price)
mult_mae <- MAE(y_hat_multivariate, test_set$price)
mult_r2 <- R2(y_hat_multivariate, test_set$price)

# Random Forest
rf_model <- randomForest(price ~ source_city + departure_time + stops + arrival_time 
                         + destination_city + class + duration + days_left, data = train_set, 
                         ntree = 50, maxnodes = 20)
rf_pred <- predict(rf_model, test_set)

# Evaluate performance
rf_rmse <- RMSE(rf_pred, test_set$price)
rf_mae <- MAE(rf_pred, test_set$price)
rf_r2 <- R2(rf_pred, test_set$price)

# XGBoost
# Prepare data for XGBoost
xgb_train <- xgb.DMatrix(data = model.matrix(~ source_city + departure_time + stops + 
                                               arrival_time + destination_city + class + 
                                               duration + days_left - 1, data = train_set), 
                         label = train_set$price)
xgb_test <- xgb.DMatrix(data = model.matrix(~ source_city + departure_time + stops + 
                                              arrival_time + destination_city + class + 
                                              duration + days_left - 1, data = test_set))

# Train XGBoost Model
xgb_model <- xgboost(data = xgb_train, max_depth = 4, eta = 0.1, nrounds = 50, 
                     objective = "reg:squarederror", verbose = 0)
xgb_pred <- predict(xgb_model, xgb_test)

# Evaluate performance
xgb_rmse <- RMSE(xgb_pred, test_set$price)
xgb_mae <- MAE(xgb_pred, test_set$price)
xgb_r2 <- R2(xgb_pred, test_set$price)

# Results Comparison
# Compile results into a table
results <- data.frame(
  Model = c("Multivariate Linear Regression", "Random Forest", "XGBoost"),
  RMSE = c(mult_rmse, rf_rmse, xgb_rmse),
  MAE = c(mult_mae, rf_mae, xgb_mae),
  R2 = c(mult_r2, rf_r2, xgb_r2)
)

# Display results
knitr::kable(results, caption = "Performance Comparison of Models")

