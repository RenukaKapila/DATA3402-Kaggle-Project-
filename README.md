# Airline Passenger Satisfaction Prediction

## Project Overview

This project uses the **Airline Passenger Satisfaction** dataset from Kaggle to predict whether a passenger is **satisfied** or **neutral or dissatisfied** with their airline experience.

The main goal of this project is not only to build a classification model, but also to understand which controllable airline service factors have the strongest relationship with passenger dissatisfaction. These factors include online boarding, inflight wifi service, seat comfort, inflight entertainment, cleanliness, food and drink, leg room service, and delay-related features.

## Kaggle Dataset

Dataset link:  
https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction?resource=download&select=train.csv

## Problem Type

This is a **binary classification** problem.

The target variable is:

- `neutral or dissatisfied`
- `satisfied`

For machine learning, the target variable was encoded as:

- `0 = neutral or dissatisfied`
- `1 = satisfied`

## Dataset Description

The dataset contains airline passenger survey information. It includes passenger details, travel information, service ratings, and delay-related variables.

Some of the main features include:

- Gender
- Customer Type
- Age
- Type of Travel
- Class
- Flight Distance
- Inflight wifi service
- Departure/Arrival time convenient
- Ease of Online booking
- Gate location
- Food and drink
- Online boarding
- Seat comfort
- Inflight entertainment
- On-board service
- Leg room service
- Baggage handling
- Checkin service
- Inflight service
- Cleanliness
- Departure Delay in Minutes
- Arrival Delay in Minutes
- Satisfaction

The dataset originally included an unnecessary index column called `Unnamed: 0`, which was removed because it did not provide useful information for prediction.

## Data Summary

The training dataset contains:

| Item | Value |
|---|---:|
| Rows | 103,904 |
| Columns before cleaning | 25 |
| Columns after removing `Unnamed: 0` | 24 |
| Target variable | satisfaction |
| Problem type | Binary classification |

Most columns had no missing values. The only column with missing data was:

| Column | Missing Values |
|---|---:|
| Arrival Delay in Minutes | 310 |

The missing values in `Arrival Delay in Minutes` were handled before machine learning.

## Project Goals

The main goals of this project were:

1. Load and inspect the airline passenger satisfaction dataset.
2. Identify categorical and numerical features.
3. Check for missing values and outliers.
4. Visualize feature distributions by satisfaction class.
5. Determine which features appear most connected to satisfaction.
6. Prepare the data for machine learning.
7. Train classification models.
8. Compare model performance.
9. Identify the most important predictors of passenger satisfaction.

## Exploratory Data Analysis

The visualization section focused on comparing the two satisfaction groups: **neutral or dissatisfied** and **satisfied**.

The visualizations included:

- Target class distribution
- Numerical feature distributions
- Categorical feature comparisons
- Service rating gap analysis
- Delay feature comparison
- Top service features separating satisfaction groups

### Main Visualization Findings

The target classes were slightly imbalanced, with more passengers labeled as **neutral or dissatisfied** than **satisfied**, but the imbalance was not extreme.

The service rating features showed some of the strongest differences between satisfied and dissatisfied passengers. In particular, the following features appeared important:

- Online boarding
- Inflight entertainment
- Seat comfort
- On-board service
- Leg room service
- Cleanliness

Categorical features also showed useful patterns. Passenger satisfaction appeared to be related to:

- Customer Type
- Type of Travel
- Class

Business class passengers and business travel passengers were more likely to be satisfied, while economy class and personal travel passengers were more likely to be neutral or dissatisfied.

Delay features were highly skewed. Most flights had small delays, while a smaller number had very large delays. Delays may help predict satisfaction, but they did not appear to explain satisfaction as strongly as service-related features.

## Best Feature from Visualization

Based on the visualizations, **Online boarding** appeared to be one of the strongest individual features for predicting passenger satisfaction.

This feature had one of the largest average rating gaps between satisfied and neutral/dissatisfied passengers. Satisfied passengers tended to give much higher online boarding ratings, while dissatisfied passengers gave lower ratings.

Because of this, online boarding was expected to be an important predictor in the machine learning models.

## Data Cleaning and Preparation

Before training machine learning models, the dataset was prepared using the following steps:

1. Removed the unnecessary `Unnamed: 0` column.
2. Removed the `id` column because it was only an identifier.
3. Handled missing values in `Arrival Delay in Minutes`.
4. Separated the features from the target variable.
5. Encoded the target variable into 0 and 1.
6. Converted categorical variables into numerical dummy variables using one-hot encoding.
7. Split the data into training and testing sets.
8. Scaled numerical features for Logistic Regression.

The categorical variables were encoded because machine learning models cannot directly use text values such as `Male`, `Female`, `Business travel`, or `Eco`.

The numerical features were scaled for Logistic Regression because this model can be affected by features being on very different scales.

## Machine Learning Models

Two classification models were trained:

1. Logistic Regression
2. Decision Tree Classifier

### Logistic Regression

Logistic Regression was used as the first baseline machine learning model.

The model achieved an accuracy of approximately:

| Model | Accuracy |
|---|---:|
| Logistic Regression | 87.66% |

Logistic Regression performed well overall, but it was slightly better at identifying neutral or dissatisfied passengers than satisfied passengers.

### Decision Tree Classifier

The Decision Tree Classifier was used as a second model. This model performed better than Logistic Regression.

| Model | Accuracy |
|---|---:|
| Decision Tree | 94.48% |

The Decision Tree model performed better because it can capture more complex decision patterns in the data.

## Model Comparison

| Model | Accuracy |
|---|---:|
| Logistic Regression | 87.66% |
| Decision Tree Classifier | 94.48% |

The Decision Tree Classifier had the highest accuracy. This suggests that passenger satisfaction may depend on non-linear patterns and combinations of features.

## Feature Importance

Feature importance from the Decision Tree model was used to understand which variables helped the model make predictions.

The most important features supported the findings from the visualization section. Service-related features, especially **online boarding**, appeared to be highly important for predicting passenger satisfaction.

This means the model results matched the earlier exploratory analysis.

## Final Conclusion

This project used an airline passenger satisfaction dataset to predict whether passengers were **satisfied** or **neutral or dissatisfied**.

The visualizations showed that service-related features had strong relationships with passenger satisfaction. In particular, **online boarding**, **inflight entertainment**, **seat comfort**, **on-board service**, **leg room service**, and **cleanliness** showed noticeable differences between satisfied and dissatisfied passengers.

The categorical visualizations also showed that **customer type**, **type of travel**, and **class** were important. Business class passengers and business travel passengers were more likely to be satisfied, while personal travel and economy class passengers were more likely to be neutral or dissatisfied.

After cleaning the data, handling missing values, encoding categorical variables, splitting the data, and scaling numerical features, I trained Logistic Regression and Decision Tree models.

The Logistic Regression model achieved about **87.66% accuracy**, while the Decision Tree model achieved about **94.48% accuracy**. The Decision Tree performed better, suggesting that passenger satisfaction depends on feature combinations that may not be fully linear.

Overall, the results suggest that airlines should prioritize improving controllable service factors, especially **online boarding** and the overall **in-flight experience**, to improve passenger satisfaction.

## Files in This Repository

| File | Description |
|---|---|
| `Kaggle_from_stash.ipynb` | Main project notebook |
| `train.csv` | Airline passenger satisfaction training data |
| `README.md` | Project overview and summary |

## Tools and Libraries Used

This project used:

- Python
- pandas
- numpy
- matplotlib
- scikit-learn
- Jupyter Notebook

## How to Run This Project

1. Download or clone this repository.
2. Make sure `train.csv` is in the same folder as the notebook.
3. Open `Kaggle_from_stash.ipynb`.
4. Run the notebook from top to bottom.

The notebook loads the dataset, performs data exploration, creates visualizations, prepares the data, trains machine learning models, and compares the results.
