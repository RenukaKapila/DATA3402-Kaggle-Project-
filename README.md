# Airline Satisfaction Predictor

This repository uses the Airline Passenger Satisfaction dataset from Kaggle to predict whether a passenger is satisfied or dissatisfied using tabular machine learning models.

## Overview

This project uses the Airline Passenger Satisfaction dataset from Kaggle. The goal is to predict whether an airline passenger is satisfied or dissatisfied based on passenger details, travel information, service ratings, and delay-related features.

The dataset is a good fit for a tabular classification project because every row represents one passenger survey response, and every column represents a feature related to the passenger, flight, service experience, or satisfaction label.

The main work included cleaning the data, handling missing arrival delay values, checking outliers, visualizing feature patterns, encoding categorical variables, training models, and comparing performance.

The best-performing model was Random Forest, with an accuracy of about **96.20%**.

## Summary of Work Done

### Data

* Data:
  * Type: CSV file with airline passenger survey features.
  * Input: passenger information, travel details, service ratings, and delay-related features.
  * Output: passenger satisfaction.
    * `0` = dissatisfied
    * `1` = satisfied
  * Size:
    * 103,904 rows
    * 25 original columns
    * 24 columns after removing the extra index column

The original target labels were:

* `satisfied`
* `neutral or dissatisfied`

For explanation purposes, we refer to `neutral or dissatisfied` as **dissatisfied**. The original dataset label was not changed.

### Important Features

Some important features included:

* Gender
* Customer Type
* Age
* Type of Travel
* Class
* Flight Distance
* Inflight wifi service
* Online boarding
* Seat comfort
* Inflight entertainment
* Cleanliness
* Departure Delay in Minutes
* Arrival Delay in Minutes

The service rating features were mostly on a scale from 0 to 5, where higher values usually represented better ratings.

---

## Preprocessing / Clean Up

The dataset included an extra column called `Unnamed: 0`. We removed this column because it was only an index column and did not provide useful information for prediction.

We also checked for duplicate rows.

| Check | Result |
|---|---:|
| Duplicate rows | 0 |

There were no duplicate rows, so no repeated records had to be removed.

### Missing Values

The only column with missing values was `Arrival Delay in Minutes`.

| Column | Missing Values |
|---|---:|
| Arrival Delay in Minutes | 310 |

We did not delete these rows because delay information could still be useful for predicting passenger satisfaction.

To fill these missing values, we used information from `Departure Delay in Minutes`. These two columns had a strong positive correlation of about **0.97**, meaning flights that leave late usually also arrive late.

We used a simple nearest-match approach, similar to the idea behind K-nearest neighbors. We did not train a KNN model.

For each missing arrival delay value:

1. We looked at that row’s departure delay.
2. We found the three rows with the closest departure delay values where arrival delay was already known.
3. We averaged those three known arrival delay values.
4. We used that average to fill the missing arrival delay value.

We also checked whether the three closest matches were close enough. The maximum distance to the third closest departure delay match was only **2 minutes**, so the method was reasonable.

After this step, the dataset had **0 missing values**.

### Outliers

We checked for outliers using the IQR method.

| Feature | Outliers Found |
|---|---:|
| Flight Distance | 2,291 |
| Departure Delay in Minutes | 14,529 |
| Arrival Delay in Minutes | 14,022 |

We kept these outliers because they may represent real airline situations.

For example, long flights and large delays can happen because of weather, maintenance issues, airport delays, or other travel disruptions. Removing these rows could remove useful information from the model.

---

## Data Visualization

We used visualizations to compare satisfied and dissatisfied passengers.

The main visualizations included:

* Target class distribution
* Age distribution by satisfaction
* Flight distance by satisfaction
* Departure delay by satisfaction
* Arrival delay by satisfaction
* Categorical feature comparisons
* Service rating difference chart
* Correlation heatmap

The target classes were slightly imbalanced.

| Class | Count | Percentage |
|---|---:|---:|
| Dissatisfied | 58,879 | 56.67% |
| Satisfied | 45,025 | 43.33% |

The numerical features showed some patterns, but many of the distributions overlapped between satisfied and dissatisfied passengers. This means those features may help the model, but they were not the strongest features by themselves.

The categorical features showed clearer patterns. Customer Type, Type of Travel, and Class appeared more connected to satisfaction than Gender.

The strongest visualization was the service rating difference chart. It compared average service ratings between satisfied and dissatisfied passengers.

Online boarding had the largest average rating difference between the two groups.

Other strong service-related features included:

* Inflight entertainment
* Seat comfort
* On-board service
* Leg room service
* Cleanliness

This suggested that service-related features were important for predicting satisfaction.

---

## Problem Formulation

* Input:
  * Passenger information, travel details, service ratings, and delay-related features.

* Output:
  * Passenger satisfaction.
  * `0` = dissatisfied
  * `1` = satisfied

* Models:
  * Logistic Regression
  * Decision Tree
  * Random Forest

The problem was formulated as a supervised binary classification task. The goal was to train models that could predict whether a passenger was satisfied or dissatisfied.

---

## Feature Preparation

Before training the models, we prepared the data by:

* Separating the features and target variable
* Encoding the target variable
* Removing the ID column
* Encoding categorical variables
* Splitting the data into training and testing sets
* Scaling features for Logistic Regression

### Target Encoding

| Original Label | Encoded Value |
|---|---:|
| Dissatisfied | 0 |
| Satisfied | 1 |

### One-Hot Encoding

The categorical columns were converted into numeric 0/1 columns using one-hot encoding.

The encoded categorical columns came from:

* Gender
* Customer Type
* Type of Travel
* Class

This allowed the machine learning models to use categorical information.

### Train-Test Split

We used an 80/20 train-test split.

| Split | Percentage |
|---|---:|
| Training set | 80% |
| Testing set | 20% |

We also used stratified splitting so the satisfied and dissatisfied class balance stayed similar in both the training and testing sets.

### Scaling

We used StandardScaler for Logistic Regression because the numerical features had very different ranges.

For example, service ratings were usually from 0 to 5, while flight distance and delay values could be much larger.

Decision Tree and Random Forest did not need scaled data because they are tree-based models.

---

## Training

We trained the models with scikit-learn on a local machine.

The training process used:

* Encoded features
* 80/20 train-test split
* Stratified class balance
* Scaled data for Logistic Regression
* Unscaled encoded data for Decision Tree and Random Forest

We used `random_state=42` to make the results repeatable.

No GPU was required. The models were trained using CPU.

---

## Performance Comparison

The main evaluation metrics were accuracy, precision, recall, F1 score, classification report, and confusion matrix.

The model accuracy results were:

| Model | Accuracy |
|---|---:|
| Logistic Regression | 87.66% |
| Decision Tree | 94.48% |
| Random Forest | 96.20% |

The Random Forest model performed the best.

Random Forest likely performed better because it combines many decision trees. This makes it more stable than a single Decision Tree and helps it capture more complex patterns in the data.

---

## Feature Importance

We used Random Forest feature importance to understand which features were most useful for prediction.

The most important feature was:

* Online boarding

Other important features included:

* Inflight wifi service
* Type of Travel
* Class
* Inflight entertainment
* Seat comfort
* Leg room service
* Flight Distance
* Customer Type
* Ease of Online booking

This matched the visualization results because Online boarding also had the largest service rating difference between satisfied and dissatisfied passengers.

This made the result stronger because both the visualization and the model pointed to the same important feature.

---

## Conclusions

The Random Forest model performed best for predicting airline passenger satisfaction.

The project showed that satisfaction was strongly associated with service quality and travel-related features. The strongest feature was Online boarding, followed by other service and travel-related features such as inflight wifi service, type of travel, class, inflight entertainment, seat comfort, and leg room service.

The results suggest that airlines should focus on controllable service factors, especially online boarding and the overall in-flight experience, to improve passenger satisfaction.

The project also showed the full data science workflow, including data loading, cleaning, visualization, preprocessing, model training, evaluation, and documentation.

---

## How to Reproduce Results

### Overview of Files in Repository

* `README.md`: project report and summary.
* `Airline_Passenger_Satisfaction_Project.ipynb`: main notebook containing data loading, inspection, visualization, preprocessing, modeling, evaluation, and conclusion.
* `train.csv`: dataset used for the project.

If the notebook file has a different name in GitHub, use that notebook file instead.

### Software Setup

This project uses Python with the following main packages:

* pandas
* numpy
* matplotlib
* scikit-learn
* jupyter
* ipykernel

### Data

The dataset comes from Kaggle:

https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction

In this repository, the dataset file is included as:

```text
train.csv
