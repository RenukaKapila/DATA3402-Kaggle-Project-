# Airline Satisfaction Predictor
This project uses the **Airline Passenger Satisfaction** dataset from Kaggle to predict whether airline passengers are **satisfied** or **dissatisfied** based on passenger information, travel details, service ratings, and delay-related features.

## Project Overview

The project is a **binary classification problem** because the target variable has two possible outcomes.

The main goal is not only to build a prediction model, but also to understand which features are most connected to passenger satisfaction. I was especially interested in service-related features that airlines could improve, such as online boarding, inflight wifi service, seat comfort, inflight entertainment, cleanliness, and leg room service.

## Dataset

Dataset link:  
https://www.kaggle.com/datasets/teejmahal20/airline-passenger-satisfaction

The dataset contains airline passenger survey responses. Each row represents one passenger, and each column gives information about that passenger, their trip, their service ratings, delay information, or their satisfaction result.

## Problem Type

This is a **binary classification** problem.

The target variable is:

- `satisfied`
- `neutral or dissatisfied`

For presentation and explanation purposes, I refer to `neutral or dissatisfied` as **dissatisfied**. The original dataset label was not changed.

For machine learning, the target variable was encoded as:

- `0 = dissatisfied`
- `1 = satisfied`

## Dataset Description

The dataset has about **103,000 rows**.

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

The service rating columns are mostly on a scale from **0 to 5**, where higher values usually mean better ratings.

## Data Cleaning

### Removed Extra Index Column

The dataset included an extra column called `Unnamed: 0`. This column was removed because it was only an index column and did not provide useful information for prediction.

### Duplicate Rows

I checked for duplicate rows and found:

| Check | Result |
|---|---:|
| Duplicate rows | 0 |

Since there were no duplicate rows, no duplicate records needed to be removed.

### Missing Values

The only column with missing values was:

| Column | Missing Values |
|---|---:|
| Arrival Delay in Minutes | 310 |

Instead of deleting these rows, I filled the missing values using a simple nearest-match approach.
<img width="842" height="671" alt="image" src="https://github.com/user-attachments/assets/2dd95e07-c5cb-4c2f-b2f4-2d58fc982a71" />

First, I checked the relationship between **Departure Delay in Minutes** and **Arrival Delay in Minutes**. These two columns had a very strong positive correlation of about **0.97**.

Because of this strong relationship, I used departure delay to estimate the missing arrival delay values.

For each row where arrival delay was missing:

1. I looked at that row’s departure delay.
2. I found the three rows with the closest departure delay values where arrival delay was already known.
3. I averaged those three known arrival delay values.
4. I used that average to fill the missing arrival delay.

This method is similar to the idea behind K-nearest neighbors, but I did not train a KNN model. I only used a simple nearest-match approach.

After filling the missing values, the dataset had **0 missing values**.

## Outlier Check

I checked for outliers using the **IQR method**.

There were many outliers in:

- Flight Distance
- Departure Delay in Minutes
- Arrival Delay in Minutes

I decided not to remove these outliers because they may represent real airline situations. For example, long flights and large delays can naturally happen because of weather, maintenance issues, airport delays, or other travel disruptions.

Removing these rows could remove useful information from the model.

## Target Variable and Class Balance

The target variable was **satisfaction**.

The dataset had a small class imbalance:

| Class | Count | Percentage |
|---|---:|---:|
| Dissatisfied | 58,879 | 56.67% |
| Satisfied | 45,025 | 43.33% |

The imbalance was not extreme, so accuracy was still useful. However, I also used precision, recall, F1-score, and confusion matrices to evaluate the models more carefully.

## Data Visualization

The visualization section compared satisfied and dissatisfied passengers.

### Numerical Features

I looked at numerical features such as:

- Age
- Flight Distance
- Departure Delay in Minutes
- Arrival Delay in Minutes

These features showed some patterns, but many of the distributions overlapped between satisfied and dissatisfied passengers. This means they may help the model, but they are not strong enough by themselves.
<img width="793" height="451" alt="image" src="https://github.com/user-attachments/assets/56d67327-8d6b-4d10-ada2-ef7d32856455" />
<img width="765" height="431" alt="image" src="https://github.com/user-attachments/assets/d71c6a37-623b-4e6a-ad6a-ae97766f1770" />
<img width="767" height="434" alt="image" src="https://github.com/user-attachments/assets/849d392d-d01a-4492-8691-8bc0df9eaaf5" />
<img width="786" height="438" alt="image" src="https://github.com/user-attachments/assets/999bdaf9-50c2-4a78-a988-8e2541404171" />

### Categorical Features

I also looked at categorical features such as:

- Gender
- Customer Type
- Type of Travel
- Class

Gender did not show a very strong difference. However, Customer Type, Type of Travel, and Class showed clearer patterns.

Business travelers and business class passengers were more likely to be satisfied, while personal travel and economy class passengers were more likely to be dissatisfied.

<img width="1004" height="518" alt="image" src="https://github.com/user-attachments/assets/7210ddf8-ba2f-4580-8bfc-28219299af1b" />
<img width="895" height="500" alt="image" src="https://github.com/user-attachments/assets/9f892c9a-1e47-43b8-b0b2-c12d5dd3c9f1" />
<img width="879" height="487" alt="image" src="https://github.com/user-attachments/assets/32249827-1029-447c-bf67-22769e57ed2a" />
<img width="899" height="490" alt="image" src="https://github.com/user-attachments/assets/404a4711-1534-4e94-b760-408a328f40a2" />

### Service Rating Difference

The most useful visualization was the **service rating difference chart**.

This chart compared average service ratings between satisfied and dissatisfied passengers for features such as:

- Online boarding
- Inflight wifi service
- Seat comfort
- Inflight entertainment
- On-board service
- Leg room service
- Cleanliness
- Food and drink
- Checkin service
- Ease of Online booking

The biggest difference was **Online boarding**. Satisfied passengers gave much higher online boarding ratings than dissatisfied passengers.

<img width="779" height="389" alt="image" src="https://github.com/user-attachments/assets/01aba229-1b35-471b-bcf7-5b9e4950f286" />

## Data Preparation for Machine Learning

Before training the models, I prepared the data using these steps:

1. Made a copy of the cleaned dataset.
2. Encoded the target variable:
   - `0 = dissatisfied`
   - `1 = satisfied`
3. Separated the data into `X` and `y`.
   - `X` contains the features.
   - `y` contains the target variable.
4. Removed the original satisfaction column, encoded satisfaction column, and ID column from `X`.
5. Used one-hot encoding to convert categorical text columns into numerical 0/1 columns.
6. Split the data into training and testing sets.
7. Scaled the features for Logistic Regression.

The categorical columns that were one-hot encoded included:

- Gender
- Customer Type
- Type of Travel
- Class

## Train/Test Split

I used an **80/20 train-test split**.

This means:

- 80% of the data was used for training.
- 20% of the data was used for testing.

I also used a **stratified split** so the training and testing sets kept a similar satisfied/dissatisfied class balance as the original dataset.

This helped make the model evaluation more fair.

## Feature Scaling

I used **StandardScaler** for Logistic Regression because the numerical features had very different ranges.

For example:

- Service ratings were mostly from 0 to 5.
- Flight Distance could be close to 5,000.
- Delay values could go above 1,000 minutes.

Scaling helped Logistic Regression handle the different feature ranges more fairly.

Decision Tree and Random Forest did not need scaled data because they are tree-based models.

## Machine Learning Models

I trained three models:

1. Logistic Regression
2. Decision Tree Classifier
3. Random Forest Classifier

## Model Results

| Model | Accuracy |
|---|---:|
| Logistic Regression | 87.66% |
| Decision Tree Classifier | 94.48% |
| Random Forest Classifier | 96.20% |

The **Random Forest Classifier** performed the best with about **96.20% accuracy**.

## Feature Importance

After training the Random Forest model, I looked at feature importance.

The most important feature was:

- Online boarding

Other important features included:

- Inflight wifi service
- Type of Travel
- Class
- Inflight entertainment
- Seat comfort
- Leg room service
- Flight Distance
- Customer Type
- Ease of Online booking

This matched the visualization section because Online boarding also had the biggest service rating difference between satisfied and dissatisfied passengers.

## Final Conclusion

This project showed that airline passenger satisfaction can be predicted well using passenger information, travel details, service ratings, and delay-related features.

The strongest patterns came from service-related and travel-related features. In particular, **online boarding**, **inflight entertainment**, **seat comfort**, **inflight wifi service**, **class**, and **type of travel** were important.

The best model was the **Random Forest Classifier**, with about **96.20% accuracy**.

Overall, the results suggest that airlines should focus on improving controllable service factors, especially **online boarding** and the overall in-flight experience, because these features were strongly associated with passenger satisfaction.

## Files in This Repository

| File | Description |
|---|---|
| `Airline_Passenger_Satisfaction_Project.ipynb` | Main project notebook |
| `train.csv` | Dataset used for the project |
| `README.md` | Project summary and explanation |

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
3. Open `Airline_Passenger_Satisfaction_Project.ipynb`.
4. Run the notebook from top to bottom.

The notebook loads the data, cleans it, creates visualizations, prepares the dataset for machine learning, trains models, compares results, and shows feature importance.
