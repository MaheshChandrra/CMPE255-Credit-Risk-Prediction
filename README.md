# CMPE255-Credit-Risk-Prediction

**Team Members**:

| Team Member                     | Role      | githubprofile                     |
|---------------------------------|-----------|-----------------------------------|
| Lokesh Vaddi                    | Developer | https://github.com/lokesh31333    |
| Nikhil Kumar Kanisetty          | Developer | https://github.com/Nikhil-Kumar-K |
| Shanmukha Yaswanth Reddy Kallam | Developer | https://github.com/shanmuk3       |
| Mareedu Mahesh Chandra          | Developer | https://github.com/MaheshChandrra |

## Introduction
- Banks lose a significant amount of money as a result of credit defaults, and it is ultimately ordinary customers who bear the brunt of this error. Credit Risk Analysis is used by banks to ensure that credit is supplied to a trustworthy consumer. Credit risk is defined as the risk of defaulting on a loan as a result of the borrower's failure to make mandatory debt payments on time.  The lender assumes this risk because the lender loses both the capital and the interest on the loan.

- Machine Learning-based credit risk analysis eliminates the time-consuming human process of assessing numerous criteria and conditions on which credit can be granted. In the process, it also eliminates the human factor of mathematical mistake and corruption.

- From this project we aim to build a model to predict whether a person is eligible to get a credit or not. The decision depends on his/her banking history and other parameters mentioned below.
- 
**Source Dataset**:
> https://www.openml.org/search?type=data&status=active&id=43454&sort=runs

## Dataset
The dataset we are using has data for 32,581 borrowers and 12 features related to each of them. Below listed are the features.

| Feature Name               | Description                                 | 
|----------------------------|---------------------------------------------|
| person_age                 | Age                                         |
| person_income              | Annual Income                               |
| person_home_ownership      | Home ownership                              |
| person_emp_length          | Employment length (in years)                |
| loan_intent                | Loan intent                                 |
| loan_grade                 | Loan grade                                  |
| loan_amnt                  | Loan amount                                 |
| loan_int_rate              | Interest rate                               |
| loan_percent_income        | Percent income                              |
| cb_person_default_on_file  | Historical default                          |
| cb_preson_cred_hist_length | Credit history length                       |
| **loan_status**            | Loan status (0 is non default 1 is default) |

From the dataset, we can see that the 'loan_status' is the target variable with '1' - the borrower will default on the loan and '0' - the borrower will not.

**Numerical Vairables:**\
person_age, person_income, person_emp_length, loan_amnt, loan_int_rate, loan_percent_income, cb_preson_cred_hist_length.

**Categorical Variables:**\
person_home_ownership, loan_intent, loan_grade, cb_person_default_on_file, loan_status.

In the data, we have missing values from the 'personemplength' and 'loan_int_rate' columns. But the missing values are a small percentage of the data, So we can remove those rows that have the missing values.

## Methodology

**Problem Type:** Binary Classification (Supervised Learning)

**Algorithms:** Logistic Regression,Random Forest,XGBoost

The dataset is a combination of numerical and categorical data,requires preprocessing  and analysis on each of the features before building the model.

Below are the to-do items:

## To do's: ## 

**On Numerical Data**:
    
* Visualizing distributions of data
* Eliminating outliers by checking how much data points are deviated away from mean/median.
* Identifying missing data columns,and filling them with Mean/Median
* Visualizing Correlation plots among features
* Performing Standardization/Normalization

**On Categorical Data**:
   
* Filling missing values with Mode
* Checking for any random text with in features,and discarding them
* Checking the value counts of each categorical feature
* Encoding features using One-Hot-Encoder/Label-Encoder/Multi Label Binarizer

**On Target label Data**:
    
* Identifying total discrete classes
* Analyzing and visualizing the distritbution of data by class
* If imbalance is identified,upsampling or downsampling is implemented

**To do in Model Building**:
   
* Create train test splits (Stratified sampling)
* Building model pipeline
* Training the data
* Validating the data
    *   Accuracy
    *   Precision
    *   Re-Call
    *   F1-Score
    *   ROC-AUC Curve
* Parameter tuning
* Saving model
* Deployment 
    

## Setup
``` pip install -r requirements.txt ```
