# CMPE255-Credit-Risk-Prediction

**Team Members**:

| Team Member                     | Role      | githubprofile                     |
|---------------------------------|-----------|-----------------------------------|
| Lokesh Vaddi                    | Developer | https://github.com/lokesh31333    |
| Nikhil Kumar Kanisetty          | Developer | https://github.com/Nikhil-Kumar-K |
| Shanmukha Yaswanth Reddy Kallam | Developer | https://github.com/shanmuk3       |
| Mareedu Mahesh Chandra          | Developer | https://github.com/MaheshChandrra |

## Introduction
Defaulted loans are a rising problem for most banks and lending institutions. Credit risk analysis is the first and most important phase in the loan approval process in order to avoid or lessen the effects of the disaster. When a debt defaults, it causes damage. When a person or a company seeks for a loan, the lender must determine whether the person or company can reliably return the loan principle and interest. Lenders frequently use profitability and leverage metrics to analyze credit risk.
From this project we aim to build a model to predict whether a person is eligible to get a credit or not.
The decision depends on his/her banking history and other parameters mentioned below.

**Source Dataset**:
> https://www.openml.org/search?type=data&status=active&id=43454&sort=runs

**Featues from the dataset**:

| Feature Name           | Description                                 | 
|------------------------|---------------------------------------------|
| person_age             | Age                                         |
| person_income          | Annual Income                               |
| personhomeownership    | Home ownership                              |
| personemplength        | Employment length (in years)                |
| loan_intent            | Loan intent                                 |
| loan_grade             | Loan grade                                  |
| loan_amnt              | Loan amount                                 |
| loanintrate            | Interest rate                               |
| loanpercentincome      | Percent income                              |
| cbpersondefaultonfile  | Historical default                          |
| cbpresoncredhistlength | Credit history length                       |
| **loan_status**        | Loan status (0 is non default 1 is default) |


## Methodology

**Problem Type:** Binary Classification (Supervised Learning)

**Algorithms:** Logistic Regression,Random Forest,XGBoost

The dataset is a combination of Numerical and Categorical data,requires preprocessing  and analysis on each of the features before building the model.

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
    
* Identifying total discre classes
* Analyzing and visualizing the distritbution of data by class
* If imbalance is identified,upsampling or downsampling is implemented

**To do in Model Building**:
   
* Create train test splits (Stratified sampling)
* Building model pipeline
* Training the data
* Validaing the data
* Parameter tuning
* Saving model
* Deployment using flask
    

## Setup
``` pip install -r requirements.txt ```
