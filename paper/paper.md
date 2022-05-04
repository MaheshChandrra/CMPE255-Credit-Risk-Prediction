---
title: Credit Rist Prediction
date: "April 2022"
author: Team 6, San José State University

header-includes: |
  \usepackage{booktabs}
  \usepackage{caption}
---

# Abstract

Banks lose a significant amount of money as a result of credit defaults, and it is ultimately ordinary customers who bear the brunt of this error. Credit Risk Analysis is used by banks to ensure that credit is supplied to a trustworthy consumer. Credit risk is defined as the risk of defaulting on a loan as a result of the borrower's failure to make mandatory debt payments on time.  The lender assumes this risk because the lender loses both the capital and the interest on the loan.Machine Learning-based credit risk analysis eliminates the time-consuming human process of assessing numerous criteria and conditions on which credit can be granted. In the process, it also eliminates the human factor of mathematical mistake and corruption.From this project we aim to build a model to predict whether a person is eligible to get a credit or not. The decision depends on his/her banking history and other parameters mentioned below.

# Introduction

Interest on loans is an important source of revenue for banks. Banks assess various characteristics before granting a loan to a client because they must be certain that the consumer will be able to repay the loan within the loan term. This carries a high level of risk; any error in evaluating client history might result in the bank losing credit. As a result, it is critical to do a customer analysis before extending credit to them.

This article explains the strategy to developing a machine learning model that can predict whether a consumer will be approved for a loan or not. We investigated an open source dataset containing data on consumers who paid their debts and those who did not in order to solve this challenge. We delegated this task to a machine learning model.



# Methods

To develop the model,we have designed a pipeline of steps to reach the end goal.The steps involved in the pipeline are :
1. Data Collection
2. Exploratory Data Analysis 
3. Preprocessing Data
4. Feature Selection
5. Model Training and Parameter tuning
6. Model Evaluation
7. Deployment 

Below is the pipeline:

![Credit risk (1)](https://user-images.githubusercontent.com/13826929/166613675-cd3722bc-8d39-4971-a12e-fd5b73e327f2.png)

**1. Data Collection** : We have used an open source dataset for solving the problem.

**2. Exploratory Data Analysis** :The dataset is a combination of both numerical and categorical data.

Below are the features in the dataset:

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
| **loan_status**            | Loan status (**0 is non default 1 is default**) |

loan_status is the target variable up for prediction.Default is the failure to repay a loan according to the terms agreed to in the promissory note.




# Comparisons

# Example Analysis

# Conclusions


# References
