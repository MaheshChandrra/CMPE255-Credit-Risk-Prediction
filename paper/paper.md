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


  <img src="https://user-images.githubusercontent.com/13826929/166613675-cd3722bc-8d39-4971-a12e-fd5b73e327f2.png" width="300" height="800" />


**1. Data Collection** : We have used an open source dataset for solving the problem.Dataset contains 32581 records and 12 columns(11 features and 1 target varible)

**2. Exploratory Data Analysis and Prepreocessing** :The dataset is a combination of both numerical and categorical data.

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

As a part of EDA and preprocessing we have performed the below methods to clean the data and visualize the insights:
Dropping Duplicate records: We have dropped all the duplicate records from the dataset.

  **Handling missing values** : The dataset is checked for find NA values, after analyzing the dataset we have found that there are no NULL values. Instead we have a placeholder “?” where ever the records are missing.
  loan_int_rate and loan_intent are two columns found to have missing values.The values are imputed 
  with their corresponding mean values.The reason for imputing mean is that mean and median of these columns are in very close range indicating no outliers.
  
  **Removing Outliers** : We have looked at the box plots of all the features.Out of all the features we have analyzed that there are few customer records whose age(person_age) is greater than 80.Considering normal expectency as 80 yrs,we have discarded data of customers whose age is greater than 80.
  
  For feature person_emp_length,considering the the retirement period is 60 years,max employement for a person would be 40-45 yrs if he/she starts working around 15-20.So we discarded data of persons where employment period is greater than 41.
  
  **Co-relation Plots**: We have checked if there exists any correlation between the features.For this purpose we have visualized heat maps.
  It is observed that cb_person_cred_hist_length and person_age have high collinearity.
  

  The correlation is positive as we can see from the  below pair plot and heat map.
  

| Heat Map             |  Pair plot 
:-------------------------:|:-------------------------:
  <img src="https://user-images.githubusercontent.com/13826929/166618176-3801b6c6-78d6-4871-a0fc-ad0b2de73672.png" width="300" height="300" />|   <img src="https://user-images.githubusercontent.com/13826929/166618574-26d864d1-a206-415b-8a43-522e73f62222.png" width="300" height="300" />


**Analysis of data using Tree maps**

From the Dataset we can make a treemap that allows us to represent a hierarchically-ordered (tree-structured) set of dataset 

From the below Treemap-1 we can observe, what is the loan amount that different age groups of people with different income ranges are requesting.

we can observe that people with age 22-28 have more loan applications than others.

Then if we observe the intent of the loan for different ages from TreeMap-2
we can see that most of the loan applications are for Education then followed by Medical, Venture, and personal. 

Then if we observe the Treemap-3 we can see get the percentage of loan defaulters in different loan intent category

Education loans have 17 percent, loan defaulters.
Venture loans have 17 percent, loan defaulters.
Medical Loans have 17.5 percent, loan defaulters.
Personal Loans have 17.3 percent, loan defaulters.
Debt consolidation Loans have 18 percent, loan defaulters.
Home improvement Loans have 19.3 percent, loan defaulters.


From the dataset, we can make a Parallel coordinates plot is used to analyze multivariate data

From the below Parallel plot we can observe how the loan grade is divided for the different Loan intents for the different situations.
From Parallel plot 2 we can see the Loan status for the different loan grades for different loan intents.

# Comparisons

# Example Analysis

# Conclusions


# References
