# CMPE255-Credit-Risk-Prediction

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

**Team Members**:

| Team Member                     | Role      | githubprofile                     |
|---------------------------------|-----------|-----------------------------------|
| Lokesh Vaddi                    | Developer | https://github.com/lokesh31333    |
| Nikhil Kumar Kanisetty          | Developer | https://github.com/Nikhil-Kumar-K |
| Shanmukha Yaswanth Reddy Kallam | Developer | https://github.com/shanmuk3       |
| Mareedu Mahesh Chandra          | Developer | https://github.com/MaheshChandrra |


## Methodology

**Problem Type:** Binary Classification (Supervised Learning)

**Algoritms:** Logistic Regression,Random Forest,XGBoost
