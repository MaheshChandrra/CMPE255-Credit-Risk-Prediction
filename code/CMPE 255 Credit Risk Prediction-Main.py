#!/usr/bin/env python
# coding: utf-8

# ### Importing modules

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from preprocessing import *
from util import *
import properties


# In[ ]:


file_path=properties.DATASET_DIR+properties.DATASET_FILENAME


# ### Reading dataset
# 
# Author: Mahesh Chandra Mareedu

# In[ ]:


df_data=read_dataset(properties.DATASET_DIR+properties.DATASET_FILENAME)


# ### Saving File
# 
# Author: Mahesh Chandra Mareedu

# In[ ]:


save_file(df_data,properties.DATASET_DIR)


# ###  EDA
# 
# The dataset is a combination of numerical and categorical data.
# 
# To Do on Numerical Data:
# - Visualizing distributions of data
# - Eliminating outliers by checking how much data points are deviated away from mean/median.
# - Identifying missing data columns,and filling them with Mean/Median
# - Visualizing Correlation plots among features
# - Performing Standardization/Normalization
# 
# To do on Categorical Data:
# - Filling missing values with Mode
# - Checking for any random text with in features,and discarding them
# - Checking the value counts of each categorical feature
# - Encoding features using One-Hot-Encoder/Label-Encoder/Multi Label Binarizer
# 
# To do on Target label Data:
# - Identifying total discre classes
# - Analyzing and visualizing the distritbution of data by class
# - If imbalance is identified,upsampling or downsampling is implemented
# 
# To do Model Building:
# - Create train test splits (Stratified sampling)
# - Building model pipeline
# - Training the data
# - Validaing the data
# - Parameter tuning
# - Saving model
# - Deployment using flask
#     

# #### Checking for missing data
# 
# Author: Mahesh Chandra Mareedu

# In[ ]:


missing_columns_list=check_missing_columns(df_data)


# In[ ]:


print("Missing data in columns:",missing_columns_list)


# #### Visualizing distribution of data for missing columns
# 
# Author: Mahesh Chandra Mareedu

# In[ ]:


for col in missing_columns_list:
    visualize_pdf(df_data,col,True)


# #### Imputing Missing Values
# 
# Author: Mahesh Chandra Mareedu

# In[ ]:


df_data,imputed_value_dict=impute_missing_values(df_data,missing_columns_list)


# In[ ]:


imputed_value_dict


# In[ ]:


df_impute=pd.DataFrame(imputed_value_dict.items(), columns=['Column', 'Mean Value'])


# In[ ]:


df_impute


# #### Distribution plot on numerical data
# 
# Author : Mahesh Chandra Mareedu

# In[ ]:


df_data.columns


# In[ ]:


df_data.columns


# ##### Converting datatype object to float for all numerical columns

# In[ ]:


NUMERICAL_COLUMNS=['person_age', 'person_income','person_emp_length', 'loan_amnt',
       'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
TARGET_LABEL='loan_status'


# In[ ]:


for col in NUMERICAL_COLUMNS:
    df_data[col]=df_data[col].astype(float)


# In[ ]:


TARGET_LABEL='loan_status'
get_distplot(df_data,NUMERICAL_COLUMNS,TARGET_LABEL,True)


# In[ ]:


get_distplot(df_data,NUMERICAL_COLUMNS,TARGET_LABEL,False)


# #### Correlation plot around numerical features
# 
# Author : Mahesh Chandra Mareedu

# In[ ]:


get_correlation_heatmap(df_data)


# In[ ]:


df_data


# In[ ]:


get_correlation_pairplot(df_data[NUMERICAL_COLUMNS])


# In[ ]:


get_correlation_pairplot(df_data[['person_age', 'person_income', 'loan_amnt','loan_status']],'loan_status')


# ####  Author : Shanmuk
# 
# Bar charts
#        
# - On counts of categorical columns
# - On categorical columns by target label
# 
# 

# #### Author - Lokesh
#     Tree Map
#     Parallel Categories
# 
# Author : Lokesh

# In[ ]:


get_correlation_treemap(df_data)


# In[ ]:


get_correlation_parallel(df_data)


# In[ ]:


get_correlation_between_data(df_data)


# ####  Author - Nikhil
#     - Box plot
#     - Violin charts
# 

# ### Plots to visualize
# - Distribution plot on numerical data
# - Correlation plot around numerical features
# - Bar charts
#         - On counts of categorical columns
#         - On categorical columns by target label
# - Tree Map
# - Parallel Categories
# - Box plot
# - Violin charts
#         
#     
#     

# #### Convert notebook to app.py
# 
# 

# In[ ]:


get_ipython().system('jupyter nbconvert CMPE*.ipynb --to python')


# ### Rough

# In[ ]:




