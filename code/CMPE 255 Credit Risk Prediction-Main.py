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


# In[2]:


file_path=properties.DATASET_DIR+properties.DATASET_FILENAME


# ### Reading dataset
# 
# Author: Mahesh Chandra Mareedu

# In[3]:


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


import plotly.figure_factory as ff
import numpy as np

# Add histogram data
x1 = np.random.randn(200) - 2
x2 = np.random.randn(200)
x3 = np.random.randn(200) + 2
x4 = np.random.randn(200) + 4

# Group data together
hist_data = [x1, x2, x3, x4]

group_labels = ['Group 1', 'Group 2', 'Group 3', 'Group 4']

# Create distplot with custom bin_size
fig = ff.create_distplot(hist_data, group_labels, bin_size=.2)
fig.show()


# In[ ]:


df_data.columns


# In[ ]:


df_data.columns


# ##### Converting datatype object to float for all numerical columns

# In[ ]:


NUMERICAL_COLUMNS=['person_age', 'person_income','person_emp_length', 'loan_amnt',
       'loan_int_rate', 'loan_status', 'loan_percent_income', 'cb_person_cred_hist_length']


# In[ ]:


for col in NUMERICAL_COLUMNS:
    df_data[col]=df_data[col].astype(float)


# #### Correlation plot around numerical features
# 
# Author : Mahesh Chandra Mareedu

# In[ ]:


get_correlation_heatmap(df_data)


# In[ ]:


get_correlation_pairplot(df_data)


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





# ####  Author - Nikhil
#     - Box plot
#     - Violin charts
# 

# In[ ]:





# In[ ]:





# In[ ]:





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





# In[ ]:




