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

# In[4]:


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

# In[5]:


missing_columns_list=check_missing_columns(df_data)


# In[6]:


print("Missing data in columns:",missing_columns_list)


# #### Visualizing distribution of data for missing columns
# 
# Author: Mahesh Chandra Mareedu

# In[7]:


for col in missing_columns_list:
    visualize_pdf(df_data,col,True)


# #### Imputing Missing Values
# 
# Author: Mahesh Chandra Mareedu

# In[8]:


df_data,imputed_value_dict=impute_missing_values(df_data,missing_columns_list)


# In[9]:


imputed_value_dict


# In[10]:


df_impute=pd.DataFrame(imputed_value_dict.items(), columns=['Column', 'Mean Value'])


# In[11]:


df_impute


# #### Distribution plot on numerical data
# 
# Author : Mahesh Chandra Mareedu

# #### Correlation plot around numerical features
# 
# Author : Mahesh Chandra Mareedu

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

# ####  Author - Nikhil
#     - Box plot
#     - Violin charts
# 

# In[ ]:





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

# In[12]:


get_ipython().system('jupyter nbconvert CMPE*.ipynb --to python')


# ### Rough

# In[ ]:




