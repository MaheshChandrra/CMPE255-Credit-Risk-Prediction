#!/usr/bin/env python
# coding: utf-8

# ### Importing modules

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import iplot
from preprocessing import *
from util import *
import properties


# In[2]:


file_path=properties.DATASET_DIR+properties.DATASET_FILENAME


# ### Reading dataset
# 
# Author: Mahesh Chandra Mareedu

# In[10]:


df_data=read_dataset(properties.DATASET_DIR+properties.DATASET_FILENAME)


# ### Saving File
# 
# Author: Mahesh Chandra Mareedu

# In[11]:


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
#     

# #### Dropping Duplicate Records

# In[12]:


print("Length of data:",len(df_data))
df_data=df_data.drop_duplicates().reset_index(drop=True)
print("Length of data  after dropping duplicates:",len(df_data.drop_duplicates()))


# #### Checking for missing data
# 
# Author: Mahesh Chandra Mareedu

# In[13]:


missing_columns_list=check_missing_columns(df_data)


# In[14]:


print("Missing data in columns:",missing_columns_list)


# #### Visualizing distribution of data for missing columns
# 
# Author: Mahesh Chandra Mareedu

# In[15]:


for col in missing_columns_list:
    visualize_pdf(df_data,col,True)


# #### Imputing Missing Values
# 
# Author: Mahesh Chandra Mareedu

# In[16]:


df_data,imputed_value_dict=impute_missing_values(df_data,missing_columns_list)


# In[17]:


imputed_value_dict


# In[18]:


df_impute=pd.DataFrame(imputed_value_dict.items(), columns=['Column', 'Mean Value'])


# In[19]:


df_impute


# In[20]:


df_data.columns


# In[21]:


df_data.columns


# #### Converting datatype object to float for all numerical columns

# In[22]:


NUMERICAL_COLUMNS=['person_age', 'person_income','person_emp_length', 'loan_amnt',
       'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
TARGET_LABEL='loan_status'


# In[23]:


for col in NUMERICAL_COLUMNS:
    df_data[col]=df_data[col].astype(float)


# #### Removing Outliers

# ##### person_age
# 
# Considering the average persons age is around 80,discarding all the values where age is greater than 80.
# 

# In[24]:


df_data[df_data['person_age']>80]


# In[25]:


df_data=df_data[df_data['person_age']<80].reset_index(drop=True)


# ##### person_emp_length Age
# 
# Considering the the retirement period is 60 years,max employement for a person would be 40-45 yrs if he/she starts working around 15-20.Discarding where employment period is greater than 41.

# In[26]:


df_data[df_data['person_emp_length']>41]


# In[27]:


df_data=df_data[df_data['person_emp_length']<41].reset_index(drop=True)


# In[28]:


df_data


# #### Distribution plot on numerical data
# 
# Author : Mahesh Chandra Mareedu

# In[29]:


TARGET_LABEL='loan_status'
get_distplot(df_data,NUMERICAL_COLUMNS,TARGET_LABEL,True)


# In[30]:


get_distplot(df_data,NUMERICAL_COLUMNS,TARGET_LABEL,False)


# #### Correlation plot around numerical features
# 
# **Observations** : cb_person_cred_hist_length and person_age have high colleniarity from below plots,so dropping  cb_person_cred_hist_length.
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


# In[ ]:


NUMERICAL_COLUMNS.remove('cb_person_cred_hist_length')


# In[ ]:


NUMERICAL_COLUMNS


# ####  Author : Shanmuk
# 
# Bar charts
#        
# - On counts of categorical columns
# - On categorical columns by target label
# 
# 

# In[ ]:


get_barplot(df_data)


# In[30]:


get_barplot_catagorical(df_data)


# #### Author - Lokesh
# Tree Map
# Parallel Categories
# 
# Author : Lokesh

# In[31]:


get_correlation_treemap(df_data)


# In[32]:


get_correlation_parallel(df_data)


# ####  Author - Nikhil
#     - Box plot
#     - Violin charts
# 

# In[9]:


get_box_plots(df_data, 'loan_status', NUMERICAL_COLUMNS)


# In[34]:


get_violin_plots(df_data, 'loan_status', NUMERICAL_COLUMNS)


# #### Normalizing the data
# 
# Author : Mahesh Chandra Mareedu

# In[35]:


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
df_data[NUMERICAL_COLUMNS]=scaler.fit_transform(df_data[NUMERICAL_COLUMNS])


# In[36]:


df_data


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

# In[37]:


get_ipython().system('jupyter nbconvert CMPE*.ipynb --to python')


# ### Rough

# In[38]:


df_data['person_age'].astype(int).max()

