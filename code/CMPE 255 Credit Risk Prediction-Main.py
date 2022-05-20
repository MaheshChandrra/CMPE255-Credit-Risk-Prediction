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
import app_models


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


file_path=properties.DATASET_DIR+properties.DATASET_FILENAME


# ### Reading dataset
# 
# Author: Mahesh Chandra Mareedu

# In[4]:


df_data=read_dataset(properties.DATASET_DIR+properties.DATASET_FILENAME)


# In[5]:


loan_status_dict={"0":"Not Default","1":"Default",0:"Not Default",1:"Default"}
df_data['loan_status_num'] = df_data['loan_status']
df_data['loan_status']=df_data['loan_status'].apply(lambda x : loan_status_dict[x])


# ### Saving File
# 
# Author: Mahesh Chandra Mareedu

# In[6]:


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

# In[7]:


print("Length of data:",len(df_data))
df_data=df_data.drop_duplicates().reset_index(drop=True)
print("Length of data  after dropping duplicates:",len(df_data.drop_duplicates()))


# #### Checking for missing data
# 
# Author: Mahesh Chandra Mareedu

# In[8]:


missing_columns_list=check_missing_columns(df_data)


# In[9]:


print("Missing data in columns:",missing_columns_list)


# #### Visualizing distribution of data for missing columns
# 
# Author: Mahesh Chandra Mareedu

# In[10]:


for col in missing_columns_list:
    visualize_pdf(df_data,col,True)


# #### Imputing Missing Values
# 
# Author: Mahesh Chandra Mareedu

# In[11]:


df_data,imputed_value_dict=impute_missing_values(df_data,missing_columns_list)


# In[12]:


imputed_value_dict


# In[13]:


df_impute=pd.DataFrame(imputed_value_dict.items(), columns=['Column', 'Mean Value'])


# In[14]:


df_impute


# In[15]:


df_data.columns


# In[16]:


df_data.columns


# #### Converting datatype object to float for all numerical columns

# In[17]:


NUMERICAL_COLUMNS=['person_age', 'person_income','person_emp_length', 'loan_amnt',
       'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length']
TARGET_LABEL='loan_status'


# In[18]:


for col in NUMERICAL_COLUMNS:
    df_data[col]=df_data[col].astype(float)


# ####  Author - Nikhil
#     - Box plot
#     - Violin charts
# 

# In[19]:


get_box_plots(df_data, 'loan_status', NUMERICAL_COLUMNS)


# In[20]:


get_violin_plots(df_data, 'loan_status', NUMERICAL_COLUMNS)


# #### Removing Outliers

# ##### person_age
# 
# Considering the average persons age is around 80,discarding all the values where age is greater than 80.
# 

# In[21]:


df_data[df_data['person_age']>80]


# In[22]:


df_data=df_data[df_data['person_age']<80].reset_index(drop=True)


# ##### person_emp_length Age
# 
# Considering the the retirement period is 60 years,max employement for a person would be 40-45 yrs if he/she starts working around 15-20.Discarding where employment period is greater than 41.

# In[23]:


df_data[df_data['person_emp_length']>41]


# In[24]:


df_data=df_data[df_data['person_emp_length']<41].reset_index(drop=True)


# In[25]:


df_data


# #### Distribution plot on numerical data
# 
# Author : Mahesh Chandra Mareedu

# In[26]:


TARGET_LABEL='loan_status'
get_distplot(df_data,NUMERICAL_COLUMNS,TARGET_LABEL,True)


# In[27]:


get_distplot(df_data,NUMERICAL_COLUMNS,TARGET_LABEL,False)


# #### Correlation plot around numerical features
# 
# **Observations** : cb_person_cred_hist_length and person_age have high colleniarity from below plots,so dropping  cb_person_cred_hist_length.
# 
# Author : Mahesh Chandra Mareedu

# In[28]:


get_correlation_heatmap(df_data)


# In[29]:


df_data


# In[30]:


get_correlation_pairplot(df_data[NUMERICAL_COLUMNS])


# In[31]:


get_correlation_pairplot(df_data[['person_age', 'person_income', 'loan_amnt','loan_status']],'loan_status')


# In[32]:


NUMERICAL_COLUMNS.remove('cb_person_cred_hist_length')


# In[33]:


NUMERICAL_COLUMNS


# ####  Author : Shanmuk
# 
# Bar charts
#        
# - On counts of categorical columns
# - On categorical columns by target label
# 
# 

# In[34]:


get_barplot(df_data)


# In[35]:


# get_barplot_catagorical(df_data)


# #### Author - Lokesh
# Tree Map
# Parallel Categories
# 
# Author : Lokesh

# In[ ]:


get_correlation_treemap(df_data)


# In[ ]:


get_correlation_parallel(df_data)


# ### Result dictionary to track results from every model

# In[ ]:


result_dict={}


# ### Random Forest Classifier 
# Author: Nikhil Kumar Kanisetty

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


import app_models


# In[ ]:


df_in = df_data.copy()
loan_status_dict={"Not Default":0,"Default":1}
df_in['loan_status']=df_in['loan_status'].apply(lambda x : loan_status_dict[x])


# In[ ]:


CATEGORICAL_COLUMNS = ["person_home_ownership","loan_intent","loan_grade","cb_person_default_on_file"]


# In[ ]:


target_column = "loan_status"


# In[ ]:


result_dict=app_models.apply_RFC(df_in, target_column, CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS,result_dict)


# ### Applying Decision Tree Classifier
# 
# Author : Lokesh Vaddi

# In[ ]:


import app_models
df_in = df_data.copy()

loan_status_dict={"Not Default":0,"Default":1}
df_in['loan_status']=df_in['loan_status'].apply(lambda x : loan_status_dict[x])

CATEGORICAL_COLUMNS = ["person_home_ownership","loan_intent","loan_grade","cb_person_default_on_file"]
target_column = "loan_status"


result_dict=app_models.apply_dt(df_in, target_column, CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS,result_dict)


# #### Normalizing the data
# 
# Author : Mahesh Chandra Mareedu

# In[ ]:


from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
df_data[NUMERICAL_COLUMNS]=scaler.fit_transform(df_data[NUMERICAL_COLUMNS])


# In[ ]:


df_data


# ### Model Building

# #### XGBoost
# 
# Author: Mareedu Mahesh Chandra
# 
# This funtion performs:
#         
#         1.One hot encoding
#         2.Train test split
#         3.Upsampling
#         4.Training XGBoost
#         5.Testing model

# In[ ]:


import warnings
warnings.filterwarnings('ignore')


# In[ ]:


import app_models
df_in=df_data.copy()
loan_status_dict={"Not Default":0,"Default":1}
df_in['loan_status']=df_in['loan_status'].apply(lambda x : loan_status_dict[x])
result_dict=app_models.apply_XGBoost(df_in,target_column,CATEGORICAL_COLUMNS,NUMERICAL_COLUMNS,result_dict)


# ### Results

# In[ ]:


df_res=pd.DataFrame(data=result_dict,index=["Accuracy","Precision","Recall","F1-Score"]).T


# In[ ]:


df_res


# ### Convert notebook to app.py
# 
# 

# In[ ]:


get_ipython().system('jupyter nbconvert CMPE*.ipynb --to python')


# ### Rough

# In[ ]:


df_data['person_age'].astype(int).max()

