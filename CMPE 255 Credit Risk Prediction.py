#!/usr/bin/env python
# coding: utf-8

# ### Importing modules

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


import os


# ### Reading dataset

# In[6]:


os.listdir('dataset')


# In[7]:


DATASET_DIR='dataset/'
DATASET_FILENAME='dataset_'


# In[20]:


data_flag=False
data=[]
features=[]
with open(DATASET_DIR+DATASET_FILENAME) as f:
    all_lines=f.readlines()
    for line in all_lines:
        if '@ATTRIBUTE' in line:
            _,feature_name,_=line.split(' ')
            features.append(feature_name)
        if data_flag:
            line=line.replace("\n",'')
            data.append(line.split(','))
        if '@DATA' in line:
            data_flag=True          


# In[22]:


df_data=pd.DataFrame(data=data,columns=features)


# In[24]:


df_data.head()


# In[25]:


df_data.info()


# In[26]:


df_data.describe()


# #### Writing dataset back to excel

# In[27]:


df_data.to_excel(DATASET_DIR+'Credit_Risk_Prediction_Dataset.xlsx')

