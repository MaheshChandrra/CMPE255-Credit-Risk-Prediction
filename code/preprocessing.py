import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import numpy as np
from IPython.display import display

def check_missing_columns(df_in):
    """
    Author : Mareedu Mahesh Chandra
    
    This function takes in a dataframe and prints the missing values in each column.
        -Some columns have "?" as data,considering them as missing values.
    
    Params:
    -------------------
    input: df_in
           =>dataframe
    -------------------
    
    output: returns list of missing columns
    
    """
    missing_columns_list=[]
    print("\n[INFO] Checking missing values in features\n")
    for col in df_in.columns:
        print("[INFO] Feature Selected:",col)
        missing_data_count=df_in[col].isna().sum()
        print("[INFO] Total nan's:",missing_data_count)
        
        values_with_palce_holder=len(df_in[df_in[col]=='?'])
        print("[INFO] Total '?' :",values_with_palce_holder)
        
        
        missing_data_count=missing_data_count+values_with_palce_holder
        if missing_data_count>0:
            print("[INFO] Total missing data in "+col+":"+str(missing_data_count))
            missing_columns_list.append(col)
            print("#"*100)
        else:
            print("[INFO] No missing data for col:",col)
            print("#"*100)
    return missing_columns_list





def visualize_pdf(df_in,col,bool_describe_data):
    """
    
    Author : Mareedu Mahesh Chandra
    
    This function takes in a dataframe and visualize data distribution(Probability Density Function)
        -Some columns have "?" as data,removing them while visualizing
    
    Params: 
    -------------------
    input: df_in
           =>Source dataframe
           
           col:
           =>Column/Feature for which distribution visualization is required
           
           bool_describe_data:
           => Boolean flag to describe properties of the feature,True to print.
    -------------------
    
    output: dataframe
            =>Dataframe with imputed values
    
    """
    
    ### Taking only feature values which are "not null" and "?"
    df_temp=df_in[df_in[col]!='?']
    df_temp=df_temp[~df_temp[col].isna()]
    
    df_temp[col]=df_temp[col].astype(float)
    
    
    
    hist_data=[df_temp[col].values]
    label=[col]
    
    if bool_describe_data:
        print("[INFO] Describing data for feature:",col)
        display(df_temp[[col]].describe().T)
    
#     fig = ff.create_distplot(hist_data, label)
#     title='Dist Plot for feature '+ col
#     fig.update_layout(title=title) 
#     fig.show()
    
    ##Seaborn plot
    sns.kdeplot(data=df_temp, x=col)
    plt.show()
    
def impute_missing_values(df_in,features_with_missing_values):
    """
    Author : Mareedu Mahesh Chandra
    
    This function takes in a dataframe and imputes the missing values with Mean.
        -Some columns have "?" as data,considering them as missing values.
    
    Params:
    -------------------
    input: df_in
           =>Source dataframe
           
           features_with_missing_values
           =>List of features which require imputation
    -------------------
    
    output: dataframe
            =>Dataframe with imputed values
            
            imputed_value_dict
            =>A dictionary containing features and their imputed values
    
    """
    imputed_value_dict={}
    print("\n[INFO] Imputing missing values\n")
    print("#"*10)
    for col in features_with_missing_values:
        print("[INFO] Imputing :",col)
        df_in[col]=df_in[col].replace('?',np.nan)
        df_in[col]=df_in[col].astype(float)
        mean_value=np.mean(df_in[col])
        df_in[col]=df_in[col].fillna(mean_value)
        print("[INFO] Imputing with Mean:",mean_value)
        imputed_value_dict[col]=mean_value
        
    return df_in,imputed_value_dict
    
    
