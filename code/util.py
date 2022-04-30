import pandas as pd
import numpy as np
import matplotlib.pyplot


def read_dataset(file_path):
    """
    Author : Mahesh Chandra Mareedu

    This function is used to read data from file and produce dataframe with file contents.

    Input: file_path

    Output: Dataframe with file contents

    """
    data_flag=False
    data=[]
    features=[]
    with open(file_path) as f:
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
    df_data=pd.DataFrame(data=data,columns=features)
    return df_data

def save_file(df_in,dir_path):
    df_in.to_excel(dir_path+'Credit_Risk_Prediction_Dataset.xlsx')