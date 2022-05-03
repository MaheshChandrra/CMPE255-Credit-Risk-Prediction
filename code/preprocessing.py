import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import numpy as np
from IPython.display import display
import plotly.express as px


IMG_DIR="../paper/images/"

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
    graph_file=IMG_DIR+'dist_plot_'+col+'.png'
    plt.savefig(graph_file)
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

def get_correlation_heatmap(df_in):
    """
    Author : Mareedu Mahesh Chandra
    
    This function takes in a dataframe visualizes heap map with correlation values among the numerical features.
    
    Params:
    -------------------
    input: df_in
           =>dataframe
    -------------------
    
    
    """
    ax = sns.heatmap(df_in.corr(), cmap="YlGnBu",annot=True)
    sns.set(rc = {'figure.figsize':(20,20)})
    graph_file=IMG_DIR+'coorelation_heapmatap.png'
    plt.savefig(graph_file)
    plt.show()
    
def get_correlation_pairplot(df_in,target_label=None):
    """
    Author : Mareedu Mahesh Chandra
    
    This function takes in a dataframe visualizes pair plots to understand positive/negative correlation among features.
    
    Params:
    -------------------
    input: df_in
           =>dataframe
           
           target_label
           => If target is provided then pair plot would datapoints divided by target label
    -------------------
    
    
    """
#     ax = sns.heatmap(df_in.corr(), cmap="YlGnBu",annot=True)
#     graph_file=IMG_DIR+'coorelation_pairplot.png'
#     plt.savefig(graph_file)
#     plt.show()
    if target_label==None:
        #fig = px.scatter_matrix(df_in)
        #fig.show()
        sns.pairplot(df_in)
        plt.show()
        
    else:
        #dimensions=df_in.columns.tolist()
        #dimensions.remove(target_label)
        #fig = px.scatter_matrix(df_in,dimensions=dimensions,color=target_label)
        #fig.show()
        sns.pairplot(df_in, hue=target_label)
        plt.show()
        

        
def get_distplot(df_in,feature_list,target_label,bool_plot_by_target):
    """
    Author : Mareedu Mahesh Chandra
    
    This function takes in a dataframe visualizes dist plots to understand data distribution of a feature data.
    Also saves plots to iamge directory. 
    
    Params:
    -------------------
    input: df_in
           =>dataframe
           
           feature_list
           =>List of features whose distribtuion is to be analyzed
           
           target_label
           =>Target variable in dataset
           
           bool_plot_by_target
           => Flag ,if set true,all the feature disrtibutions are visualized by target varible
    -------------------
    
    output:
        =>Dist plot
    
    """
    #feature_list.remove('loan_status')
    n_rows=4
    n_cols=2

    if bool_plot_by_target:

        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols)
        
        for i, col in enumerate(feature_list):
            sns.histplot(df_in, x=col, hue=target_label, kde=True, stat='density', fill=True,ax=axes[i//n_cols,i%n_cols])
            graph_file=IMG_DIR+col+'_dist_plot_by_'+target_label+'.png'
        plt.savefig(graph_file)
    else :
        fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols)
        for i, col in enumerate(feature_list):
            sns.kdeplot(df_in[col],ax=axes[i//n_cols,i%n_cols])
            graph_file=IMG_DIR+col+'_dist_plot.png'
        plt.savefig(graph_file)


def get_correlation_treemap(df_in):
    """
    Author : Lokesh Vaddi
    
    This function takes in a dataframe visualizes TreeMap to understand hierarchical correlation among features.
    
    Params:
    -------------------
    input: df_in
           =>dataframe
           
           
    -------------------
    
    
    """
    TreeMap = px.treemap(df_in, path=['person_age','person_income', 'loan_amnt'])
    
    TreeMap.show()


def get_correlation_parallel(df_in):
    """
    Author : Lokesh Vaddi
    
    This function takes in a dataframe visualizes parallel to understand correlation between data.
    
    Params:
    -------------------
    input: df_in
           =>dataframe
           
           
    -------------------
    
    
    """
    correlation_parallel = px.parallel_categories(df_in, dimensions=['loan_intent', 'loan_grade'])
    correlation_parallel.show()
    correlation_parallel = px.parallel_categories(df_in, dimensions=['loan_intent', 'loan_grade', 'loan_status'])
    correlation_parallel.show()




def get_correlation_between_data(df_in):
    """
    Author : Lokesh Vaddi
    
    This function takes in a dataframe and returns the correlation between all the columns.
    
    Params:
    -------------------
    input: df_in
           =>dataframe
           
           
    -------------------
    """
    plt.figure(figsize=(10,8))
    sns.heatmap(df_in.corr(), annot=True, cmap="YlGnBu")
    plt.title("Correlations Between Features", size=15)
    plt.show()



def get_barplot(df_in):
    """
    Author : Shanmuk 
    
    This function takes in a dataframe and returns the barplot.
    
    Params:
    -------------------
    input: df_in
           =>dataframe
           
           
    -------------------
    """

    #df_categorical_col['loan_grade'].value_counts().iplot(kind='bar')
    df_catagorical_col=df_in.select_dtypes(include=['object'])

    for i in df_catagorical_col:
        fig = px.bar(df_catagorical_col[i].value_counts())
        fig.show()
        
        
def get_barplot_catagorical(df_in):
    """
    Author : Shanmuk
    
    This function takes in a dataframe and displays the barplot with target variable loan status.
    
    Params:
    -------------------
    input: df_in
           =>dataframe
           
           
    -------------------
    """

    df_catagorical_col=df_in.select_dtypes(include=['object'])
    for i in df_catagorical_col:
        plt.figure(figsize=(12,5))
        count_plot=sns.countplot(data=df_catagorical_col, y= df_catagorical_col[i],hue=df_in['loan_status'].astype(int),palette='RdYlBu_r')
        count_plot.bar_label(count_plot.containers[0])
        count_plot.bar_label(count_plot.containers[1])
        plt.show()
        
def get_box_plots(df_in, x_col, y_cols):
    """
    Author : Nikhil Kumar Kanisetty
    
    This function takes in a dataframe and visualizes box plots
    
    Params:
    ---------------------
    input: df_in
           => dataframe
            
           
           => target 
    """
    for col in y_cols:
        if col != "loan status":
            plt.figure(figsize=(12,5))
            sns.boxplot(x = x_col, y = col, data = df_in)
            plt.show()

    
def get_violin_plots(df_in, x_col, y_cols):
    """
    Author : Nikhil Kumar Kanisetty

    This function takes in a dataframe and visualizes violin plots

    Params:
    ---------------------
    input: df_in
           => dataframe


           => target 
    """
    for col in y_cols:
        if col != "loan status":
            plt.figure(figsize=(12,5))
            sns.violinplot(x = x_col, y = col, data = df_in)
            plt.show()

