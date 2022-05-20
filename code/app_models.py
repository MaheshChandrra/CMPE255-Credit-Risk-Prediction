import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.figure_factory as ff
import numpy as np
from IPython.display import display
import plotly.express as px
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold, cross_validate, train_test_split
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectFromModel

import warnings
warnings.filterwarnings('ignore')


def show_results(X_tr,Y_tr,X_tst,y_tst,classifier):
    
    """
    
    Author : Mareedu Mahesh Chandra

    This funtion plots confusion matrix and provides classification report.
    
    Params:
    -----------------
    
    X_tr,Y_tr,X_tst,y_tst
    => Train and Test sets
    
    
    classifier
    =>An ML classifier used for prediction.
    

    -----------------
    

    """
    
    print("Results on Train:\n")
    y_tr_prd = classifier.predict(X_tr)
    
    print(classification_report(Y_tr, y_tr_prd))
    #plot_confusion_matrix(classifier, X_tr, Y_tr)
    plt.show()
    
    print("#"*100)
    
    print("Results on Test:\n")
    y_tst_pred = classifier.predict(X_tst)
    print(classification_report(y_tst, y_tst_pred))
    #plot_confusion_matrix(classifier, X_tst, y_tst) 
    plt.show()


def create_train_test_split(df_in,target_column):
    """
    
    Author : Mareedu Mahesh Chandra

    This funtion creates train test split on the data
    
    Params:
    -----------------
    
    df_in
    => Input dataset
    
    
    target_column
    =>A target column used to create split on data,for separating features and target varibles.
    

    
    Returns:
    -----------------
    Train test splits
    
    """
    

    
    X_train, X_test, y_train, y_test = train_test_split(df_in.loc[:, df_in.columns !=target_column ],
                                                    df_in[target_column],stratify=df_in[target_column], 
                                                    test_size = 0.30, random_state = 100)
    return X_train, X_test, y_train, y_test
    
def upsample(X_in,y_in):
    
    """
    
    Author : Mareedu Mahesh Chandra

    This funtion upsamples minorty class records and balances with majority class records.
    
    Params:
    -----------------
    
    X_in,y_in
    => features,corresponding target variables
    

    
    Returns:
    -----------------
    Balanced X_train and y_train
    
    """
    
    
    print("[INFO] Applying Upsampling")
    oversample = SMOTE()
    X_in_upsampled, y_in_upsampled = oversample.fit_resample(X_in, y_in)
    print("[INFO] Upsampling Completed")
    
    return X_in_upsampled, y_in_upsampled

def apply_one_hot_encoding(df_in,CATEGORICAL_COLUMNS,NUMERICAL_COLUMNS):
    
    """
    
    Author : Mareedu Mahesh Chandra

    This funtion applies one hot encoding on categorical features of the dataset.
    
    Params:
    -----------------
    
    df_in
    => Input dataframe
    
    CATEGORICAL_COLUMNS
    => List of categorical columns
    
    NUMERICAL_COLUMNS
    => List of numerical columns
    

    
    Returns:
    -----------------
    dataframe with encoded values.
    
    """
    
    
    print("[INFO] Applying One hot encoding")
    ohe = OneHotEncoder(categories='auto')
    feature_arr = ohe.fit_transform(df_in[CATEGORICAL_COLUMNS]).toarray()
    feature_labels = ohe.categories_
    df_encoded=pd.DataFrame(data=feature_arr,columns=ohe.get_feature_names_out())
    df_final=df_in[NUMERICAL_COLUMNS].join(df_encoded)
    df_final['loan_status']=df_in['loan_status']
    
    print("[INFO] One hot encoding completed")
    
    
    return df_final
    
    
    
    
def apply_XGBoost(df_in,target_column,CATEGORICAL_COLUMNS,NUMERICAL_COLUMNS):
    
    """
    
    Author : Mareedu Mahesh Chandra

    This funtion performs:
        
        1.One hot encoding
        2.Train test split
        3.Upsampling
        4.Training XGBoost
        5.Testing model
        
    Params:
    -----------------
    
    df_in
    => Input dataframe
    
    target_column
    =>target column in the dataset
    
    CATEGORICAL_COLUMNS
    => List of categorical columns
    
    NUMERICAL_COLUMNS
    => List of numerical columns
    
    
    """
    
    
    ### Applying One hot encoding
    df_in=apply_one_hot_encoding(df_in,CATEGORICAL_COLUMNS,NUMERICAL_COLUMNS)
    
    ### Creatig Train test splits
    X_train, X_test, y_train, y_test=create_train_test_split(df_in,target_column)
    
    
    ### Upsamplign the data
    X_train_upsampled,y_train_upsampled=upsample(X_train,y_train)
    
    
    ### Applying XGBoost
    xgb_clf = xgb.XGBClassifier()
    xgb_clf.fit(X_train_upsampled , y_train_upsampled)
    
    show_results(X_train_upsampled,y_train_upsampled,X_test,y_test,xgb_clf)
    
    
    
def apply_RFC(df_in,target_column,CATEGORICAL_COLUMNS,NUMERICAL_COLUMNS):
    
    """
    Author: Nikhil Kumar Kanisetty

    This function performs:
    - Splitting train and test
    - Upsample the data since the data is imbalanced
    - Train a Random Forest Classifier
    - Predict using the above model

    params:
    df -> input_df
    target -> target_column
    CATEGORICAL_COLUMNS -> all the categorical columns in the data
    NUMERICAL_COLUMNS -> all the numereical columns in the data
    """
    df_in = apply_one_hot_encoding(df_in, CATEGORICAL_COLUMNS, NUMERICAL_COLUMNS)

    X_train, X_test, y_train, y_test = create_train_test_split(df_in, target_column)

    X_train_upsampled, y_train_upsampled = upsample(X_train, y_train)

    rf = RandomForestClassifier(max_depth = 3, random_state = 100)
    rf.fit(X_train_upsampled, y_train_upsampled)

    show_results(X_train_upsampled, y_train_upsampled, X_test, y_test, rf)
    
#     grid_search = GridSearchCV(RandomForestClassifier(), {
#                     "n_estimators": range(100, 300, 100),
#                     "max_depth": range(2, 20, 2),
#                     "max_features": range(1, 4),
#                     'criterion': ["gini", "entropy"]
#                     }, cv = 5, n_jobs = -1, verbose = 2)
#     grid_search.fit(X_train_upsampled, y_train_upsampled)
#     print(grid_search.best_params_)

#    {'criterion': 'entropy', 'max_depth': 18, 'max_features': 3, 'n_estimators': 200}
    
    rf = RandomForestClassifier(n_estimators = 200, max_features = 3, max_depth = 18, criterion = "entropy", random_state = 100)
    rf.fit(X_train_upsampled, y_train_upsampled)
    
    show_results(X_train_upsampled, y_train_upsampled, X_test, y_test, rf)
    
    select_features = SelectFromModel(RandomForestClassifier(n_estimators = 200, max_features = 3, max_depth = 18, criterion = "gini", random_state = 100))
    select_features.fit(X_train_upsampled, y_train_upsampled)
    
    df = df_in.drop(columns = 'loan_status', axis = 1)
    columns = df.columns[(select_features.get_support())]
    
    print(columns)
    
    X_train_upsampled = X_train_upsampled[columns]
    X_test = X_test[columns]
    
    rf.fit(X_train_upsampled, y_train_upsampled)
    
    show_results(X_train_upsampled, y_train_upsampled, X_test, y_test, rf)
    


