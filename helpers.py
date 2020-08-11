###########################################
# Suppress matplotlib user warnings
# Necessary for newer version of matplotlib
import warnings
warnings.filterwarnings("ignore", category = UserWarning, module = "matplotlib")
#
# Display inline matplotlib plots with IPython
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')
###########################################

import matplotlib.pyplot as pl
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score, accuracy_score

def feature_plot(importances, X_train, y_train, top_k=5):
    
    # Display the five most important features
    indices = np.argsort(importances)[::-1]
    columns = X_train.columns.values[indices[:top_k]]
    values = importances[indices][:top_k]

    # Creat the plot
    fig = pl.figure(figsize = (15,20))
    pl.title(f"Normalized Weights for First {top_k} Most Predictive Features", fontsize = 16)
    #pl.bar(np.arange(top_k), values, width = 0.6, align="center", color = '#00A000', \
    #      label = "Feature Weight")
    pl.barh(np.arange(top_k), values[::-1], align="center", height=0.4, label = "Feature Weight")
    #pl.bar(np.arange(top_k) - 0.3, np.cumsum(values), width = 0.2, align = "center", color = '#00A0A0', \
    #      label = "Cumulative Feature Weight")
    pl.barh(np.arange(top_k) - 0.3, np.cumsum(values)[::-1], height=0.4, align="center", label = "Cumulative Feature Weight") 
            
    pl.yticks(np.arange(top_k), columns[::-1])
    pl.ylim((-0.5, top_k-.5))
    pl.xlabel("Weight", fontsize = 12)
    pl.ylabel("Feature", fontsize = 12)
    
    pl.legend(loc = 'upper right')
    pl.tight_layout()
    pl.show()
    
    return pd.DataFrame({'Features':columns, 'Importance value':values})

def datetime_feature_engineering(df, inplace = False):
    """
        This function generates year, month and day for each datetime feature
    """
    date_indx = np.where(df.dtypes=='<M8[ns]')[0]
    df_temp = df.copy(deep=True)
    for i in date_indx:
        curr_column = df_temp.columns[i]
        df_temp[curr_column+"_Year"] = df_temp[curr_column].dt.year
        df_temp[curr_column+"_Month"] = df_temp[curr_column].dt.month
        df_temp[curr_column+"_Day"] = df_temp[curr_column].dt.day
    
    return df_temp.drop(list(df_temp.columns[date_indx]), 1, inplace=inplace)

def oneHotEnc_to_classes(predictions, column_names):
    class_dict = {0:column_names[0][-1], 1:column_names[1][-1], 2:column_names[2][-1], 3:column_names[3][-1], 4:column_names[4][-1]}
    return np.vectorize(class_dict.get)(np.argmax(predictions, axis=1))