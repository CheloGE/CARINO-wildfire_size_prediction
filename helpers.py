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

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import roc_auc_score
from keras.callbacks import Callback
from sklearn.metrics import f1_score, accuracy_score
import pickle

def feature_plot(importances, X_train, y_train, top_k=5):
    
    # Display the five most important features
    indices = np.argsort(importances)[::-1]
    columns = X_train.columns.values[indices[:top_k]]
    values = importances[indices][:top_k]

    # Creat the plot
    fig = plt.figure(figsize = (15,20))
    plt.title(f"Normalized Weights for First {top_k} Most Predictive Features", fontsize = 16)
    #plt.bar(np.arange(top_k), values, width = 0.6, align="center", color = '#00A000', \
    #      label = "Feature Weight")
    plt.barh(np.arange(top_k), values[::-1], align="center", height=0.4, label = "Feature Weight")
    #plt.bar(np.arange(top_k) - 0.3, np.cumsum(values), width = 0.2, align = "center", color = '#00A0A0', \
    #      label = "Cumulative Feature Weight")
    plt.barh(np.arange(top_k) - 0.3, np.cumsum(values)[::-1], height=0.4, align="center", label = "Cumulative Feature Weight") 
            
    plt.yticks(np.arange(top_k), columns[::-1])
    plt.ylim((-0.5, top_k-.5))
    plt.xlabel("Weight", fontsize = 12)
    plt.ylabel("Feature", fontsize = 12)
    
    plt.legend(loc = 'upper right')
    plt.tight_layout()
    plt.show()
    
    return pd.DataFrame({'Features':columns, 'Importance value':values})

def datetime_feature_engineering2(df, inplace = False):
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

def datetime_feature_engineering(df_temp, inplace = False):
    """
        This function generates new features based on datetime features
    """
    datetime_features = df_temp.columns[df_temp.dtypes=='<M8[ns]']
    df = df_temp.copy(deep=True)
    df['fire_duration'] = (df['ex_fs_date']-df['fire_start_date']).astype('timedelta64[m]')
    df['time_to_ex'] = (df['ex_fs_date']-df['bh_fs_date']).astype('timedelta64[m]')
    df['time_to_uc'] = (df['uc_fs_date']-df['bh_fs_date']).astype('timedelta64[m]')
    df['time_to_bh2'] = (df['bh_fs_date']-df['fire_fighting_start_date']).astype('timedelta64[m]')
    df['time_to_bh'] = (df['bh_fs_date']-df['fire_start_date']).astype('timedelta64[m]')
    #df['fire_fight_response_time'] = (df['fire_fighting_start_date']-df['discovered_date']).astype('timedelta64[m]')
    #df['time_to_discover'] = (df['discovered_date']-df['fire_start_date']).astype('timedelta64[m]')
    df['time_to_report'] = (df['reported_date']-df['discovered_date']).astype('timedelta64[m]')
    #df['time_to_assess'] = (df['start_for_fire_date']-df['discovered_date']).astype('timedelta64[m]')
    
    return df.drop(datetime_features, 1, inplace=inplace)

def oneHotEnc_to_classes(predictions, column_names):
    class_dict = {0:column_names[0][-1], 1:column_names[1][-1], 2:column_names[2][-1], 3:column_names[3][-1], 4:column_names[4][-1]}
    return np.vectorize(class_dict.get)(np.argmax(predictions, axis=1))

def parse_datetime_features_to_hours(df_temp):
    df = df_temp.copy(deep=True)
    datetime_columns = df.columns[df.dtypes=='<M8[ns]']
    for column in datetime_columns:
        df[column]=(df[column]-(df[column].min())).astype('timedelta64[h]')
    return df

class RocCallback(Callback):
    def __init__(self,training_data,validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]


    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred_train = self.model.predict_proba(self.x)
        roc_train = roc_auc_score(self.y, y_pred_train)
        y_pred_val = self.model.predict_proba(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc_train: %s - roc-auc_val: %s' % (str(round(roc_train,4)),str(round(roc_val,4))),end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
    
class F1Callback(Callback):
    def __init__(self,training_data,validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]
    def  on_train_begin(self,logs={}):
        self.f1_macro=[]
    def on_epoch_end(self, epoch, logs=None):
        y_pred_train=self.model.predict(self.x.reshape(-1,8,1)).round()
        score_train=f1_score(self.y, y_pred_train, average='macro')
        y_pred_val=self.model.predict(self.x_val.reshape(-1,8,1)).round()
        score_val=f1_score(self.y_val, y_pred_val, average='macro')
        print('\rf1_macro_train: %s - f1_macro_val: %s' % (str(round(score_train,4)),str(round(score_val,4))),end=100*' '+'\n')
    
def fill_datetime_with_neighbors(df):
    """
        Fills datetime features with dates that are close to the closest feature
    """
    df.fire_fighting_start_date.fillna(df.assessment_datetime, inplace=True)
    #df.fire_fighting_start_date.fillna(df.start_for_fire_date, inplace=True)
    df.reported_date.fillna(df.fire_start_date, inplace=True)
    df.discovered_date.fillna(df.reported_date, inplace=True)
    df.fire_start_date.fillna(df.discovered_date, inplace=True)
    #df.fire_start_date.fillna(df.ex_fs_date, inplace=True)
    
def save_model(path, model_to_save):
    """
        path should end in *.pkl extension
    """
    with open(path,'wb') as f:
        pickle.dump(model_to_save,f)
        
def load_model(path):
    """
        path should end in *.pkl extension
    """
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def fire_number_feature_engineering(df_input):
    df = df_input.copy(deep=True)
    df['Location'] = df['fire_number'].apply(lambda x: x[0])
    df['wildfires_in_location'] = df['fire_number'].apply(lambda x: int(x[-3:]))
    return df

def get_df_by_size_class(df_input, size_class):
    df = df_input.copy(deep=True)
    df = fire_number_feature_engineering(df)
    df.drop(['fire_number'], 1, inplace=True)
    size_class_indx = df['size_class']==size_class
    non_date_columns_indx = ['date' not in column for column in df.columns]
    non_date_columns = df.columns[non_date_columns_indx]
    df_size_class = df[non_date_columns][size_class_indx]
    return df_size_class

def data_analysis_by_size_class(df_input):
    df = df_input.copy(deep=True)
    classes = df['size_class'].unique()
    df_array = [get_df_by_size_class(df, curr_class) for curr_class in classes]
    for feature in df_array[0].columns:
        feature_pd_series = [curr_df[feature] for curr_df in df_array]
        if feature_pd_series[0].dtype=='O':
            plt.figure(figsize=(15,20))
            for i in range(len(feature_pd_series)):
                plt.subplot(int(511+i))
                plt.title(feature+'\n class : '+classes[i])
                if feature == 'det_agent':
                    feature_pd_series[i].value_counts()[:20].plot(kind='barh');
                else:
                    feature_pd_series[i].value_counts().plot(kind='barh');
        else:
            plt.figure(figsize=(15,7))
            for i in range(len(feature_pd_series)):
                plt.subplot(int(511+i))
                plt.title(feature+'\n class : '+classes[i])
                plt.boxplot(feature_pd_series[i], vert=False); 
            
    return df_array