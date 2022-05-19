# -*- coding:utf-8 -*-
import pandas as pd
import datetime
import sys

# Replace missing values by interpolation
def replace_missing (attribute):
    return attribute.interpolate(inplace=True)

def missing_data_preprocess(df, target_col):
    # missing data filter columnwise
    df = df[df[target_col].notna()]

    notnulldata = df.notnull().sum()
    #columns_min = notnulldata[notnulldata > past_history + future_target + 5].index.tolist()
    columns_60 = notnulldata[notnulldata > df.shape[0]*0.6].index.tolist()
    #columns = [col for col in columns_min if col in columns_60]
    df =  df[columns_60]

    col_with_missing = df.isnull().sum()[df.isnull().sum()>0].index.tolist()
    for col in col_with_missing:
        replace_missing(df.loc[:, col])
    df.fillna(method='bfill', inplace=True)
    df.dropna(inplace=True)
    return df

def standardize(df):
    """ dataframe standardize """
    dataset = df.values
    TRAIN_SPLIT = df.shape[0] // 5 * 4 
    # standardize the dataset using the mean and standard deviation of the training data
    data_mean = dataset[:TRAIN_SPLIT].mean(axis=0)
    data_std = dataset[:TRAIN_SPLIT].std(axis=0)
    data_std[data_std==0]=1
    # if target = constant, mean = 0, std = 1
    if data_std.shape[0] == 1 and data_std == 0:
        data_std = np.array([1.])
        data_mean = np.array([0])
    else:
        data_std = data_std
        data_mean = data_mean
    dataset = (dataset-data_mean) / data_std
    return dataset, data_mean, data_std

def data_lack_amount(df, past_history, future_target):
    if df.shape[0] < 5 * (past_history + future_target) + 1:
        print("df shape: {}".format(df.shape))
        lack_data = 5 * (past_history + future_target) + 1 - df.shape[0]
        print("still need {} data, ".format(5 * (past_history + future_target) + 1 - df.shape[0]))
        print("Equivalently, we need to wait {}.".format(str(datetime.timedelta(seconds=lack_data*900))))
        sys.exit()
    return df

def inference_data_lack(df, past_history, future_target):
    if df.shape[0] < past_history + future_target + 1:
        print("df.shape: {}".format(df.shape))
        lack_data = past_history + future_target + 1 - df.shape[0]
        print("still need {} data.".format(lack_data))
        print("Equivalently, we need to wait {}.".format(str(datetime.timedelta(seconds=lack_data*900))))
        sys.exit()
    return df
