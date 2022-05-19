# -*- coding:utf-8 -*-
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import pandas as pd
import datetime
import sys
import tensorflow as tf
from store_data_for_infer import store_train_info
from argparse import ArgumentParser
from sklearn.model_selection import train_test_split, KFold
from preprocessing import missing_data_preprocess, data_lack_amount, standardize
import warnings
warnings.filterwarnings("ignore")
def multivariate_data(dataset, target, start_index, end_index, \
                      history_size, target_size, step, single_step=False):
    data = []
    labels = []
    
    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size
    
    for i in range(start_index, end_index):
        indices = range(i - history_size, i, step)
        data.append(dataset[indices])
        
        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])
    
    return np.array(data), np.array(labels)

def pearson_feature_selection(SensorValue, predictSensor):
    """ pearson correlation """
    cor = SensorValue.corr()
    cor_target = abs(cor[predictSensor])
    relevant_features = list(cor_target[cor_target>0.7].index)
    if len(relevant_features) > 0:
        return relevant_features
    else:
        return [predictSensor]

parser = ArgumentParser()
parser.add_argument('-guid', type=str, nargs='?', help='guid')
parser.add_argument('-target',type=str, nargs='?', help='partition name which to be predicted')   
#parser.add_argument('-timeFreq', type=str, nargs='?', help='csv name timefreq')
parser.add_argument('-els_host', type=str, nargs='?', help='els host ip')
args = parser.parse_args()

past_history = 24*7*4
future_target = 24*4*7
STEP = 6
BATCH_SIZE = 256
BUFFER_SIZE = 10000
EVALUATION_INTERVAL = 500

uti_csv_name = args.guid + '_uti_15m.csv'
target_col = 'utili_' + args.target

try:
    df_uti = pd.read_csv("./csvfile/" + uti_csv_name, index_col=0)
    df = df_uti
    #df_sensor = pd.read_csv("./csvfile/" + sensor_csv_name, index_col=0)

    #df = df_sensor.join(df_uti)
    #df.dropna(inplace=True)
except Exception as err:
    print("read csv and join error \n", err) 
    df = df_uti


df = missing_data_preprocess(df, target_col)
data_lack_amount(df, past_history, future_target)

related_columns=pearson_feature_selection(df, target_col)


df = df[related_columns]

target_column_index = df.columns.tolist().index(target_col)

TRAIN_SPLIT = df.shape[0] // 5 * 4
dataset, data_mean, data_std = standardize(df)

x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, target_column_index], 0,
                                                 TRAIN_SPLIT, past_history,
                                                 future_target, STEP)
x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, target_column_index],
                                             TRAIN_SPLIT, None, past_history,
                                             future_target, STEP)
if x_train_multi.shape[0]==0 or x_val_multi.shape[0]==0:
    print('data size too small')
    sys.exit()


def error_rate_5percent(y_true, y_pred, tol=0.05):
    # check tensor dtype
    diff = y_true - y_pred
    lst = tf.math.less(tf.abs(diff), tf.abs(y_true*tol))
    right_wrong_list = tf.dtypes.cast(lst, tf.float16)
    try:
        score = tf.reduce_mean(right_wrong_list)
    except Exception as error:
        print(error)
    return score


model_name = './model/' + args.guid + "utili_" + args.target + ".h5"
model = tf.keras.models.load_model(model_name, custom_objects=\
    {"error_rate_5percent": error_rate_5percent})

prediction = model.predict(np.array([x_val_multi[-1].tolist()]))
predictData = prediction * data_std[target_column_index] + data_mean[target_column_index]
real_y = y_val_multi[-1] * data_std[target_column_index] + data_mean[target_column_index]
print(predictData[0])

acc_list = []
for i in range(len(x_val_multi)):
    prediction = model.predict(np.array([x_val_multi[i].tolist()]))
    predictData = prediction * data_std[target_column_index] + data_mean[target_column_index]
    real_y = y_val_multi[i] * data_std[target_column_index] + data_mean[target_column_index]
    acc_list.append(error_rate_5percent(real_y, predictData[0]))

print("avg acc: {:.2f}".format(sum(acc_list)/len(acc_list)))

