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
from preprocessing import missing_data_preprocess, data_lack_amount, standardize
import warnings
warnings.filterwarnings("ignore")
import time
time_start=time.time()
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
    try:
        cor = SensorValue.corr()
        cor_target = abs(cor[predictSensor])
        relevant_features = list(cor_target[cor_target>0.7].index)
        if len(relevant_features) > 0:
            return relevant_features
        else:
            return [predictSensor]
    except Exception as err:
        print("target column is not in dataframe.", "\n", err)

parser = ArgumentParser()
parser.add_argument('-guid', type=str, nargs='?', help='guid')
parser.add_argument('-target',type=str, nargs='?', help='partition name which to be predicted')   
#parser.add_argument('-timeFreq', type=str, nargs='?', help='csv name timefreq')
parser.add_argument('-history_type', type=str, nargs='?', help='sensor or utilization')
parser.add_argument('-els_host', type=str, nargs='?', help='els host ip')
args = parser.parse_args()

past_history = 24*7*4
future_target = 24*4*7
STEP = 6 
BATCH_SIZE = 256
BUFFER_SIZE = 10000
EVALUATION_INTERVAL = 500

if args.history_type == "sensor":
    csv_name = args.guid + '_sensor_15m.csv'
    target_col = args.target
elif args.history_type == "utilization":
    csv_name = args.guid + '_uti_15m.csv'
    target_col = 'utili_' + args.target
else:
    print("--history_type is neither sensor nor utilization")
    sys.exit()

try:
    df = pd.read_csv("./csvfile/" + csv_name, index_col=0)
except Exception as err:
    print("read csv error \n", err) 
    sys.exit()

def data_lack_amount(df, past_history, future_target):
    if df.shape[0] < 5 * (past_history + future_target) + 1:
        print("df shape: {}".format(df.shape))
        lack_data = 5 * (past_history + future_target) + 1 - df.shape[0]
        print("still need {} data, ".format(5 * (past_history + future_target) + 1 - df.shape[0]))
        print("Equivalently, we need to wait {}.".format(str(datetime.timedelta(seconds=lack_data*900))))
        sys.exit()
    return df

df = missing_data_preprocess(df, target_col)
data_lack_amount(df, past_history, future_target)

related_columns=pearson_feature_selection(df, target_col)
df = df[related_columns]

target_column_index = df.columns.tolist().index(target_col)


#dataset = df.values
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

train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()

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

try:
    multi_step_model = tf.keras.models.Sequential()
    multi_step_model.add(tf.keras.layers.LSTM(32,
                                          dropout=0.1, recurrent_dropout=0.2,
                                          input_shape=x_train_multi.shape[-2:]))

    multi_step_model.add(tf.keras.layers.Dense(future_target))

    multi_step_model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=0.0008, clipvalue=1.0), 
                             loss='mae', metrics=['mae', error_rate_5percent])
except ValueError as err:
    print("model build fail, x_train_multi len: ", len(x_train_multi), '\n', err)
    print("no training data")

try:
    #callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5) 
    multi_step_history = multi_step_model.fit(train_data_multi, epochs=40,
                                          steps_per_epoch=EVALUATION_INTERVAL,
                                          validation_data=val_data_multi,
                                          validation_steps=50)

    multi_step_model.save("./model/" + args.guid + "_" + target_col + ".h5")
except ValueError as err:
    print("x_train_multi len:",len(x_train_multi), '\n', err)
    print("val_data_multi: ", val_data_multi)
    print()
except Exception as err:
    print(err)

# write info to els for infer
response = store_train_info(args.els_host, args.target, data_mean.tolist(), data_std.tolist(), related_columns, args.guid)
time_end=time.time()
print('time cost',time_end-time_start,'s')
