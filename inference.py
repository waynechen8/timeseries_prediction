# -*- coding: utf-8 -*-
import pandas as pd
from elasticsearch import Elasticsearch
import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
import pandas as pd
import datetime
import tensorflow as tf
from influxdb import DataFrameClient
from argparse import ArgumentParser
from preprocessing import missing_data_preprocess, inference_data_lack
#import time

#time_start=time.time()

def multivariate_data(dataset, target, start_index, end_index, \
                      history_size, target_size, step, single_step=False):
    """ dataframe to windows np.array """
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

def predict_by_timefreq(prediction, timefreq):
    """ prediction according to time frequency, min frequency is 15m """
    groupby_count = 1
    if timefreq == "15min":
        return prediction
    elif timefreq == "30min":
        groupby_count = 2
    elif timefreq == "1h":
        groupby_count = 4
    elif timefreq == "6h":
        groupby_count = 24
    try:    
        group_list = [prediction[i:i+groupby_count] for i in range(0, 24*4*7, groupby_count)]
        mean_list = [sum(i)/groupby_count for i in group_list]
        return mean_list
    except Exception as err:
        print(err)

parser = ArgumentParser()
parser.add_argument('-guid', type=str, nargs='?', help='guid')
parser.add_argument('-target',type=str, nargs='?', help='partition name which to be predicted')   
parser.add_argument("-influx_host",    type=str, nargs="?", help="influxdb ip", default="10.10.75.106")
parser.add_argument("-influx_passwd",  type=str, nargs="?", default="!QAZ1qaz")
parser.add_argument("-database", type=str, nargs="?")
parser.add_argument("-startTime",      type=str, nargs="?", \
    help="query time start, YYYY-MM-DDTHH:mm:ssZ", default="2020-10-01T00:00:00Z")
parser.add_argument("-endTime",        type=str, nargs="?", \
    help="query time end, YYYY-MM-DDTHH:mm:ssZ", default="2020-12-31T00:00:00Z")
parser.add_argument('-history_type', type=str, nargs='?', help='sensor or utilization')
#parser.add_argument("-timeFreq", type=str, nargs="?", help="utilization data time frequency")
parser.add_argument("-output_freq", type=str, nargs="?", help="output frequency, raw data is per 15min, options: 30min, 1h, 6h")
args = parser.parse_args()

past_history = 24*7*4
future_target = 24*4*7
STEP = 6
BATCH_SIZE = 256
BUFFER_SIZE = 10000
EVALUATION_INTERVAL = 500

client = DataFrameClient(host=args.influx_host, port=8086, database=args.database, username='root', password=args.influx_passwd)

es = Elasticsearch([{"host" : args.influx_host, "port": 9200}], http_auth=("elastic", "!QAZ1qaz"))
info = es.get(index="history_predict", doc_type="_doc", id=args.guid)
related_columns = info["_source"][args.target]["feature_columns"]

if args.history_type == "sensor":
    target_col = args.target
    sensor_columns = [elt for elt in related_columns if 'utili_' not in elt]
    query = ['SELECT mean(Value) FROM ', args.database, '.HardwareSensorRetention.SensorReading where time>=\'', 
        args.startTime, '\' and time<=\'', args.endTime, '\' and ', '(\"Name\"= \'' + '\' or \"Name\"= \''.join(sensor_columns) + '\')',
        ' and ClientGUID=\'', args.guid, '\' GROUP BY time(15m), "Name"']
elif args.history_type == "utilization":
    target_col = 'utili_' + args.target
    utili_columns = [elt.replace('utili_', '') for elt in related_columns if 'utili_' in elt]
    query = ['SELECT mean(UsedPercent) FROM ', args.database, '.UtilizationRetention.Utilization where time>=\'', 
        args.startTime, '\' and time<=\'', args.endTime, '\' and ', '(\"Name\"= \'' + '\' or \"Name\"= \''.join(utili_columns) + '\')',
        ' and ClientGUID=\'', args.guid, '\' GROUP BY time(15m), "Name"']
else:
    print("--history_type is neither sensor nor utilization")
    sys.exit()

influxQL = "".join(query)
QueryResult = client.query(influxQL)

if args.history_type == "utilization":
    try:
        sensorByName = [i for i in QueryResult.keys()]
        if sensorByName == []:
            print("No query result")
            sys.exit(0)
        utili_df = pd.DataFrame()
        for i in range(len(sensorByName)):
            utili_df = pd.concat([utili_df, QueryResult[sensorByName[i]]], axis=1)
            utili_df = utili_df.rename(columns={'mean': 'utili_' + sensorByName[i][1][0][1]})
    except Exception as err:
        print(err)
        sys.exit(0)

if args.history_type == "sensor":
    try:
        sensorByName = [i for i in QueryResult.keys()]
        utili_df = pd.DataFrame()
        for i in range(len(sensorByName)):
            utili_df = pd.concat([utili_df, QueryResult[sensorByName[i]]], axis=1)
            utili_df = utili_df.rename(columns={'mean': sensorByName[i][1][0][1]})
    except Exception as err:
        print("No query result")
        print(err)
        sys.exit(0)


df = utili_df

df = missing_data_preprocess(df, target_col)
inference_data_lack(df, past_history, future_target)

target_column_index = df.columns.tolist().index(target_col)

dataset = df.values
# standardize the dataset using the mean and standard deviation of the training data
data_mean = info["_source"][args.target]["mean"]
data_std = info["_source"][args.target]["std"]
dataset = (dataset-data_mean) / data_std

x, y = multivariate_data(dataset, dataset[:, target_column_index], 0, None, past_history, future_target, STEP)

def error_rate_5percent(y_true, y_pred, tol=0.05):
    """evaluate accuracy between ytruth and ypredict"""
    # check tensor dtype
    diff = y_true - y_pred
    lst = tf.math.less(tf.abs(diff), tf.abs(y_true*tol))
    right_wrong_list = tf.dtypes.cast(lst, tf.float16)
    try:
        score = tf.reduce_mean(right_wrong_list)
    except Exception as error:
        print(error)
    return score

if args.history_type == "utilization":
    model_name = './model/' + args.guid + "utili_" + args.target + ".h5"
elif args.history_type == "sensor":
    model_name = './model/' + args.guid + "_" + args.target + ".h5"
model = tf.keras.models.load_model(model_name, custom_objects=\
    {"error_rate_5percent": error_rate_5percent})

prediction = model.predict(np.array([x[-1].tolist()]))
predictData = prediction * data_std[target_column_index] + data_mean[target_column_index]
real_y = y[-1] * data_std[target_column_index] + data_mean[target_column_index]

# history json
print("{ \"history\":")
past_7_df = df.iloc[-672:, target_column_index]
if args.output_freq == "15min":
    groupby_count = 1
elif args.output_freq == "30min":
    groupby_count = 2
elif args.output_freq == "1h":
    groupby_count = 4
elif args.output_freq == "6h":
    groupby_count = 24

print(past_7_df[[x for x in range(past_7_df.shape[0]) if x%groupby_count == 0]].to_json(orient='columns', date_format='iso'))

print(", \"prediction\":")
# predict json
predict_array = predict_by_timefreq(predictData[0], args.output_freq)
now = datetime.datetime.today()
start_ts = now + datetime.timedelta(minutes = 15*(int(now.minute/15) + 1) - now.minute, seconds = -now.second, microseconds= -now.microsecond)
idx = pd.date_range(datetime.datetime.today(), periods=len(predict_array), freq=args.output_freq)
predict_df = pd.DataFrame(predict_array, index=idx, columns=['prediction'])
print(predict_df['prediction'].to_json(orient='columns', date_format='iso'))
print("}")

#time_end=time.time()
#print('time cost',time_end-time_start,'s')
