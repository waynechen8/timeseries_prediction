# -*- coding: utf-8 -*-

from influxdb import InfluxDBClient, SeriesHelper, DataFrameClient
from argparse import ArgumentParser
import pandas as pd
import sys

parser=ArgumentParser()
parser.add_argument("-influx_host",    type=str, nargs="?", help="influxdb ip", default="10.10.75.106")
parser.add_argument("-influx_passwd",  type=str, nargs="?")
parser.add_argument("-guid",     type=str, nargs="?", help="device name")
parser.add_argument("-startTime",      type=str, nargs="?", \
    help="query time start, YYYY-MM-DDTHH:mm:ssZ", default="2020-10-01T00:00:00Z")
parser.add_argument("-endTime",        type=str, nargs="?", \
    help="query time end, YYYY-MM-DDTHH:mm:ssZ", default="2020-12-31T00:00:00Z")
#parser.add_argument("-timeFreq",       type=str, nargs="?", help="mini time frquency")
args=parser.parse_args()

client = DataFrameClient(host=args.influx_host, port=8086, database='aswment', username='root', password=args.influx_passwd)

queryTypeByHost='show tag values on aswment with KEY="Type" where ClientGUID=\'' + args.guid + '\''
TypeName = client.query(queryTypeByHost)
TypeNameList = [i['value']+ '\'' for i in TypeName['Utilization']]

if len(TypeNameList) == 0:
    print("there is no threshold type sensor in " + args.guid)
else:
    influxquery = ['SELECT mean(UsedPercent) FROM aswment.UtilizationRetention.Utilization where time>=\'', args.startTime,
            '\' and time<=\'', args.endTime, '\' and ', '(Type= \'' + ' or Type= \''.join(TypeNameList) + ')', ' and ClientGUID=\'', 
            args.guid, '\' GROUP BY time(15m), "Name"']
    influxQL = ''.join(influxquery)
    QueryResult = client.query(influxQL)

if QueryResult:
    sensorByName = [i for i in QueryResult.keys()]
    utili_df = pd.DataFrame()
    for i in range(len(sensorByName)):
        utili_df = pd.concat([utili_df, QueryResult[sensorByName[i]]], axis=1)
        utili_df = utili_df.rename(columns={'mean': 'utili_' + sensorByName[i][1][0][1]})
else:
    print("No query result")
    sys.exit(0)
try:
    #utili_df.dropna(inplace=True)
    #utili_df.fillna(0, inplace=True)
    csv_path = "./csvfile/" + args.guid + "_uti_15m.csv"
    utili_df.to_csv(csv_path)
except Exception as err:
    print(err)
