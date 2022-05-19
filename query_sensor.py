from __future__ import absolute_import, division, print_function, unicode_literals
from influxdb import InfluxDBClient, SeriesHelper, DataFrameClient
from argparse import ArgumentParser
import pandas as pd
import sys
import datetime

parser=ArgumentParser()
parser.add_argument("-influx_host",    type=str, nargs="?", help="influxdb ip")
parser.add_argument("-influx_passwd",  type=str, nargs="?")
parser.add_argument("-guid",     type=str, nargs="?", help="guid")
parser.add_argument("-startTime",      type=str, nargs="?", \
    help="query time start, YYYY-MM-DDTHH:mm:ssZ", default="2020-10-01T00:00:00Z")
parser.add_argument("-endTime",        type=str, nargs="?", \
    help="query time end, YYYY-MM-DDTHH:mm:ssZ", default="2020-12-31T00:00:00Z")
#parser.add_argument("-timeFreq",       type=str, nargs="?", help="mini time frquency")
parser.add_argument("-database", type=str, nargs="?")
args=parser.parse_args()

client = DataFrameClient(host=args.influx_host, port=8086, database=args.database, username='root', password=args.influx_passwd)

queryTypeByHost='show tag values on ' + args.database +' with KEY="Type" where EventType=\'Threshold\' and ClientGUID=\'' + args.guid + '\''
typeName = client.query(queryTypeByHost)
typeNameList = [i['value']+ '\'' for i in typeName['SensorReading']]

if len(typeNameList) == 0:
    print("there is no threshold type sensor in " + args.guid)
else:
    influxquery = ['SELECT mean(Value) FROM ', args.database, '.HardwareSensorRetention.SensorReading where time>=\'', args.startTime,
            '\' and time<=\'', args.endTime, '\' and EventType=\'Threshold\' and ', '(Type= \'' + ' or Type= \''.join(typeNameList) + ')', ' and ClientGUID=\'', args.guid, '\' GROUP BY time(15m), "Name"']
    influxQL = ''.join(influxquery)
    QueryResult = client.query(influxQL)

if QueryResult:
    sensorByName = [i for i in QueryResult.keys()]
    SensorValue = pd.DataFrame()
    for i in range(len(sensorByName)):
        SensorValue = pd.concat([SensorValue, QueryResult[sensorByName[i]]], axis=1)
        SensorValue = SensorValue.rename(columns={'mean': sensorByName[i][1][0][1]})
else:
    print("No query result")
    sys.exit(0)

try:
    csv_path = "./csvfile/" + args.guid + "_sensor_15m.csv"
    SensorValue.to_csv(csv_path)
except Exception as err:
    print(err)
