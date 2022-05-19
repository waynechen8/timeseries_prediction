# -*- coding: utf-8 -*-
from elasticsearch import Elasticsearch
'''
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-host", help="els ip", type=str)
parser.add_argument("-guid", help="clientguid", type=str)
args = parser.parse_args()
'''
# check els connection
def store_train_info(host_ip, target_name, mean, std, features, guid):
    es = Elasticsearch([{"host": host_ip, "port": 9200}], http_auth=("elastic", "!QAZ1qaz"))

    try:
        body = {
            "doc":{
                target_name:{
                    "mean": mean,
                    "std": std,
                    "feature_columns": features
                }
            },
            "doc_as_upsert": True
        }
        response = es.update(index="history_predict", id=guid, doc_type="_doc", body=body)
        return response
    except Exception as err:
        print(err)
