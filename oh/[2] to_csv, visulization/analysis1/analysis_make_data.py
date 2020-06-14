import pandas as pd
import numpy as np

datapath = "./data/winequalityN.csv"

pd_data = pd.read_csv(datapath, delimiter= ',')

q_list = dict()

for i in pd_data["quality"]:
    if(i not in q_list):
        q_list[i] = 1
    else :
        q_list[i] += 1
        
for e,m in q_list.items():
    print("{} 등급 : {}".format(e, m))

type_list = {}

for i in pd_data["type"]:
    type_list[i] = 0

for i in pd_data["type"]:
    type_list[i] += 1

for e,m in type_list.items():
    print("{} : {}".format(e, m))

q_list = dict()
pd_white = pd_data[pd_data['type'] == 'white']

for i in pd_white["quality"]:
    if(i not in q_list):
        q_list[i] = 1
    else :
        q_list[i] += 1
        
for e,m in q_list.items():
    print("white - {} 등급 : {}".format(e, m))

pd_data = pd_data[pd_data['quality'] != 9.0]

pd_data = pd_data[pd_data['quality'] != 3.0]

pd_data_white = pd_data[pd_data['type'] == 'white']

pd_data_white.to_csv("./data/v0.1.1.csv", mode='w')