import pandas as pd

datapath = "../data/winequalityN.csv"

data = pd.read_csv(datapath, delimiter= ',')

q_list = {}

for i in data["quality"]:
    q_list[i] = 0

for i in data["quality"]:
    q_list[i] += 1

for e,m in q_list.items():
    print("{} : {}".format(e, m))

type_list = {}

for i in data["type"]:
    type_list[i] = 0

for i in data["type"]:
    type_list[i] += 1

for e,m in type_list.items():
    print("{} : {}".format(e, m))

