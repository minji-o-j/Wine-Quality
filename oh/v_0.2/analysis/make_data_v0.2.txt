import pandas as pd
import numpy as np

datapath = "../data/winequalityN.csv"

pd_data = pd.read_csv(datapath, delimiter= ',')

pd_data = pd_data[pd_data['quality'] != 9.0]

pd_data_white = pd_data[pd_data['type'] == 'white']

pd_data_white['y_or_n'] = pd_data_white['quality'] > 5

pd_data_white.to_csv("../data/filename.csv", mode='w')