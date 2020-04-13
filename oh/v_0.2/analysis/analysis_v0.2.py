import pandas as pd
import matplotlib.pyplot as plt

data_path = "./data/white.csv"

def read_use_pandas():
    data_set = pd.read_csv(data_path)
    data_set.dropna(inplace = True)
    see_maxtrix_use_pyplot(data_set)

def see_maxtrix_use_pyplot(aa):
    aa.drop(['index'], axis = 1)
    y_data = aa['y_or_n']
    x_data = aa.drop(['y_or_n'], axis = 1)
    pd.plotting.scatter_matrix(x_data, c = y_data, figsize=(50,50), marker ='o', hist_kwds = {'bins' : 20}, s =30, alpha = 20)
    plt.show()
    
read_use_pandas()