import matplotlib.pyplot as plt
import pandas as pd

data_path = "./data/"
data_name = "v0.1.1.csv"
data_set = pd.read_csv(data_path + data_name)
data_set.dropna(inplace = True)

def read_use_pandas():
    data_schema = data_set.columns.tolist()[1:-1]
    
    for c in data_schema:
        see_use_pyplot(c)
    
def see_use_pyplot(c):
    y_data = data_set["quality"]
    x_data = data_set[c]
    
    plt.plot(x_data, y_data, "bo")
    plt.xlabel(c)
    plt.ylabel("quality")
    plt.show()
    
read_use_pandas()