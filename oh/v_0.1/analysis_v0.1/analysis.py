import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# type	fixed acidity	volatile acidity	citric acid	residual sugar	chlorides	free sulfur dioxide	total sulfur dioxide	density	pH	sulphates	alcohol	quality
data_path = "./data/winequalityN.csv"

def read_use_pandas(data_path):
    data_set = pd.read_csv(data_path)
    see_maxtrix_use_pyplot(data_set)

def see_maxtrix_use_pyplot(aa):
    sns.pairplot(aa, kind="scatter", hue = 'quality')
    plt.show()

read_use_pandas(data_path)