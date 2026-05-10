import numpy as np 
import pandas as pd

df = pd.read_csv('iris.csv')

df.head()

df.describe()

df.info()

summary_stats = df.groupby('Species')['sepal_length'].agg(['mean', 'median', 'min', 'max', 'std'])
summary_stats

species_num =  df['Species'].astype('category').cat.codes.tolist()
species_num.head()

species_list = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
for s in species_list:
    print("\n Detailed Statistics for:",s)
    specific_species_df = df[df['Species'] == s]
    print(specific_species_df.describe())