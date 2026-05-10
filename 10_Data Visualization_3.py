import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

df = pd.read_csv('iris.csv')

df.hist( figsize=(10,8))


plt.figure(figsize=(10,8))
sns.boxplot(data=df.iloc[:,1:5])

