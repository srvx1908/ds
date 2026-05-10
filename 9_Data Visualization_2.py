import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 

df = sns.load_dataset('titanic')
df

df.isnull().sum()

df['age'] = df['age'].fillna(df['age'].mean())

df = df.drop('deck',axis = 1)

df['fare'] = df['fare'].fillna(df['fare'].mean())

sns.boxplot(
    x = 'sex',
    y = 'age',
    hue = 'survived',
    data = df
)
plt.xlabel('sex')
plt.ylabel('age')

