import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = sns.load_dataset('titanic')
df

df.isnull().sum()

df['age'] = df['age'].fillna(df['age'].mean())

df = df.drop('deck',axis = 1)

df['fare'] = df['fare'].fillna(df['fare'].mean())

sns.countplot(x = 'survived',data=df)
plt.title('survivel rate')
plt.show()

sns.countplot(x='survived', hue='sex', data=df)
plt.title("Survival Based on Gender")
plt.show()

sns.countplot(x='pclass',hue= 'survived',data=df)

sns.histplot(df['fare'],bins = 20, kde=1)