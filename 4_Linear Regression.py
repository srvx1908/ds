import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

df = pd.read_csv('boston_housing.csv')
df

df.head()

df.describe()

df.isnull().sum()

plt.figure(figsize=(10, 7))
sns.heatmap(df.corr(), annot= True,)
plt.show()

x = df.drop('medv',axis = 1)
y = df['medv']

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size = 0.2,
    random_state = 42
)

model = LinearRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)
y_pred

mse = mean_squared_error(y_test,y_pred)
mse

r2 = r2_score(y_test,y_pred)
r2