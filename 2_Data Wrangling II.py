import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = {
    'Student_ID': range(1, 11),
    'Math_Score': [85, 90, np.nan, 70, 88, 92, 5, 78, 82, 95], 
    'Science_Score': [70, 72, 75, 80, 85, np.nan, 88, 90, 92, 85],
    'History_Score': [95, 80, 10, 85, 90, 92, 88, 75, 200, 82] 
}

df = pd.DataFrame(data)

df.describe()

df.isnull().sum()

df['Math_Score'] = pd.to_numeric(df['Math_Score'], errors = 'coerce')
df['Science_Score'] = pd.to_numeric(df['Science_Score'], errors = 'coerce')

df.fillna(df['Math_Score'].mean(),inplace = True)
df.fillna(df['Science_Score'].mean(),inplace = True)

df

sns.boxplot(df[['Math_Score','Science_Score','History_Score']])

Q1 = df['Math_Score'].quantile(0.25)
Q1

Q3 = df['Math_Score'].quantile(0.75)
Q3

IQR = Q3 - Q1
IQR

upper_bound = Q3 + 1.5 * IQR
upper_bound

lower_bound = Q1 - 1.5 * IQR
lower_bound

outlier = df[(df['Math_Score'] < lower_bound)  | (df['Math_Score'] > upper_bound)]
outlier

new_df = df[(df['Math_Score'] > lower_bound)  & (df['Math_Score'] < upper_bound)]
new_df

Q1 = new_df['History_Score'].quantile(0.25)
Q1

Q3 = new_df['History_Score'].quantile(0.75)
Q3

IQR = Q3 - Q1
IQR

upper_bound = Q3 + 1.5 * IQR
upper_bound

lower_bound = Q1 - 1.5 * IQR
lower_bound

outlier = new_df[(new_df['History_Score'] < lower_bound)  | (new_df['History_Score'] > upper_bound)]
outlier

new_df1 = new_df[(new_df['History_Score'] > lower_bound )& (new_df['History_Score'] < upper_bound)]
new_df1

new_df1['Math_Score_Log']= np.log1p(new_df1['Math_Score'])
new_df1['Math_Score_Log']

plt.figure(figsize=(10, 4))
sns.histplot(new_df1['Math_Score_Log'], kde=True)
plt.title('Log Transformed Math Score')
plt.show()
