import numpy as np
import pandas as pd

df = pd.read_csv('StudentPerformance.csv')

print(df.head())

print(df.describe())

print(df.info())

df.isnull().sum()

df['Hours Studied'] = df['Hours Studied'].astype('float')

df['Extracurricular Activities'] = df['Extracurricular Activities'].map({
    'Yes':1,
    'No':0
})

df['Normalised Hours Studied'] = (
    (df['Hours Studied'] - df['Hours Studied'].min()) /
    (df['Hours Studied'].max() - df['Hours Studied'].min())
)

print(df)