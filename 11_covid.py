import pandas as pd
import numpy as np

df = pd.read_csv('covid_vaccine_statewise.csv')
df

df.describe()

df.groupby('State')['First Dose Administered'].max()

df.groupby('State')['Second Dose Administered'].max()

total_males = df['Male(Individuals Vaccinated)'].max()
total_males

total_females = df['Female(Individuals Vaccinated)'].max()
total_females