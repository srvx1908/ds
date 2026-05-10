import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

df = pd.read_csv('Social_Network_Ads.csv')

df.head()

df.describe()

df.info()

x = df[['Age','EstimatedSalary']]
y = df['Purchased']
print('x',x)
print('y',y)

x_train,x_test,y_train,y_test = train_test_split(
    
        x,
        y,
        test_size=0.25,
        random_state=0 
)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

model = LogisticRegression()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
y_pred

cm = confusion_matrix(y_test,y_pred)
cm

TN = cm[0][0]
TP = cm[1][1]
FN = cm[1][0]
FP = cm[0][1]
print('true positive =',TP)
print('true negative =',TN)
print('false positive =',FP)
print('fales negative =',FN)

accuracy = accuracy_score(y_test,y_pred)
accuracy

error_rate = 1 - accuracy
error_rate

precision = TP / (TP + FP)
precision

recall = TP / (TP + FN)
recall