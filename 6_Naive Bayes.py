import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

df = pd.read_csv('iris.csv')
df

x = df[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
y = df['Species']

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size = 0.25,
    random_state = 42
)

model = GaussianNB()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)
print(y_pred)

cm = confusion_matrix(y_test,y_pred)
cm

TP = cm[0][0]
FP = cm[1][0] + cm[2][0]
FN = cm[0][1] + cm[0][2]
TN = cm.sum() - (TP + FP + FN)
print(TP) 
print(FN)
print(FP)
print(TN)

accuracy = accuracy_score(y_test,y_pred)
accuracy

error_rate = 1 - accuracy
error_rate

presison = TP / (TP + FP)
presison

recall = TP / (TP + FN)
recall