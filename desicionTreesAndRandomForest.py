import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv('kyphosis.csv')

X = df.drop('Kyphosis', axis=1)
y = df['Kyphosis']

#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

#Desicion Tree
dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
predictions = dtree.predict(X_test)

#Random Forest
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
rfcPredict = rfc.predict(X_test)

#output data
print(confusion_matrix(y_test, predictions),'\n')
print(classification_report(y_test, predictions),'\n')
print(confusion_matrix(y_test, rfcPredict),'\n')
print(classification_report(y_test, rfcPredict))

