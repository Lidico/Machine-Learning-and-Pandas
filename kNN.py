from operator import index

import pandas as pd
import numpy as np
from pip._vendor.urllib3.connectionpool import xrange
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report

#method that helps to find the optimal neighbors
def getErrorRate(xtrain, ytrain, xtest, ytest):
    errorRate = []
    for i in xrange(1,40):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(xtrain, ytrain)
        predI = knn.predict(xtest)
        errorRate.append(np.mean(predI != ytest))
    return errorRate

df = pd.read_csv('Classified_Data.csv', index_col=0)
scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS',axis=1))
scaledFeaturs = scaler.transform(df.drop('TARGET CLASS',axis=1))
dfFeat = pd.DataFrame(scaledFeaturs, columns= df.columns[:-1])

X = dfFeat
y = df['TARGET CLASS']
#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

errorRate = getErrorRate(X_train, y_train, X_test, y_test)
numOfOptimalNeig = errorRate.index((min(errorRate)))
knn = KNeighborsClassifier(n_neighbors=numOfOptimalNeig)
knn.fit(X_train, y_train)
predictions = knn.predict(X_test)

#output data
print(confusion_matrix(y_test, predictions),'\n')
print(classification_report(y_test, predictions))
