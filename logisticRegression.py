import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report


#fill all missing age Data.
def imputAge(cols):
    age = cols[0]
    pClass = cols[1]

    if pd.isnull(age):
        if pClass == 1:
            return 37
        elif pClass == 2:
            return 29
        else:
            return 24
    else:
        return age


train = pd.read_csv('titanic_train.csv')
train['Age'] = train[['Age', 'Pclass']].apply(imputAge, axis=1)
#clean all Null data
train.drop('Cabin', axis=1, inplace=True)
train.dropna(inplace=True)
#convert Sex and Embarked to dummies
sex = pd.get_dummies(train['Sex'], drop_first=True)
embark = pd.get_dummies(train['Embarked'], drop_first=True)
train = pd.concat([train, sex, embark], axis=1)
#drop unnecessary data
train.drop(['PassengerId', 'Sex', 'Embarked', 'Name', 'Ticket'], axis = 1, inplace=True)

X = train.drop('Survived', axis=1)
y = train['Survived']
#train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
logModel = LogisticRegression()
#fit the model
logModel.fit(X_train, y_train)
predictions = logModel.predict(X_test)

#output data
print(confusion_matrix(y_test, predictions),'\n')
print(classification_report(y_test, predictions))