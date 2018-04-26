import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import matplotlib.pyplot as plt


train = pd.read_csv(open("./data/Titanic/train.csv"))
test = pd.read_csv(open("./data/Titanic/test.csv"))

drop_attr = ['PassengerId', 'Name', 'Cabin', 'Ticket']
str_attr = ['Sex', 'Embarked']

train = train.drop(drop_attr, axis=1)
test = test.drop(drop_attr, axis=1)
train['Embarked'] = train['Embarked'].fillna('S')
train['Age'] = train['Age'].fillna(train['Age'].mean())
test['Age'] = test['Age'].fillna(train['Age'].mean())
test['Fare'] = test['Fare'].fillna(test['Fare'].median())

le = LabelEncoder()
for attr in str_attr:
    train[attr] = le.fit_transform(train[attr])
    test[attr] = le.fit_transform(test[attr])

X_attr = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
Y_attr = ['Survived']
train_X = train[X_attr]
train_Y = train[Y_attr]
test_X = test[X_attr]

res_nb = GaussianNB().fit(train_X, train_Y).predict(test_X)
res_dt = DecisionTreeClassifier(max_depth=5).fit(train_X, train_Y).predict(test_X)


def ShowResult(name, test_Y, i):
    temp = pd.concat([pd.DataFrame(test_Y, columns=['Survived']), test[['Age', 'Fare']]], axis=1)
    alive = temp.loc[temp['Survived'] == 1]
    dead = temp.loc[temp['Survived'] == 0]
    figure1 = plt.subplot(2, 1, i)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    figure1.set_title(name)
    alive_distribute = figure1.scatter(alive['Age'], alive['Fare'], c='green', marker='s')
    dead_distribute = figure1.scatter(dead['Age'], dead['Fare'], c='red', marker='^')
    plt.xlabel('Age')
    plt.ylabel('Fare')
    figure1.legend((alive_distribute, dead_distribute), ('survived', 'dead'), loc=1)


plt.figure()
ShowResult('NaiveBayes', res_nb, 1)
ShowResult('DecisionTree', res_dt, 2)
plt.show()