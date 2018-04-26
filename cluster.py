import numpy as np
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
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


clu_km = KMeans(n_clusters=5, random_state=0).fit(train)
clu_ms = MeanShift().fit(train)

colors = ['b','g','r','y','m','k','c']


def ShowResult(name, cluster, i):
    temp = pd.concat([pd.DataFrame(cluster.labels_, columns=['ClassLabel']), train], axis=1)
    n_class = len(temp['ClassLabel'].unique())

    figure = plt.subplot(2, 1, i)
    plt.rcParams['font.sans-serif'] = ['SimHei']
    figure.set_title(name)

    for i in range(n_class):
        type = temp.loc[temp['ClassLabel'] == i]
        figure.scatter(type['Age'], type['Fare'], c=colors[i])

    plt.xlabel('Age')
    plt.ylabel('Fare')


plt.figure()
ShowResult('K-Means', clu_km, 1)
ShowResult('MeanShift', clu_ms, 2)
plt.show()