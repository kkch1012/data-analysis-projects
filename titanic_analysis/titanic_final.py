# -*- coding: utf-8 -*-
"""
Created on Fri Mar 28 14:35:39 2025

@author: Admin
"""

import pandas as pd

uri = "./data/titanic_1309.xlsx"
df = pd.read_excel(uri, sheet_name='total')

train_1000 = df.iloc[:1000]
test_309 = df.iloc[1000:]

train_1000.drop(['boat','body','home.dest'],axis=1,inplace=True)
test_309.drop(['boat','body','home.dest'],axis=1,inplace=True)
train_df=train_1000
test_df=test_309

total=[train_df,test_df]
titles={'Mr':1,'Miss':2,'Mrs':3,'Master':4,'Special':5}

for dataset in total:
    dataset['title']=dataset.name.str.extract('([A-Za-z]+)\.',expand=False)
    dataset['title']=dataset['title'].replace(['Lady','Countess','Capt','Col','Don','Dr','Major',
                                               'Rev','Sir','Jonkheer','Dona'],'Special')
    dataset['title']=dataset['title'].replace('Mlle','Miss')
    dataset['title']=dataset['title'].replace('Ms','Miss')
    dataset['title']=dataset['title'].replace('Mme','Mrs')
    dataset['title']=dataset['title'].map(titles)
    dataset['title']=dataset['title'].fillna(0)
    
train_df=train_df.drop(['name'],axis=1)
test_df=test_df.drop(['name'],axis=1)
####
train_df['age'].fillna(train_df.groupby('title')['age'].transform('median'),inplace=True)
test_df['age'].fillna(test_df.groupby('title')['age'].transform('median'),inplace=True)

data=[train_df,test_df]
for dataset1 in data:
    dataset1['age']=dataset1['age'].astype(int)
    dataset1.loc[dataset1['age']<=11,'age']=0
    dataset1.loc[(dataset1['age']>11)&(dataset1['age']<=18),'age']=1
    dataset1.loc[(dataset1['age']>18)&(dataset1['age']<=22),'age']=2
    dataset1.loc[(dataset1['age']>22)&(dataset1['age']<=27),'age']=3
    dataset1.loc[(dataset1['age']>27)&(dataset1['age']<=33),'age']=4
    dataset1.loc[(dataset1['age']>33)&(dataset1['age']<=40),'age']=5
    dataset1.loc[(dataset1['age']>40)&(dataset1['age']<=66),'age']=6
    dataset1.loc[dataset1['age']>66,'age']=6

data=[train_df,test_df]

sex_mapping={'male':0,'female':1}
for dataset in data:
    dataset['sex']=dataset['sex'].map(sex_mapping)
for dataset in data:
    dataset['embarked']=dataset['embarked'].fillna('S')
    
embarked_mapping={'S':0,'C':1,'Q':2}
for dataset in data:
    dataset['embarked']=dataset['embarked'].map(embarked_mapping)
    
    
for dataset in data:
    dataset['sibpar']=dataset['sibsp']+dataset['parch']
    dataset.loc[dataset['sibpar']>0, 'n_alone']=0
    dataset.loc[dataset['sibpar']==0,'n_alone']=1
    dataset['n_alone']=dataset['n_alone'].astype(int)
    
train_df["fare"].fillna(train_df.groupby('pclass')['fare'].transform('median'),inplace=True)
test_df["fare"].fillna(test_df.groupby('pclass')['fare'].transform('median'),inplace=True)

for dataset in data:
    dataset.loc[dataset['fare']<=20,'fare']=1
    dataset.loc[(dataset['fare']>20)&(dataset['fare']<=30),'fare']=2
    dataset.loc[(dataset['fare']>30)&(dataset['fare']<=50),'fare']=3
    dataset.loc[(dataset['fare']>50)&(dataset['fare']<=100),'fare']=4
    dataset.loc[dataset['fare']>100,'fare']=5
    
    
for dataset1 in data:
    dataset1['fare_person']=dataset1['fare']/(dataset1['sibpar']+1)
    dataset1['fare_person']=dataset1['fare_person'].astype(int)
    
    
X_columns=['pclass','sex','age','embarked','title','sibpar','n_alone','fare_person']
y_column=['survived']
X_train=train_df[X_columns]
y_train=train_df[y_column]
X_test=test_df[X_columns]
y_test=test_df[y_column]


import numpy as np
import pandas as pd

from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB

import seaborn as sns
%matplotlib inline
from matplotlib import pyplot as plt
from matplotlib import style

decision_tree=DecisionTreeClassifier()
decision_tree.fit(X_train,y_train)
Y_pred=decision_tree.predict(X_test)
train_acc_decision_tree=round(decision_tree.score(X_train,y_train)*100,2)
test_acc_decision_tree=round(decision_tree.score(X_test,y_test)*100,2)
train_acc_decision_tree,test_acc_decision_tree

random_forest=RandomForestClassifier(n_estimators=100)
random_forest.fit(X_train,y_train)
Y_prediction=random_forest.predict(X_test)
random_forest.score(X_train,y_train)
train_acc_random_forest=round(random_forest.score(X_train,y_train)*100,2)
test_acc_random_forest=round(random_forest.score(X_test,y_test)*100,2)
train_acc_random_forest,test_acc_random_forest    
    

#p. 368 
#로지스틱 회귀 분석
log_reg = LogisticRegression() 
log_reg.fit(X_train, y_train) 
Y_pred = log_reg.predict(X_test) 
train_acc_log=round(log_reg.score(X_train, y_train)*100, 2) 
test_acc_log=round(log_reg.score(X_test, y_test)*100, 2) 
train_acc_log, test_acc_log

#나이브 베이지안
gaussian=GaussianNB()
gaussian.fit(X_train, y_train)
Y_pred = gaussian.predict(X_test)
train_acc_gaussian=round(gaussian.score(X_train, y_train)*100, 2)
test_acc_gaussian=round(gaussian.score(X_test, y_test)*100, 2)
train_acc_gaussian, test_acc_gaussian


#K-Nearest Neighbor
knn=KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
Y_pred=knn.predict(X_test)
train_acc_knn=round(knn.score(X_train, y_train)*100, 2)
test_acc_knn=round(knn.score(X_test, y_test)*100,2)
train_acc_knn, test_acc_knn

#SVM 모델
svc = LinearSVC()
svc.fit(X_train, y_train)
Y_pred=svc.predict(X_test)
train_acc_svc=round(svc.score(X_train, y_train)*100, 2)
test_acc_svc=round(svc.score(X_test, y_test)*100, 2)
train_acc_svc, test_acc_svc

#퍼셉트론 모델
perceptron = Perceptron(max_iter=5)
perceptron.fit(X_train, y_train)
Y_pred=perceptron.predict(X_test)
train_acc_perceptron=round(perceptron.score(X_train, y_train)*100,2)
test_acc_perceptron=round(perceptron.score(X_test, y_test)*100,2)
train_acc_perceptron, test_acc_perceptron

#확률적 경사하강법 방법
sgd = linear_model.SGDClassifier(max_iter=5, tol=None)
sgd.fit(X_train, y_train)
Y_pred=sgd.predict(X_test)
sgd.score(X_train, y_train)
train_acc_sgd=round(sgd.score(X_train, y_train)*100,2)
test_acc_sgd=round(sgd.score(X_test, y_test)*100, 2)
train_acc_sgd, test_acc_sgd 

#p.370
results = pd.DataFrame({
    'Model':['Support Vector Machines', 'KNN', 'Logistic Regression', 'Random Forest',
             'Naive Bayes', 'Perceptron', 'Stochastic Gradient Decent', 'Decision Tree'],
    'train_Score': [train_acc_svc, train_acc_knn, train_acc_log, train_acc_random_forest,
                    train_acc_gaussian, train_acc_perceptron, train_acc_sgd, train_acc_decision_tree],
    'test_Score':[test_acc_svc, test_acc_knn, test_acc_log, test_acc_random_forest, test_acc_gaussian,
                  test_acc_perceptron, test_acc_sgd, test_acc_decision_tree]})
result_df = results.sort_values(by='train_Score', ascending=False)
result_df=result_df.set_index('Model')
result_df.head(10) 

#p.371
from sklearn.model_selection import cross_val_score
rf=RandomForestClassifier(n_estimators=100)
scores=cross_val_score(rf, X_train, y_train, cv=10, scoring="accuracy")
print("Scores:", scores)
print("Mean:", scores.mean())
print("Standard Deviation:", scores.std())
    
    
                                    
    
    
    
    











































