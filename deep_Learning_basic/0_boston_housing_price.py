# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 09:05:24 2025

@author: Admin
"""

import tensorflow as tf
import pandas as pd

filepath = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/boston.csv'
boston = pd.read_csv(filepath)
print(boston.columns)
'''
Index(['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax',
       'ptratio', 'b', 'lstat', 'medv'],
      dtype='object')
'''
boston.head()

# 11. 독립변수 종속변수 분리
ind = boston[['crim','zn','indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax',
       'ptratio', 'b', 'lstat']]

dep = boston['medv']
print(ind.shape,dep.shape)

# 2. 모델의 구조
X = tf.keras.layers.Input(shape=[13])
Y = tf.keras.layers.Dense(1)(X)
model = tf.keras.models.Model(X,Y)
model.compile(loss='mse')

# 3.데이터로 모델을 학습
model.fit(ind, dep, epochs=1000, verbose=0)
model.fit(ind, dep, epochs=10)

history = model.fit(ind, dep, epochs=200)

print(history[5:10])

print(dep[5:10])

print(model.summary())

















































