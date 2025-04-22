# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 09:05:30 2025

@author: Admin
"""

import tensorflow as tf
import pandas as pd

# 1.과거의 데이터
filePath = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/iris.csv'
iris = pd.read_csv(filePath)
iris.head()
'''
   꽃잎길이  꽃잎폭  꽃받침길이  꽃받침폭      품종
0   5.1  3.5    1.4   0.2  setosa
1   4.9  3.0    1.4   0.2  setosa
2   4.7  3.2    1.3   0.2  setosa
3   4.6  3.1    1.5   0.2  setosa
4   5.0  3.6    1.4   0.2  setosa
'''

# 원핫인코딩
encode = pd.get_dummies(iris)
encode.head()
'''
   꽃잎길이  꽃잎폭  꽃받침길이  꽃받침폭  품종_setosa  품종_versicolor  품종_virginica
0   5.1  3.5    1.4   0.2       True          False         False
1   4.9  3.0    1.4   0.2       True          False         False
2   4.7  3.2    1.3   0.2       True          False         False
3   4.6  3.1    1.5   0.2       True          False         False
4   5.0  3.6    1.4   0.2       True          False         False
'''
print(encode.columns)

# 독립변수, 종속변수
ind = encode[['꽃잎길이','꽃잎폭','꽃받침길이','꽃받침폭']]
dp = encode[['품종_setosa', '품종_versicolor',
       '품종_virginica']]

# 2. 모델의 구조
X = tf.keras.layers.Input(shape=[4])
Y = tf.keras.layers.Dense(3, activation='softmax')(X)
model = tf.keras.models.Model(X,Y)

model.compile(loss='categorical_crossentropy',metrics=['accuracy'])

# 3. 데이터로 모델을 학습
model.fit(ind, dp, epochs=100)

print(model.predict(ind[:5]))
'''
[[0.8933927  0.04328993 0.06331744]
 [0.8462022  0.05275417 0.10104363]
 [0.8618938  0.0556251  0.08248115]
 [0.81841284 0.07277011 0.108817  ]
 [0.89377695 0.04652931 0.05969373]]
'''

print(dp[:5])
'''
   품종_setosa  품종_versicolor  품종_virginica
0       True          False         False
1       True          False         False
2       True          False         False
3       True          False         False
4       True          False         False
'''





































