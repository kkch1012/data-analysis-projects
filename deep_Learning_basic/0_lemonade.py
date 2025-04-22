# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 09:05:38 2025

@author: Admin
"""

import tensorflow as tf
import pandas as pd
import numpy as np
### 데이터를 준비
'''
파일경로 = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/lemonade.csv'

레모네이드 = pd_read_csv(파일경로)

레모네이드.head()
'''

filePath = 'https://raw.githubusercontent.com/blackdew/tensorflow1/master/csv/lemonade.csv'
lemonade = pd.read_csv(filePath)
lemonade.head()

### 종속변수, 독립변수
'''
독립 = 레모네이드[['온도]]
종속 = 레모네이드[['판매량']]
print(독립.shape, 종속.shape)
            '''
independence = lemonade[['온도']]
dependence = lemonade[['판매량']]
print(independence.shape, dependence.shape)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

### 모델

X = tf.keras.layers.Input(shape=[1]) # 입력 레이어
Y = tf.keras.layers.Dense(1)(X)      # 출력 레이어 (완전 연결 층)

model = tf.keras.models.Model(X, Y) # 모델 생성

model.compile(loss='mse')
# 모델 학습
'''
model.fit(독립, 종속, epochs=100, verbose =0) verbose: 학습 진행률 보여줄건지 아닌지
model.fit(독립, 종속, epochs=10)
'''
model.fit(independence,dependence,epochs=1000,verbose=1)
model.fit(independence,dependence,epochs=200,verbose=1)

### 모델 검증
'''
print(model.predict(독립))
print(model.predict([[15]]))
'''
print(model.predict(np.array([[15]])))


## 가중치 확인: get_weights()
print(model.get_weights())
# [array([[1.8639823]], dtype=float32), array([1.214488], dtype=float32)]

print(model.summary())