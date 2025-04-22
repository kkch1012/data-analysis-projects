# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 14:05:11 2025

@author: Admin
"""

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers

import tensorflow as tf
from tensorflow import keras

'''

독립 변수 데이터: x -50 ~ 50
종속 변수 데이터: y 숫자 10 이상인 경우에는 1 / 미만인 경우에는 0

활성화 함수 : 시그모이드
옵티마이저 : sgd
손실 함수 : 크로스 엔트로피(binary_crossentropy)

'''
x = np.array([-50, -40, -30, -20, -10, -5, 0, 5, 10, 20, 30, 40, 50])
y = np.array([  0,   0,   0,    0,  0,  0,  0,  0,  1,  1,  1,  1,  1])

model = Sequential()
model.add(Dense(1, input_dim=1,activation='sigmoid'))

sgd = optimizers.SGD(learning_rate=0.01)

model.compile(optimizer=sgd, loss='binary_crossentropy',
              metrics=['binary_accuracy'])

model.fit(x,y, epochs=200)

plt.plot(x, model.predict(x), 'b', x, y, 'k.')
plt.show()

print(model.predict(np.array([1, 2, 3, 4, 4.5])))
'''
[[0.51284856]
 [0.58049524]
 [0.64524615]
 [0.7050804 ]
 [0.73268914]]
'''
print(model.predict(np.array([11, 21, 31, 41, 500])))
'''
[[0.94188267]
 [0.99600786]
 [0.9997397 ]
 [0.9999831 ]
 [1.        ]]
'''









































