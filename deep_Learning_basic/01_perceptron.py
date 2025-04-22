# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 09:05:46 2025

@author: Admin
"""


def AND_gate(x1, x2):
    w1 = 0.5
    w2 = 0.5
    b = -0.7
    result = x1*w1 + x2*w2 + b
    if result <= 0:
        return 0
    else:
        return 1


AND_gate(0, 0),AND_gate(0, 1),AND_gate(1, 0),AND_gate(1,1)
# Out[89]: (0, 0, 0, 1)


# NAND 게이트

def NAND_gate(x1, x2):
    w1 = -0.5
    w2 = -0.5
    b = 0.7
    result = x1*w1 + x2*w2 + b
    if result <= 0:
        return 0
    else:
        return 1


NAND_gate(0, 0),NAND_gate(0, 1),NAND_gate(1, 0),NAND_gate(1,1)
# Out[91]: (1, 1, 1, 0)


def OR_gate(x1, x2):
    w1 = 0.6
    w2 = 0.6
    b = -0.5
    result = x1*w1 + x2*w2 + b
    if result <= 0:
        return 0
    else:
        return 1


OR_gate(0, 0),OR_gate(0, 1),OR_gate(1, 0),OR_gate(1,1)
# Out[93]: (0, 1, 1, 1)

# 활성화 함수
import numpy as np
import matplotlib.pyplot as plt

# Step Function
def step(x):
    return np.array(x > 0, dtype=np.int64)

x = np.arange(-5.0, 5.0, 0.1) # -5.0부터 5.0까지 0.1 간격 생성

y = step(x)

plt.title('Step Function')
plt.plot(x,y)
plt.show()


def sigmoid(x):
    return 1/(1+np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)

y = sigmoid(x)

plt.plot(x, y)
plt.plot([0,0],[1.0,0.0], ':') # 가운데 점선 추가
plt.title('Sigmoid Function')
plt.show()

x = np.arange(-5.0, 5.0, 0.1) # -5.0부터 5.0까지 0.1 간격 생성
y = np.tanh(x)

plt.plot(x, y)
plt.plot([0,0],[1.0,-1.0], ':')
plt.axhline(y=0, color='orange', linestyle='--')
plt.title('Tanh Function')
plt.show()

def relu(x):
    return np.maximum(0, x)

x = np.arange(-5.0, 5.0, 0.1)
y = relu(x)

plt.plot(x, y)
plt.plot([0,0],[5.0,0.0], ':')
plt.title('Relu Function')
plt.show()

a = 0.1
def leaky_relu(x):
    return np.maximum(a*x, x)
        
x = np.arange(-5.0, 5.0, 0.1)
y = leaky_relu(x)

plt.plot(x, y)
plt.plot([0,0],[5.0,0.0], ':')
plt.title('Leaky ReLU Function')
plt.show()

x = np.arange(-5.0, 5.0, 0.1) # -5.0부터 5.0까지 0.1 간격 생성
y = np.exp(x) / np.sum(np.exp(x))

plt.plot(x, y)
plt.title('Softmax Function')
plt.show()


### 행렬곱 신경망 ###
## 1. 순전파(Foward propagation)



from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(2, input_dim=3,activation='softmax'))
model.summary()

'''
Model: "sequential"
┌─────────────────────────────────┬────────────────────────┬───────────────┐
│ Layer (type)                    │ Output Shape           │       Param # │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_5 (Dense)                 │ (None, 2)              │             8 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 8 (32.00 B)
 Trainable params: 8 (32.00 B)
 Non-trainable params: 0 (0.00 B)
 '''

model = Sequential()
# 4개의 입력과 8개의 출력
model.add(Dense(8, input_dim=4,activation='relu'))
# 이어서 8개의 출력
model.add(Dense(8, activation='relu'))
# 이어서 3개의 출력
model.add(Dense(3, activation='softmax'))

model.summary()
'''
Model: "sequential_1"
┌─────────────────────────────────┬────────────────────────┬───────────────┐
│ Layer (type)                    │ Output Shape           │       Param # │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_6 (Dense)                 │ (None, 8)              │            40 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_7 (Dense)                 │ (None, 8)              │            72 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_8 (Dense)                 │ (None, 3)              │            27 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 139 (556.00 B)
 Trainable params: 139 (556.00 B)
 Non-trainable params: 0 (0.00 B)
 '''

model.compile(optimizer='adam',loss='mse',metrics=['mse'])

## 1. 손실 함수(Loss function)

model.compile(optimizer='adam',loss='mse',metrics=['mse'])

model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['acc'])

model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['acc'])

model.compile(loss='sparse_categorical_crossentropy',optimizer='adam',metrics=['acc'])



adam = tf.keras.optimizers.Adam(learning_rate=0.001,beta_1=0.9,beta_2=0.999,epsilon=None,decay=0.0,amsgrad=False)

model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['acc'])















