# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 10:33:23 2025

@author: Admin

선형회귀
"""

import tensorflow as tf

### 자동 미분 ㅣ tape,granndient() ###
w = tf.Variable(2.)

def f(w):
    y = w**2
    z = 2*y + 5
    return z

with tf.GradientTape() as tape:
    z = f(w)

gradients = tape.gradient(z, [w])
print(gradients)

#### 자동 미분을 이용한 선형 회귀 구현
# 가중치 변수 w와 편향 변수 b를 선언
W = tf.Variable(4.0)
b = tf.Variable(1.0)

# 가설을 함수로서 정의 : W*x+b (가중치*독립변수 + 편차)
@tf.function
def hypothesis(x):
    return W*x+b

x_test = [3.5, 5, 5.5, 6]

print(hypothesis(x_test).numpy())
# [15. 21. 23. 25.]

# 평균 제곱 오차를 손실 함수로서 정의
@tf.function
def mse_loss(y_pred, y):
    # 두 개의 차이값을 제곱(square())을 해서 평균(reduce_mean())을 리턴
    return tf.reduce_mean(tf.square(y_pred - y))

# 공부하는 시간
x = [1, 2, 3, 4, 5, 6, 7, 8, 9]
# 각 공부하는 시간에 매핑되는 성적 
y = [11, 22, 33, 44, 53, 66, 77, 87, 95]

# 옵티마이저는 경사 하강법을 사용 / 학습률(learning rate)는 0.01
optimizer = tf.optimizers.SGD(0.01)

# 약 300번에 걸쳐서 경사 하강법을 수행 : epoch

for i in range(301):
    with tf.GradientTape() as tape:
        # 현재 파라미터에 기반한 입력 x에 대한 예측값
        y_pred = hypothesis(x)
        
        # 평균 제곱 오차를 계산
        cost = mse_loss(y_pred, y)
        
    # 손실 함수에 대한 파라미터의 미분값 계산
    gradients = tape.gradient(cost, [W, b])
    
    # 파라미터 업데이트
    optimizer.apply_gradients(zip(gradients, [W, b]))
    
    if i % 10 == 0:
        print("epoch:{:3} | w의 값:{:5.4f} | b의 값:{:5.4} | cost:{:5.6f}".format(i, W.numpy(), b.numpy(), cost))

### 케라스로 구현하는 선형 회귀
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras import optimizers
import numpy as np
'''
케라스로 모델을 만드는 기본적인 형식
1. Sequential로 모델을 만들고
2. add를 통해 입력과 출력 벡터의 차원과 같은 필요한 정보들을 추가
'''
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
y = np.array([11, 22, 33, 44, 53, 66, 77 ,87, 95])

model = Sequential()

# 입력 x의 차원은 1, 출력 y의 차원도 1 / 선형 회귀이므로 activation은 'linear'
model.add(Dense(1, input_dim=1, activation='linear'))

# 경사 하강법: SGD / 학습률(learning rate) : 0.01
sgd = optimizers.SGD(learning_rate=0.01)

# compile()
# optimizer=
# loss
# metrics=
model.compile(optimizer=sgd,loss='mse',metrics=['mse'])

# 주어진 x와 y데이터에 대해서 오차를 최소화하는 작업을 300번: fit()
model.fit(x, y, epochs=300)

import matplotlib.pyplot as plt

plt.plot(x, model.predict(x), 'b', x, y, 'k.')
plt.show()

print(model.predict(np.array([9])))
print(model.predict(np.array([9.5])))
print(model.predict(np.array([0.5])))

'''
시그모이드함수(sigmoid function)


score(x)    result(y)
45          불합격
50          불합격
55          불합격
60          합격
65          합격
70          합격

합격을 1, 불합격을 0 이라고 하였을때

x와 y의 관계를 표현 => s자 형태
    실제값(레이블) : 0 또는 1이라는 두 가지 값
    예측값        : 0과 1사이의 값
    
    최종 예측값이 0.5보다 작으면 0으로 예측했다고 판단하고,
    0.5보다 크면 1로 예측했다고 판단
    
    => 출력이 0과 1사이의 값을 가지면서 S자 형태로 그려지는 함수: 시그모이드 함수
    
'''


import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1/(1+np.exp(-x))

x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)

plt.plot(x, y,'g')
plt.plot([0,0],[1.0,0.0],':')
plt.title('sigmoid func')
plt.show()
'''
출력값을 0과 1사이의 값: y
x 가 0일 때 출력값은 0.5의 값
x 가 증가하면 1에 수렴.
x 가 감소하면 0에 수렴.
'''

y1 = sigmoid(0.5*x)
y2 = sigmoid(x)
y3 = sigmoid(2*x)

plt.plot(x, y1, 'r', linestyle='--')
plt.plot(x, y2, 'g')
plt.plot(x, y3, 'b', linestyle='--')
plt.plot([0,0],[1.0,0.0],':')
plt.title('sigmoid func')
plt.show()

'''
그래프의 경사도가 w(가중치)에 따라 변화..: 그래프의 경사도를 결정
=> w(가중치)가 크면 경사가 커진다.
선형 회귀에서 직선을 표현할 때, w(가중치) 기울기


'''
# b(편향 추가)
y1 = sigmoid(x+0.5)
y2 = sigmoid(x+1)
y3 = sigmoid(x+1.5)

plt.plot(x, y1, 'r', linestyle='--')
plt.plot(x, y2, 'g')
plt.plot(x, y3, 'b', linestyle='--')
plt.plot([0,0],[1.0,0.0],':')
plt.title('sigmoid func')
plt.show()

'''
b(편향 추가) 값에 따라 그래프가 이동
'''

'''
비용 함수(Cost function)

로지스틱 회귀 또한 경사 하강법을 사용하여 가중치를 찾아내지만
비용 함수로는 평균 제곱 오차를 사용하지 않는다!
경사 하강법을 사용하였을때
찾고자 하는 최소값이 아닌 잘못된 최소값에 빠질 가능성 매우 높기 때문!
=> 크로스 엔트로피(cross Entropy)

'''























