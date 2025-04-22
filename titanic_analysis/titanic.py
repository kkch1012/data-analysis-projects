# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 09:08:56 2025

@author: Admin

각 모델에 적용하기 위한 사전 작업

컬럼명을 소문자로 바꾸기

필요한 컬럼만 가져오기

1. sex열을 gender로 바꾸고 여성은 0, 남자는 1로
2. 호칭을 숫자형으로

경사하강법의 기본 원리 - 값을 최소화하는 알파 베타를 여러 번 반복해서 찾는 것
회귀계수 변화에 따른 오차 크기 확인
learnin rate를 곱
SSE가 작아지는 절편 회귀계수 값 찾기

학습률 - 얼마나 큰 폭으로 업데이트 할지
반복 업데이트 
경사하강법 : 배치 경사하강법 / 확률적 경사하강법 / 미니배치 경사하강법 등...

매 업데이트마다 하나의 샘플만 처리 게산이 매우 저렴하고 빠르며 온라인학습 적합
단점 최적값에서 크게 진동 가능


미니배치 경사하강법

데이터 사용: 전체 훈련 데이터셋을 부분 집합으로 나누어, 각 배치에 대해 기울기 계산
파라미터 업데이트: 각 미니배치에 대해 평균 기울기를 계산하여 업데이트
장점: 배치 경사하강법보다 계산 비용이 적고,
    확률적 경사하강법보다 업데이트가 앉어적
    행렬 연산 최적화 효율젹인 계산 가능
    
단점: 하라퍼파라미터 튜닝
    전체 데이터를 여러번 반복
"""

### 회귀계수 구하기 : 평수:전력량

dataset =[[25,100],[52,256],[38,152],[32,140],[25,150],
          [45,183],[40,175],[55,203],[28,152],[42,198]]

train=dataset[:5]
test=dataset[5:]

#회귀계수 값 초기화
coef=[0.0 for i in range(len(train[0]))]

# 회귀계수 값을 이용해서 예측값을 추출하기 위한 함수

def predict(row, coef):
    yhat=coef[0]
    
    for i in range(len(row) -1):
        yhat+=coef[i+1]*row[i]
        
    return yhat
    
def coefficients_sgd(train, l_rate, n_epoch):
    coef=[0.0 for i in range(len(train[0]))]
    
    for epoch in range(n_epoch):
        sse =0
        
        for row in train:
            yhat = predict(row, coef)
            error = yhat - row[-1]
            sse += error**2
            coef[0] = coef[0] - l_rate * error
            
            for i in range(len(row)-1):
                coef[i+1] = coef[i+1]-l_rate*error*row[i]
                
            return coef, sse
        
import math
def coefficients_sgd(train, l_rate, n_epoch):
    coef=[0.0 for i in range(len(train[0]))]
    
    for epoch in range(n_epoch):
        sum_error=0
        
        for row in train:
            yhat = predict(row, coef)
            error = row[-1]-yhat
            sum_error += (error**2)
            coef[0] = coef[0] + l_rate * error
            
            for i in range(len(row)-1):
                coef[i+1] = coef[i+1]+l_rate*error*row[i]
                
        return coef,math.sqrt(sum_error/len(train))
    
l_rate=0.0001
n_epoch=10
coef=coefficients_sgd(train,l_rate,n_epoch)
coef

'''
coefficients_sgd()    
간단한 선형 회귀 모델의 회귀 계수를 찾기 위해
확률적 경사 하강법(SGD)를 구현

데이터 세트에는 두개의 열이 있는데
첫 번째 열은 독립 변수(x)이고
두 번째 열은 종속 변수(y) (전력량)


기능
predict() : 주어진 입력 행과 계수에 대한 출력(yhat)을 예측.
            첫 번째 요소가 coef절편 / 입력 피처의 계수
            
coefficients_sgd() : SGD 알고리즘을 구현
    계수를 0으로 초기화하고,
    지정된 횟수의 에포크 동안 반복하고,
    각 에포크에서 학습 데이터를 반복.
    각 데이터에 대해 예측 오류를 계산하고
    계수와 함께 반환
    
1. SGD 구현 방식을 확인
2. 결과 계수와 RMSE가 무엇을 의미하는지?
3. 모델이나 학습 과정을 개선하는 방법(예: 학습 속도 변경, 에포크 추가) 에 대한 제안
4. 다른 회귀 방법과 결과를 비교

dataset: 학습과 테스트에 사용될 전체 데이터셋 
    
    
'''
    
dataset =[[25,100],[52,256],[38,152],[32,140],[25,150],
          [45,183],[40,175],[55,203],[28,152],[42,198]]

train=dataset[:5]
test=dataset[5:]

actual=[j[1] for j in test]
l_rate=0.0001
n_epoch=10

alpha,beta = (coefficients_sgd(train,l_rate,n_epoch)[0][0],
             coefficients_sgd(train,l_rate,n_epoch)[0][1])
# Out[16]: (0.061421208965554516, 2.3704156095272326)
    
def predicted(train, test, alpha, beta):
    predictions=list()
    x=[i[0] for i in train]
    y=[j[1] for j in train]
    
    for i in test:
        yhat=alpha+beta*i[0]
        predictions.append(yhat)
        
    return predictions
pred = predicted(train, test, alpha, beta)
# [106.73012363769102,
#  94.87804559005485,
#  130.43427973296335,
#  66.43305827572807,
#  99.61887680910932]
    
    
    
from math import sqrt

def RSME(actual, predicted):
    sum_error=0.0
    
    for i in range(len(actual)):
        prediction_error = predicted[i] - actual[i]
        sum_error += (prediction_error**2)
        
    mean_error=sum_error/float(len(actual))
    return sqrt(mean_error)

RSME(actual, pred)
    # Out[20]: 83.06979269627581
def sq_error(xi,yi,theta):
    alpha, beta = theta
    error=yi-(beta*xi+alpha)
    sq_error=error**2
    return sq_error
def sq_error_grad(xi, yi, theta):
    alpha, beta = theta
    return [-1 *(yi-(beta*xi+alpha)),-2*(yi-(beta*xi+alpha))*xi]

def vector_subtract(v,w):
    return [vi-wi for vi, wi in zip(v,w)]
def scalar_multiply(c,v):
    return [c*vi for vi in v]
def sgd1(sq_error, sq_error_grad,x,y,theta_0,l_rate_0):
    data=list(zip(x,y))
    # x: 훈련 데이터 독립 변수 값 목록
    # y: 훈련 데이터의 종속 변수 값 목록
    
    theta=theta_0 # 매개변수(알파 및 베타)에 대한 초기값
    l_rate=l_rate_0 # 초기 학습률
    min_value=float("inf") # 무한대
    iterations=0 
    # 총 제곱 오차가 5회 연속 반복에서 감소하지 않는 한 계속
    while iterations <5:
        # 현재 시점의 모든 학습 데이터에 대한 총 제곱 오차를 계산
        value=sum(sq_error(xi,yi,theta) for xi,yi in data)
        # 현재 총 제곱 오차(value)가 현재 최소값 보다 작으면
        # 업데이트: 지금까지 발견된 min_theta가장 좋은값
        # 0으로 재설정하고 학습률 초기 학습률로 재설정
        # =>"개선 없음" 카운터를 다시 시작하고, 원래 학습률을 다시 사용한다.
        if value < min_value:
            min_theta,min_value=theta,value
            iterations=0
            l_rate=l_rate_0
            # 총 제곱오차가 감소하지 않으면
            # 학습률을 0.9로 곱하여 증가
            # =>최소값에 가까워짐에 따라 매개변수를 미세 조정하는 일반적인 기술
        else:
            iterations+=1
            l_rate*=0.9
            
            
        for xi,yi in data:
            gradient_i=sq_error_grad(xi,yi,theta)
            theta=vector_subtract(theta, scalar_multiply(l_rate,gradient_i))
            
    return min_theta
    
    
    
dataset =[[25,100],[52,256],[38,152],[32,140],[25,150],
          [45,183],[40,175],[55,203],[28,152],[42,198]]

train=dataset[:5]
test=dataset[5:]

x=[i[0] for i in train]
y=[j[0] for j in train]
l_rate_0=0.0001
theta_0=[0,0]
sgd1(sq_error, sq_error_grad,x,y,theta_0,l_rate_0)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    





















































