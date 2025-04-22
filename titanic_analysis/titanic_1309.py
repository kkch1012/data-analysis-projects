# -*- coding: utf-8 -*-
"""
Created on Thu Mar 27 09:08:49 2025

@author: Admin
"""

### 기본 데이터
train = [[25,100],[52,256],[38,152],[32,140],[25,150]]

x =[i[0] for i in train]
y =[j[1] for j in train]

# 기초 통계 함수 구현

def mean(x):
    return sum(x) / len(x)
mean(x),mean(y)

def d_mean(x):
    x_mean = mean(x)
    return [i - x_mean for i in x]

d_mean(x),d_mean(y)

x1 = d_mean(x)
round(mean(x1))

def dot(x, y):
    return sum([x*y for x, y in zip(x,y)])

dot(x,y)


## 분산 : variance(x)

'''
1. x의 개별값에서 x의 평균 치를 구하고, : d_mean()
2. 그 값을 제곱한 후, sum_of_squares()
3. 합계를 구한다. : sum_of_squares()
4. 구한 값의 제곱의 합을 전체 갯수인 n-1로 나눈다!
    표존분산의 경우 n-1로
    모분산의 경우 n으로 나눈다!
'''
def sum_of_squares(v):
    return dot(v,v)
    

def variance(x):
    n = len(x)
    d = d_mean(x)
    return sum_of_squares(d) / (n-1)

variance(x)

def standard_deviation(x):
    return variance(x)**0.5

standard_deviation(x)

def covariance(x,y):
    n=len(x)
    return dot(d_mean(x),d_mean(y))/(n-1)
covariance(x,y)

def correlation(x,y):
    stdev_x=standard_deviation(x)
    stdev_y=standard_deviation(y)
    if stdev_x > 0 and stdev_y >0:
        return covariance(x, y) / (stdev_x * stdev_y)
    else:
        return 0
correlation(x, y)

import numpy as np
x1 = np.array(x)
x1.mean(),x1.var(),x1.std()

np.cov(x1,y),np.corrcoef(x1,y)

np.cov(x1,y)[0][1],np.corrcoef(x1,y)[0][1]

'''
OLS: Ordinary Least Squares

RMSE 평균 제곱근 오차

'''

def OLS(x,y):
    beta=covariance(x, y)/variance(x)
    alpha=mean(y) -beta*mean(x)
    return [alpha, beta]

OLS(x,y)

def OLS_fit(x,y):
    beta=(correlation(x, y)*standard_deviation(y)/standard_deviation(x))
    
    alpha=mean(y)-beta*mean(x)
    return [alpha, beta]

OLS_fit(x,y)

'''
alpha와 beta는 단순 선형 회귀 모델인 회귀 계수
alpha y 절편
beta 기울기

1. 예측값 선언
2. 변수 x에 train 데이터
    변수 y에 train 데이터
    
'''

def predict(alpha, beta,train,test):
    predictions=list()
    x=[i[0] for i in train]
    y=[j[1] for j in train]
    alpha, beta = OLS_fit(x,y)
    for i in test:
        yhat=alpha+beta*i[0]
        predictions.append(yhat)
    return predictions

train = [[25,100],[52,256],[38,152],[32,140],[25,150]]
alpha,beta = OLS_fit(x,y)

pr = predict(alpha, beta,train,train)
print(pr)

import matplotlib.pyplot as plt

plt.rc('font', family='NanumGothic')
plt.scatter(x,y, c='red')
plt.plot(x,pr)
plt.xlabel('평형')
plt.ylabel('전기사용량')
plt.show()




def SSE(alpha,beta,train,test):
    sse =0
    for i in test:
        error = ((i[1])-(alpha+beta*i[0]))**2
        sse=error+sse
    return sse

SSE(alpha,beta,train,train)




def SST(alpha,beta,train,test):
    sst =0
    x=[i[0] for i in train]
    y=[j[1] for j in train]
    for i in test:
        sum_ds = ((i[1])-mean(y))**2
        sst=sum_ds+sst
    return sst

SST(alpha,beta,train,train)

def R_squared(alpha,beta,train,test):
    return 1.0-(SSE(alpha,beta,train,train)/SST(alpha,beta,train,train))

R_squared(alpha,beta,train,train)


train = [[25,100],[52,256],[38,152],[32,140],[25,150]]

x =[i[0] for i in train]
y =[j[1] for j in train]

import statsmodels.api as sms

_X = sms.add_constant(x)
model=sms.OLS(y, _X).fit()
print(model.summary())

'''
sms.add_constant(x) : 독립 변수에 상수 항(절편)을 추가
OLS 회귀를수행

검정 통계랑
확률 통계량

다중 공선성: 높은 상관관계를 갖는 현상
완벽한 다중공선선ㅇ은 정확한 선형관계가 있을때 존재
실제 확률 드뭄
'''
def predict(alpha, beta,train,test):
    predictions=list()
    x=[i[0] for i in train]
    y=[j[1] for j in train]
    alpha, beta = OLS_fit(x,y)
    for i in test:
        yhat=alpha+beta*i[0]
        predictions.append(yhat)
    return predictions
predict(alpha,beta,train,test)

actual=[j[1] for j in test]
predicted=predict(alpha,beta,train,test)
actual, predicted

from math import sqrt

def RMSE(actual, predicted):
    sum_error = 0.0
    
    for i in range(len(actual)):
        prediction_error=predicted[i] - actual[i]
        
        sum_error+=(prediction_error**2)
        mean_error=sum_error/float(len(actual))
    return sqrt(mean_error)

RMSE(actual,predicted)








































