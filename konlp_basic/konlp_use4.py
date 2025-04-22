# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 14:14:03 2025

@author: Admin

데이터의 분리
"""

X, y = zip(['a',1],['b',2],['c',3])


import Numpy as np

np_array = np.arange(0,16).reshape((4,4))
'''
array([[ 0,  1,  2,  3],
       [ 4,  5,  6,  7],
       [ 8,  9, 10, 11],
       [12, 13, 14, 15]])
'''

x =  np_array[:, :3]
'''
array([[ 0,  1,  2],
       [ 4,  5,  6],
       [ 8,  9, 10],
       [12, 13, 14]])
'''
y = np_array[:,3]


### 테스트 데이터 분리
# 1) 사이킷 런을 이용하여 분리 : train_test_split()
# X_train,X_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=1234)

'''
x: 독립 변수 데이터
y: 종속 변수 데이터, 레이블 데이터
test_size : 테스트용 데이터 개수를 지정
train_size : 학습용 데이터의 개수를 지정
random_state : 난수 시도
'''

# 전체 데이터
X, y = np.arange(10).reshape((5,2)),range(5)

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,
                                                 random_state=1234)

'''
X_train
array([[2, 3],
       [4, 5],
       [6, 7]])

X_test
Out[215]: 
array([[8, 9],
       [0, 1]])
y_train
Out[216]: [1, 2, 3]
y_test
Out[217]: [4, 0]

'''

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,
                                                 random_state=1)


'''
X_train
Out[219]: 
array([[8, 9],
       [0, 1],
       [6, 7]])
'''



















