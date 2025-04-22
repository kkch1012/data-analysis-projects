# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 10:01:54 2025

@author: Admin
"""
import pandas as pd

sr = pd.Series([17000,10000,1000,5000],
                index=["피자","치킨","콜라","맥주"])
print('시리즈의 값: {}'.format(sr.values))
print('시리즈의 인덱스: {}'.format(sr.index))

values = [[1,2,3], [4,5,6],[7,8,9]]
index = ['one','two','three']
columns = ['A','B','C']

df = pd.DataFrame(values, index=index, columns= columns)
df.index
df.columns
df.values

# 리스트로 생성
data = [
        ['1000','Steve',90.72],
        ['1001','James',78.09],
        ['1002','Doyeon',98.43],
        ['1003','Jane',64.19],
        ['1004','Pilwoong',81.30],
        ['1005','Tony',99.14]
        ]

df = pd.DataFrame(data)

df = pd.DataFrame(data, columns=['학번','이름','점수'])

# 딕셔너리로 생성
data = {
        '학번': ['1000','1001','1002','1003','1004','1005'],
        '이름': ['luke','jack','james','scalet','johnson','wilder'],
        '점수': [97.98, 78.09, 56.34,31.41,59.26,53.58]
        }

df= pd.DataFrame(data)

print(df['학번'])


import numpy as np

vec = np.array([1,2,3,4,5])

mat = np.array([[10,20,30],[60,70,80]])

print('vec의 차원:',vec.ndim)
print('vec의 크기{shape} : ',vec.shape)

## ndarray의 초기화 ###
# 모든 값이 0인 2x3 배열 생성
zero_mat = np.zeros((2,3))

# 모든 값이 1인 2x3 배열 생성
one_mat = np.ones((2,3))

# 모든 값이 특정 수인 배열 생성
same_value_mat = np.full((2,2), 7)

# 대각선 값이 1이고 나머지 값이 0인 2차원 배열을 생성.
eye_mat = np.eye(3)

# 임의의 값으로 채워진 배열 생성
random_mat = np.random.random((2,2))

### np.arrange() ###
# 0부터 9까지
range_vec = np.arange(10)

# 1부터 9까지 _2 적용되는 범위
n=2
range_n_step_vec = np.arange(1,10,n)

### reshape_mat 
reshape_mat = np.array(np.arange(30)).reshape

### Numpy 슬라이싱 ###

# 두번째 열 출력
slicing_mat = mat[:, 1]

### Numpy 정수 인덱싱(integer indexing)
mat = np.array([[1,2],[4,5],[7,8]])

# 1행 0열의 원소
print(mat[1,0])

# mat[[2행,1행],[0열, 1열]]
indexing_mat = mat[[2,1],[0,1]]

### Numpy 연산

x = np.array([1,2,3])
y = np.array([4,5,6])

result = x+y
print(result)

result = x - y
print(result)

result = result / x
print(result)

# result = np.divide(result,x)와 동일

### Numpy에서 벡터와 행렬곱 또는 행렬곱을 위해서는 dot()을 사용 ###
mat1 = np.array([[1,2],[3,4]])
mat2 = np.array([[5,6],[7,8]])

mat3 = np.dot(mat1, mat2)











































