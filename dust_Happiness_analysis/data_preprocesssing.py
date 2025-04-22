# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 09:02:54 2025

@author: Admin
"""

import pandas as pd

df = pd.read_excel('./data/medical.xls')
df.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 65535 entries, 0 to 65534
Data columns (total 9 columns):
 #   Column          Non-Null Count  Dtype 
---  ------          --------------  ----- 
 0   PatientId       65535 non-null  int64 
 1   AppointmentID   65535 non-null  int64 
 2   Gender          65535 non-null  object
 3   ScheduledDay    65535 non-null  object
 4   AppointmentDay  65535 non-null  object
 5   Age             65535 non-null  int64 
 6   Neighbourhood   65535 non-null  object
 7   SMS_received    65535 non-null  int64 
 8   No-show         65535 non-null  object
 '''

data = pd.DataFrame({'A':['1','2','3'],
                     'B':['4','5','6'],
                     'C':['7','8','9']})
data.info()

data[['A','B','C']] = data[['A','B','C']].apply(pd.to_numeric)

### 2. 결측치
# 1. 결측값 제거 : DataFrame.dropna()
data = pd.DataFrame({'A':['1','2','3',None],
                     'B':['4',None,'5','6'],
                     'C':[None,'7','8','9']})
data.info()

# 결측값이 있는 행을 제거 
data_dropna =data.dropna(axis=1)
data_dropna.info()


# 3.결측값 평균으로 채우기
data_fillna = data.fillna(data.mean())
# int형으로
data[['A','B','C']] = data[['A','B','C']].apply(pd.to_numeric)

# 각 열의 결측값을 해당 열의 최빈값으로 채운 데이터프레임
data_fillna = data.apply(lambda x: x.fillna(x.mode()[0]))

# apply() : 각 열에 대하여 적용시켜주는 함수
# mode() : 각 열의 최빈값 계산
# mode()[0] : 가장 빈도가 높은 값을 반환


### 3. 이상치 찾기 ###
# 이상치 찾기
import numpy as np
import matplotlib.pyplot as plt

df = pd.DataFrame(np.random.randn(8,3),
                  columns=['C1','C2','C3'])

df.loc[1,'C1'] = 11
df.loc[3,'C3'] = -10

# 박스플롯으로 시각화
plt.boxplot([df['C1'],df['C3']])
plt.show()

### 4.목적에 맞는 특정 변수 추출 방법
# 변수간의 상관 분석을 이용하여 추출
import seaborn as sns
df = pd.DataFrame(np.random.randn(8,3),
                  columns=['C1','C2','C3'])
df.loc[1,'C1'] = 11
df.loc[3,'C3'] = -10
sns.heatmap(df.corr(),annot=True)
plt.show()




























