# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 14:02:03 2025

@author: Admin
"""

import pandas as pd

d = pd.DataFrame({'date':['2019-01-03','2021-11-22','2023-01-05'],'name':['J','Y','O']})

d.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 3 entries, 0 to 2
Data columns (total 2 columns):
 #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
 0   date    3 non-null      object
 1   name    3 non-null      object
dtypes: object(2)
memory usage: 180.0+ bytes
'''

d['date'] = pd.to_datetime(d.date, format='%Y-%m-%d')
'''
 #   Column  Non-Null Count  Dtype         
---  ------  --------------  -----         
 0   date    3 non-null      datetime64[ns]
 1   name    3 non-null      object        
dtypes: datetime64[ns](1), object(1)
memory usage: 180.0+ bytes
'''
### datetime 형의 컬럼을 인덱스로 설정 : set_index()
d.set_index(keys=['date'],inplace=True)

### 결측치 확인: isnull(), sum 함수 사용 
import numpy as np
d = pd.DataFrame({'date':['2019-01-03','2021-11-22','2023-01-05','2025-02-21'],
                 'x1':[0.1, 2.0, np.nan, 1.2]})
d['date'] = pd.to_datetime(d.date, format='%Y-%m-%d')
d.set_index(keys=['date'],inplace=True)
d.isnull().sum()


### 결측치 처리: fillna(method='ffill' / drop() / interpolate())
d = d.fillna(method='ffill') # 결측치를 이전값으로 대체

d = pd.DataFrame({'date':['2019-01-03','2021-11-22','2023-01-05','2025-02-21'],
                 'x1':[0.1, 2.0, np.nan, 1.2]})
d= d.dropna()

d = d.interpolate() # 선형보간법 
'''
시간에 따라 시스템이 동작하는 방식을 미리 알고 있을 때
'''
### 빈도 설정 : index 속성 / asfreq() ###
d = pd.DataFrame({'date':['2019-01-03','2021-11-22','2023-01-05','2025-02-21'],
                 'x1':[0.1, 2.0, np.nan, 1.2]})
d['date'] = pd.to_datetime(d.date, format='%Y-%m-%d')
d.set_index(keys=['date'],inplace=True)
print(d.index)

d2 = d.asfreq('Y',method='ffill') # method='ffill' : 이전

### 특징량 만들기 : rolling() ###
# 시계열 데이터에서 빈도 설정 넒게 잡아 한묶음으로 설정했을 경우,
# 패턴에 대한 본질이 묻힐 수 있다!!
# 방지 : rolling() 데이터를 shift 하여 더 상세한 특징량을 생성!!
d = pd.DataFrame({'date':['2019-01-03','2021-11-22','2023-01-05','2025-02-21','2026-02-20'],
                 'x1':[5,4,3,2,7]})
d['date'] = pd.to_datetime(d.date)
d.set_index(keys=['date'],inplace=True)
'''
index 0     1   2   3   4
-------------------------
date 5,     4,  3,  2,  7


'''
d.rolling(2).mean

### 이전 값과 차이 계산 : diff()


### 자연값 추출 : shift() 사용

'''
지연값: 시계열 데이터의 경우 특정 값이 미래의 값을 영향을 주는 경우가 있다.
이런 특성을 반영하기 위해 shift()를 사용
shift(2) : 데이터가 2개씩 뒤로 밀리는 것을 의미.
             결측치가 발생! => 
'''

d = pd.DataFrame({'date':['2019-01-03','2021-11-22','2023-01-05','2025-02-21','2026-02-20'],
                 'x1':[5,4,3,2,7]})
d['date'] = pd.to_datetime(d.date)
d.set_index(keys=['date'],inplace=True)

d['shift'] = d['x1'].shift(2)
d = d.fillna(method='bfill')
### 원 핫 인코딩

'''
범주형 데이터를 컴퓨터가 처리할 수 있도록 형태를 변환하는 방법
텍스트 형태 =>
카테고리형 데이터의 각 범주를 하나의 열로 만들고
헤당 데이터가 있는 곳만 1을 표시, 0으로 표시 =>
     숫자형으로 인식할 수 있게 하는 방법
'''

d = pd.DataFrame({'date':['2021-01-06',
                          '2021-01-13',
                          '2021-01-20',
                          '2021-01-27',
                          '2021-02-03'],
                  'x1':[5,4,3,2,7],
                  '과목':['a','b','c','d','e']})
d['date'] =pd.to_datetime(d.date)
d.set_index(keys=['date'],inplace=True)

x = pd.get_dummies(d['과목'])





































