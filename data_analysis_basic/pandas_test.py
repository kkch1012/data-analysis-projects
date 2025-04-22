# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 09:26:09 2025

@author: Admin
"""

'''
Pandas의 특징
1. 빠르고 효율적 표현, 실세계 데이터 분석
2. 다양한 형태의 데이터 표현 가능
    서로 다른 형태의 데이터를 표현
        시계열, 레이블을 가진 데이터
    다양한 관측 데이터
3. Series: 1차원 / DataFrame: 행렬(2차원)
4. 결측(NULL) 데이터 처리
    데이터 추가/ 삭제
    데이터 정렬, 조작
'''


'''
Pandas로 할 수 있는 일

1. 리스트, 딕셔너리, Numpy 배열 등을 DataFrame으로 변환 가능
2. CSV/XLS 파일 등을 열어 작업 가능
3. URL 을 통해 웹 사이트의 csv, json, 과 같은 원격 데이터
   데이터베이스 등의 데이터를 다룰 수 있다.
4. 데이터 보기/ 검사 기능을 제공
5. 필터/ 정렬/ 그룹화
   sort_values() : 정렬
   groupby() : 기준에 따라 몇 개의 그룹화도 가능
6. 데이터 정제 
   데이터 누락, 특 값을 다른 값으로 일괄 변경 가능
'''


#Pandas 모듈 import
from pandas import Series, DataFrame

kakao = Series([92600, 92400, 92100 , 94300, 92300])
print(type(kakao))
print(kakao[0])
print(len(kakao))


'''
Series 객체 생성 시, 별도의 idx를 부여하지 않으면
기본적으로 0 시작하는 장소로 idx가 부여됨!!!
'''

# Series의 인덱스로 사용할 Series 객체 생성
idx = ['2021-12-01','2025-02-04','1212-12-12','1998-05-10','1998-04-14']
data = [92600,92400,93100,92100,92300]

sample = Series(data, index=idx)

mine = Series([10,20,30], index=['naver','kt','sk'])
friend = Series([10,20,30],index=['kt','naver','sk'])

merge = mine + friend
sub = mine - friend
mul = mine * friend
div = mine / friend

'''
DataFrame 생성 방법 : 주로 딕셔너리를 사용
딕셔너리를 통해 각 컬럼에 대한 데이터 저장한 후,
DataFrame의 생성자에게 전달.
'''

# DataFrame 객체 생성을 위한 딕셔너리 생성
raw_data = {'col0':[1,2,3,4],
            'col1':[10,20,30,40],
            'col2':[100,200,300,400]}

dataframe_data = DataFrame(raw_data)

'''
딕셔너리를 이용하여 데이터 프레임 객체를 생성하면
딕셔너리의 key가
데이터프레임의 컬럼명으로 자동 인덱싱 되고,
딕셔너리의 value에 해당하는 row에는
리스트처럼 0부터 시작하는 정수로 index가 인덱싱된다
'''



print(dataframe_data['col0'])

'''
0    1
1    2
2    3
3    4
Name: col0, dtype: int64
'''

print(type(dataframe_data['col0']))



daeshin = {'open': [11650, 11100, 11200, 11100, 11000],
           'high': [12100,11800,11200,11100,11150],
           'low' : [11600, 11050, 10900, 10950, 10980],
           'close': [11900,11600,12300,14700,11100]}

daeshin_day = DataFrame(daeshin)


daeshin_day2 = DataFrame(daeshin, 
                         columns = ['open','low','close','high'])


dataframe_date = ['2021-12-01','2025-02-04','1212-12-12','1998-05-10','1998-04-14']


daeshin_day3 = DataFrame(daeshin,columns = ['open','low','close','high']
                         ,index=dataframe_date)


print(daeshin_day3['open'])

'''
2021-12-01    11650
2025-02-04    11100
1212-12-12    11200
1998-05-10    11100
1998-04-14    11000
Name: open, dtype: int64
'''
print(daeshin_day3['1212-12-12':'2025-02-04'])


'''
******* 숫자 관련 Numpy 모듈의 기능 사용하는 방법 ***

'''

import numpy as np
import pandas as pd
import random
lst = [1, 3, 5, np.nan, 6, 8]
s = pd.Series(lst)

import numpy as np
import pandas as pd
import random
dates = ['2021-12-01','2021-12-10','2021-12-11','2021-12-12','2021-12-14','2021-12-15']
df = pd.DataFrame(np.random.randn(6, 4),
                  index = dates,
                  columns=['A','B','C','D'])


df.head()
df.head(3)
df.tail()
df.tail(2)

df.index

df.columns

df.values

df.info()
'''
<class 'pandas.core.frame.DataFrame'>
Index: 6 entries, 2021-12-01 to 2021-12-15
Data columns (total 4 columns):
 #   Column  Non-Null Count  Dtype  
---  ------  --------------  -----  
 0   A       6 non-null      float64
 1   B       6 non-null      float64
 2   C       6 non-null      float64
 3   D       6 non-null      float64
dtypes: float64(4)
memory usage: 240.0+ bytes
'''


df.describe()


df['A']

df[0 : 3]

df['2021-12-10' : '2021-12-12']


df.loc['2021-12-10' : '2021-12-12',['A','B']]


df2 = df.copy()

df2['E'] = ['one','two','three','four','one','two']

df2['E'].isin(['two','three'])


df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'], 
                    'B': ['B0', 'B1', 'B2', 'B3'],
                    'C': ['C0', 'C1', 'C2', 'C3'],
                    'D': ['D0', 'D1', 'D2', 'D3']},
                   index=[0, 1, 2, 3])

df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
                    'B': ['B4', 'B5', 'B6', 'B7'],
                    'C': ['C4', 'C5', 'C6', 'C7'],
                    'D': ['D4', 'D5', 'D6', 'D7']},
                   index=[4, 5, 6, 7])

df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],
                    'B': ['B8', 'B9', 'B10', 'B11'],
                    'C': ['C8', 'C9', 'C10', 'C11'],
                    'D': ['D8', 'D9', 'D10', 'D11']},
                   index=[8, 9, 10, 11])
result = pd.concat([df1, df2, df3])

result = pd.concat([df1, df2, df3],
                   keys=['x', 'y', 'z'])

print(result)


df4 = pd.DataFrame({'B':['B2','BB3','B6','B7'],
                    'D':['D2','D3','D6','D7'],
                    'F':['F2','F3','F6','F7']
                    },
                   index = [2,3,6,7])

result = pd.concat([df1,df4])
result = pd.concat([df1,df4],axis=1)

result = pd.concat([df1,df4],axis=1, join="inner")

result = pd.concat([df1,df4],join="inner")

result = pd.concat([df1,df4],join="inner",ignore_index=True)

left = pd.DataFrame({'key': ['K0', 'K4', 'K2', 'K3'],
                     'A': ['A0', 'A1', 'A2', 'A3'],
                     'B': ['B0', 'B1', 'B2', 'B3']})

right = pd.DataFrame({'key': ['K0', 'K1', 'K2', 'K3'],
                      'C': ['C0', 'C1', 'C2', 'C3'],
                      'D': ['D0', 'D1', 'D2', 'D3']})

pd.merge(left,right,on='key')


pd.merge(left,right, on='key',how='outer')

pd.merge(left,right, on='key',how='inner')

import pandas as pd

df = pd.read_csv("./data/csv/weather.csv", 
                 index_col=0, 
                 encoding="cp949")  # 한글 파일일 경우


'''
Pandas의 read_csv() 함수는 
인덱스 지정없이 파일을 읽을 경우
파일의 첫 번째 행을 각 시리즈의 열이름으로 자동 설정하고,
각 레코드에 대한
인덱스를 0으로 시작하는 함수를 이용하여 자동 생성한다.
'''

df_count = pd.read_csv("./data/csv/countries.csv",
                       index_col=0)

print(df_count["population"])


print(df_count[ ['population','area']])


df_count.head()

df_count[:3]

df_count.loc['US','capital']


df_count['population'][:3]

df_count['density'] = df_count['population']/df_count['area']

pandas_std = df['평균기온(C)'].std()
df.head()
import numpy as np
numpy_std = np.std(df['평균기온(°C)'])

numpy_std

# 사용 모듈 import

# 분석할 데이터 읽기

# 분석시 필요한 데이터 저장 리스트 선언

# '일시' 컬럼의 데이터에서 2020-07-30 부분을 
# DataTime 형식의 index를 만들어서
# 데이터 프레임에 신규컬럼('month')에 추가

# 월별로 분리하여 저장 테스트

# 전체 데이터를 이용하여 1 ~ 12 월까지의 평균 풍속을 저장

# matplotlib를 이용한 간단한 시각화














