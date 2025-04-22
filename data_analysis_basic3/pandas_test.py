# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 15:31:24 2025

@author: Admin
"""

import pandas as pd

path = "./data/"
file = path+'vehicle_prod.csv'
df = pd.read_csv(file, index_col=0)
'''
         2007   2008   2009   2010   2011
China    7.71   7.95  11.96  15.84  16.33
EU      19.02  17.71  15.00  16.70  17.48
US      10.47   8.45   5.58   7.60   8.40
Japan   10.87  10.83   7.55   9.09   7.88
Korea    4.04   3.78   3.45   4.20   4.62
Mexico   2.01   2.05   1.50   2.25   2.54
'''
df['total'] = df.sum(axis=1)

'''
         2007   2008   2009   2010   2011  total
China    7.71   7.95  11.96  15.84  16.33  59.79
EU      19.02  17.71  15.00  16.70  17.48  85.91
US      10.47   8.45   5.58   7.60   8.40  40.50
Japan   10.87  10.83   7.55   9.09   7.88  46.22
Korea    4.04   3.78   3.45   4.20   4.62  20.09
Mexico   2.01   2.05   1.50   2.25   2.54  10.35
'''
df.head()

df['2009'].plot(kind='bar',color=('orange','r','b','m','c','k'))

df['2009'].plot(kind='pie')

df.plot.line()

df.plot.bar()

df = df.transpose()

df.plot.line()

df.loc['Korea','2011']
df.loc['Korea']

# 인덱스 번호로 할려면 iloc
df.iloc[4]
'''
그룹핑과 필터링
'''
path = 'https://github.com/dongupak/DataML/raw/main/csv/'
weather_file = path + 'weather.csv'
weather = pd.read_csv(weather_file, encoding='CP949')

weather['month'] = pd.DatetimeIndex(weather['일시']).month

weather.head()
'''
groupby('기준 컬럼')
'''

month_mean = weather.groupby('month').mean()

'''
데이터 구조를 변경하는 pivot()
index = 어떤 컬럼을 index
columns = 어떤 컬럼을 열로 하는
values = 어떤 컬럼을 값으로 쓰겟다

fillna() : 존재하지 않는 값 => 지정한 값으로 채우겠다
value = 
'''


df = pd.DataFrame({'상품' : ['시계','반지','목걸이','반지','팔찌'],
                   '재질' : ['금','은','금','백금','은'],
                   '가격' : [500000,20000,350000,300000,60000]})
df
'''
상품  재질      가격
0   시계   금  500000
1   반지   은   20000
2  목걸이   금  350000
3   반지  백금  300000
4   팔찌   은   60000
'''
new_df = df.pivot(index='상품',
                  columns='재질',
                  values='가격')
new_df
'''
재질          금        백금        은
상품                              
목걸이  350000.0       NaN      NaN
반지        NaN  300000.0  20000.0
시계   500000.0       NaN      NaN
팔찌        NaN       NaN  60000.0
'''

new_df = new_df.fillna(value=0)

new_df

# 두 개의 데이터프레임을 하나로 합치는 concat() / merge()
# outer inner left right how











































