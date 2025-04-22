# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 15:02:28 2025

@author: Admin
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

dust = pd.read_excel('./data_ana/environment/dust.xlsx')
weather = pd.read_excel('./data_ana/environment/weather.xlsx')

### 데이터 수집 ###
## 미세먼지 데이터 ##
dust.shape # Out[68]: (744, 7)

dust.info()

### 데이터 가공 ###
# 컬럼의 이름을 영문으로 변경
dust.rename(columns={'날짜':'date','아황산가스':'so2',
                    '일산화탄소':'co','오존':'o3',
                    '이산화질소':'no2'},inplace=True)
# 날짜 데이터에서 년도-월-일만 추출
# '2021-01-01 01' => '2021-01-01'
dust['date']=dust['date'].str[:11]

# 날짜 컬럼의 자료형을 날짜형으로 변환
dust['date']=pd.to_datetime(dust['date'])

# 날짜 컬럼에서 년도, 월, 일을 추출하여 각각 새로운 컬럼으로 추가
# 후. 여러 년도로 분석 시 필요할 수도 있기 때문에
dust['year'] = dust['date'].dt.year
dust['month'] = dust['date'].dt.month
dust['day'] = dust['date'].dt.day

#  새롭게 추가한 컬럼 순서 재정렬
dust=dust[['date','year','month','day','so2','co','o3','no2','PM10','PM2.5']]

### 데이터 전처리 ###
# 각 컬럼별(변수) 결측치(null) 수 확인

dust.isnull().sum()
# dust.isna().sum()

# 시계열분석이므로 null값을 이전 시간의 값을 기준으로 채워줌
# 결측값을 앞-방향 혹은 뒷 방향으로 채우기
dust=dust.fillna(method='pad')

# 이전값이 없는 경우 혼자 Nan은 20으로 채워줌
dust.fillna(20,inplace=True)

# dust['month']=dust['month'].astype(int) => int64

weather.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 743 entries, 0 to 742
Data columns (total 7 columns):
 #   Column   Non-Null Count  Dtype         
---  ------   --------------  -----         
 0   지점       743 non-null    int64         
 1   지점명      743 non-null    object        
 2   일시       743 non-null    datetime64[ns]
 3   기온(°C)   743 non-null    float64       
 4   풍속(m/s)  743 non-null    float64       
 5   강수량(mm)  743 non-null    float64       
 6   습도(%)    743 non-null    float64       
dtypes: datetime64[ns](1), float64(4), int64(1), object(1)
memory usage: 40.8+ KB
'''
### 데이터 가공 ###
# 분석에 필요 없는 컬럼 제거
weather.drop('지점', axis=1, inplace=True)
weather.drop('지점명',axis=1,inplace=True)

# 특수 기호가 포함된 컬럼명을 변경
weather.columns=['date','temp','wind','rain','humid']

# 미세먼저데이터와 동일한 타입을 위해
# 컬럼 일부 데이터(시간) 제거한 후,
weather['date'] = pd.to_datetime(weather['date']).dt.date
# 데이터 타입 변경
weather['date'] = weather['date'].astype('datetime64[ns]')

# 결측치 확인
weather.isnull().sum()

# 강수량 데이터 확인
weather['rain'].value_counts()

weather['rain'] = weather['rain'].replace([0],0.01)
weather['rain'].value_counts()

#  데이터 병합 #
# 미세먼지 데이터와 날씨 데이터 병합하기 위해
# 두 데이터 프레임의 차원을 파악
dust.shape
# Out[92]: (744, 10)
weather.shape
# Out[93]: (743, 5)

dust.drop(index=743, inplace=True)

df=pd.merge(dust,weather, on='date')

### 데이터 분석 및 시각화 ###
# 미세먼지 데이터와 날씨 데이터의 모든 요소별 상관관계 확인
df.corr()

corr = df.corr()


corr['PM10'].sort.values(ascending=False)

# 히스토그램으로 시각화
df.hist(bins=50,figsize=(20,15))


plt.figure(figsize=(15,10))
sns.barplot(x='day',y='PM10',data=df,palette='Set1')
plt.xticks(rotation=0)
plt.show()

# 히트맵 상관관계
plt.figure(figsize=(15,10))
sns.heatmap(data=corr,annot=True,fmt='.2f',cmap='hot')
plt.show()

'''
pm10,pm2.5,no2,co : 이들은 모두 대기 오염물질이기에 관련성 있음
o3와 wind : 바람과 오존이 약한 관게성
'''

# 산점도 그래프로 시각화1 : 온도와 미세먼지 상관관계
plt.figure(figsize=(15,10))
x=df['temp']
y=df['PM10']
plt.plot(x,y,marker='o', linestyle='none',alpha=0.5)
plt.title('temp , pm10')
plt.xlabel('temp')
plt.ylabel('pm10')
plt.show()




















