# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 12:36:37 2025

@author: Admin
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df_dust = pd.read_excel('./data_ana/environment/dust.xlsx')
df_weather = pd.read_excel('./data_ana/environment/weather.xlsx')

test = df_dust['날짜'].split(' ')[1]
hour_every = []
day_every = []
for tg in df_dust['날짜']:
    hour = tg.split(' ')[1]
    hour_every.append(hour)
    day = tg.split(' ')[0].split('-')[2]
    day_every.append(day)
time_every = [f'{x} {y}'for x, y in zip(day_every,hour_every)]

df_dust['일시'] = time_every
hour_every2 = []
day_every2 = []
for tst in df_weather['일시']:
    hour_wea = str(tst).split(' ')[1].split(':')[0]
    hour_every2.append(hour_wea)
    day_wea = str(tst).split(' ')[0].split('-')[2]
    day_every2.append(day_wea)
time_every2 = [ f'{x} {y}'for x, y in zip(day_every2,hour_every2)]
df_weather['일시'] = time_every2



df_merged = pd.merge(df_dust,df_weather, left_on='일시',right_on='일시',how='inner')

df_merged_update = df_merged.drop('일시',axis=1)
df_merged_update = df_merged_update.drop('지점명',axis=1)



df_dust.info()
df_weather.info()


# merge로 두 데이터를 dust - 날짜 weather - 일시 기준으로 합치기

df_dust['날짜'][0]
# Out[11]: '2021-01-01 01'
df_weather['일시'][0]
# Out[12]: Timestamp('2021-01-01 01:00:00')

df_dust['PM2.5']
df_dust['PM10']

correlation = df_dust[['PM10', 'PM2.5']].corr()

# 시각화
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df_dust['PM10'], y=df_dust['PM2.5'])
plt.title('PM10 vs PM2.5')
plt.xlabel('PM10')
plt.ylabel('PM2.5')
plt.grid(True)
plt.show()

# 상관 행렬 히트맵
plt.figure(figsize=(6, 5))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('PM10 & PM2.5 상관관계')
plt.show()


# 미세먼지 변수 중 대기오염과 관련된 변수

df_dust['아황산가스']

correlation_2 = df_dust[['아황산가스','일산화탄소','오존','이산화질소','PM10']].corr()
correlation_2

'''
          아황산가스     일산화탄소        오존     이산화질소      PM10
아황산가스  1.000000         0.136025        -0.062404  0.078900  0.156999
일산화탄소  0.136025  1.000000 -0.755688  0.841014  0.529966
오존    -0.062404 -0.755688  1.000000 -0.923828 -0.345387
이산화질소  0.078900  0.841014 -0.923828  1.000000  0.416201
PM10   0.156999  0.529966 -0.345387  0.416201  1.000000
'''
# => 일산화탄소,이산화질소,오존,아황산가스 순으로 

corre_3 = df_dust[['일산화탄소','이산화질소']].corr()
corre_3

'''
          일산화탄소     이산화질소
일산화탄소  1.000000  0.841014
이산화질소  0.841014  1.000000
'''

from matplotlib import font_manager,rc
import platform

if platform.system() == 'Windows':
    path  = 'c:/Windows/Fonts/malgun.ttf'
    font_name=font_manager.FontProperties(fname = path).get_name()
    rc('font',family=font_name)
elif platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
else:
    print('Check your OS system')
    
plt.figure(figsize=(10,6))
sns.scatterplot(x=df_dust['일산화탄소'],y=df_dust['이산화질소'])
plt.title('일산화탄소 vs 이산화질소')
plt.xlabel('일산화탄소')
plt.ylabel('이산화질소')
plt.grid(True)
plt.show()


corre_4 = df_merged_update[['오존','풍속(m/s)']].corr()
corre_4
'''
               오존   풍속(m/s)
오존       1.000000  0.632346
풍속(m/s)  0.632346  1.000000
'''
plt.figure(figsize=(10,6))
sns.scatterplot(x=df_merged_update['오존'],y=df_merged_update['풍속(m/s)'])
plt.title('오존 vs 풍속')
plt.xlabel('오존')
plt.ylabel('풍속')
plt.grid(True)
plt.show()

corre_5 = df_merged_update[['기온(°C)','PM10', 'PM2.5']].corr()
corre_5
'''
        기온(°C)      PM10     PM2.5
기온(°C)  1.000000  0.251522  0.229701
PM10    0.251522  1.000000  0.836345
PM2.5   0.229701  0.836345  1.000000

'''# 상관계수 히트맵
plt.figure(figsize=(4,3))
sns.heatmap(corre_5, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap')
plt.show()

df_merged_update.columns

'''
Out[64]: 
Index(['날짜', '아황산가스', '일산화탄소', '오존', '이산화질소', 'PM10', 'PM2.5', '지점', '기온(°C)',
       '풍속(m/s)', '강수량(mm)', '습도(%)'],
      dtype='object')
'''

# 산점도 그래프로 시각화 : 미세먼지와 초미세먼지 상관관계
plt.figure(figsize=(15,10))
x=df['PM10']
y=df['PM2.5']

plt.plot(x, y, marker='o',linestyle='none',color='red',alpha=0.5)
plt.title('pm10-pm2.5')
plt.xlabel('pm10')
plt.ylabel('pm2.5')
plt.show()







































