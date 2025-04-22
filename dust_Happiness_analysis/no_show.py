# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 09:02:59 2025

@author: Admin
"""
### 1.데이터 일기와 확인 ###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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
### 2. 결측치 확인 : isnull().any(axis=1)
df.isnull().any(axis=1)
df.isnull().any(axis=0)
### 3. 통계량을 이용하여 이상치 제거
df.describe()

df = df[df.Age>=0]
### 4.데이터 타입 변환 ###
df['No-show'] = df['No-show'].map({'Yes':1,'No':0})
df['No-show'].value_counts()

df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
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
 3   ScheduledDay    65535 non-null  datetime64[ns, UTC]
 4   AppointmentDay  65535 non-null  datetime64[ns, UTC]
 5   Age             65535 non-null  int64              
 6   Neighbourhood   65535 non-null  object             
 7   SMS_received    65535 non-null  int64              
 8   No-show         65535 non-null  int64              
dtypes: datetime64[ns, UTC](2), int64(5), object(2)
memory usage: 4.5+ MB
'''
### 5. 새로운 변수 추가
df['Waiting_day'] = df['AppointmentDay'].dt.dayofyear - df['ScheduledDay'].dt.dayofyear
df.describe()

### 6. 값 확인하여 이상치 제거
df = df[df.Waiting_day >= 0]
df['Waiting_day'].min()

df.Age.unique()

plt.figure(figsize=(16,2))
sns.boxplot(x=df.Age)
plt.show()
# 그래프를 보아 튀는 데이터 발견
df = df[df.Age<=110]


### 7. 목적에 적합한 변수 추출
# 방문 당일 예약한 환자
a = df[df.Waiting_day==0]['Waiting_day'].value_counts()
'''a
Out[71]: 
Waiting_day
0    21789
Name: count, dtype: int64
'''
# 방문 당일 예약한 환자 노쇼
b = df[(df['Waiting_day']==0)&(df['No-show']==1)]['Waiting_day'].value_counts()
b
b/a
'''
b/a
Out[76]: 
Waiting_day
0    0.049475
Name: count, dtype: float64
'''
# 기다리는 날이 10일 이하
no_show = df[df['No-show']==1]
show = df[df['No-show']==0]

no_show[no_show['Waiting_day']<=10]['Waiting_day'].hist(alpha=0.7,label='no_show')
show[show['Waiting_day']<=10]['Waiting_day'].hist(alpha=0.3,label='show')
plt.legend()
plt.show()

''' 당일 예약에서는 노쇼가 거의 발생하지 않는다.. '''
no_show['ScheduledDay'].hist(alpha=0.7,label='no_show')
show['ScheduledDay'].hist(alpha=0.3,label='show')
plt.legend()
plt.show()
'''
2016년 5월 초 ~ 5월 말 가장 많이 예약
2016년 4월말 ~ 5월초 no-show가 가장 많다
'''
no_show['AppointmentDay'].hist(alpha=0.7,label='no_show')
show['AppointmentDay'].hist(alpha=0.3,label='show')
plt.legend()
plt.show()
''' No-show 발생 건수와 방문일 간에는 특이점이 없다'''
# 재방문 환자와 'No-show'
# 재방문 환자 : 환자의 병원 예약 횟수 : 환자 번호로 value_counts()
df.PatientId.value_counts().iloc[0:10]

# 상위 500명에 대한 예약 횟수 분포
df.PatientId.value_counts().iloc[0:500].hist()

# PatientId 와 waiting_day >= 50 & No-show == 1
df[(df['Waiting_day']>=30) & (df['No-show']==1)].PatientId.value_counts().iloc[0:10]
'''
예약 기간이 길수록 No - show 

'''

# SMS_received와 waiting_day, 'No-show 발생 횟수 확인
# 알림 메시지 허용 여부와 기다리는 기간에 따른 노쇼 발생 횟수 확인
sns.barplot(x='SMS_received',
            y='Waiting_day',
            hue='No-show',
            data=df)
plt.show()
'''
알림 메시지 허용하지 않았을 경우, 기다리는 기간 5일이상 No-show
알림 메시지 허용했을 경우, 기다리는 기간 18일 이상 No-show
'''


a = len(df[(df['SMS_received']==0)&(df['No-show']==1)])
b = len(df[(df['SMS_received']==0)&(df['No-show']==0)])

print(f'SMS가 0일때 노쇼:{a}')
print(f'SMS가 0일때 온:{b}')

c = len(df[(df['SMS_received']==1)&(df['No-show']==1)])
d = len(df[(df['SMS_received']==1)&(df['No-show']==0)])

print(f'SMS가 1일때 노쇼:{c}')
print(f'SMS가 1일때 쇼:{d}')

# 상관관계로 확인
# corr() SMS,waiting와 No-show 간의 상관관계를 구한뒤 heatmap사용
temp = df[['Waiting_day','SMS_received','No-show']].corr()
sns.heatmap(temp)
plt.show()
### 8. 노쇼에 특징 파악 정리
'''
1. 예약기간이 길수록 No-show
2. 3개월 안에 50회 이상의 재방문
3. 재방문 횟수 상위10명, 횟수는 많지 않지만 No-show
4. 알림 미허용시, 대기일수가 5일 이상이면 no-show
5. 알림 미허용시 대기일수가 18일 이상이면 No show
6. 알림 미허용시 No show
'''
'''
대기일수가 길수록 noshow
알림 미허용시 대기 일수가 5일이상 no show
짧은 기간에 재 방문 = 가끔 no show
'''
### 9. no_show와 연결하여 변수의 특성 파악
# 1. 얼마나 많은 환자가 예정된 약속에 오지 않았는가?

# 2. 성별에 따른 노쇼
sns.countplot(x='Gender',hue='No-show',data=df)
plt.show()

# 3. No-show의 남녀 비율
f = df[(df['Gender']=='F') & (df['No-show']==1)]['Gender'].value_counts()
m = df[(df['Gender']=='M') & (df['No-show']==1)]['Gender'].value_counts()

F = df[(df['Gender']=='F')]['Gender'].value_counts()
M = df[(df['Gender']=='M')]['Gender'].value_counts()

f/F
# F    0.214528
# Name: count, dtype: float64
m/M
# M    0.208294
# Name: count, dtype: float64

# 성별에는 영향을 미치지 않는다

### 해결 방안 ####
# 1. 대기 기간을 짧게
# 2. 알림허용 장려

















