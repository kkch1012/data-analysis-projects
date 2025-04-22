# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 09:21:53 2025

@author: Admin
"""

import pandas as pd
df = pd.read_excel('./files/kto_total.xlsx')


# 한글을 표기하기 위한 글꼴 변경(윈도우, macOS에 대해 각각 처리)
from matplotlib import font_manager, rc
import platform
if platform.system() == 'Windows':
    path  = 'c:/Windows/Fonts/malgun.ttf'
    font_name=font_manager.FontProperties(fname = path).get_name()
    rc('font',family=font_name)
elif platform.system() == 'Darwin':
    rc('font', family='AppleGothic')
else:
    print('Check your OS system')

yer = df_filter['기준년월']

import matplotlib.pyplot as plt

### 중국인 관광객 시계열 ###
# 1. 중국 국적 데이터 필터링 => df_filter
# df_filter = df[df['국적'] == '중국'] - 내가 한 필터링

condition=(df['국적']=='중국')
df_filter = df[condition]

# 2. 시계열 : plot(x, y)

plt.plot(df_filter['기준년월'], df_filter['관광'])
plt.show()
# 위 그래프 해석하기 힘듦
# 3. 시게열 : 그래프 크기 조절/타이틀, X축 Y축 이름 / x축 눈금 값
# 그래프 크기 조절 : figure() => figsize=( , )
plt.figure(figsize=(12 , 4))

# 그래프 데이터 설정
plt.plot(df_filter['기준년월'], df_filter['관광'])

# 타이틀: title()
plt.title('중국의 관광 데이터')
# x축 : xlabel()
plt.xlabel('기준년월')
# y축 : ylabel()
plt.ylabel('관광객수')
# x축 눈금 : xticks() => [ , , ]
plt.xticks(yer)
plt.show()

### 국내 외국인 관광객 중 상위 5개 국가를 각각 시게열
### (중국, 일본, 대만, 미국, 홍콩)
country = ['중국','일본','대만','미국','홍콩']

# 2. 위의 시각화 코드를 매번 사용? / 반복처리 / 함수처리
# 반복 처리
mis = pd.DataFrame()
for cn in country:
    
    mis = pd.concat([df[df['국적'] == cn],mis], ignore_index=True)

mis

country = ['중국','일본','대만','미국','홍콩']

for cntry in country:
    df_filter = df[df['국적'] == cntry]
    
    plt.figure(figsize=(12,4))
    plt.plot(df_filter['기준년월'], df_filter['관광'])

    # 타이틀: title()
    plt.title('{} 국적의 관광객 추이'.format(cntry))
    # x축 : xlabel()
    plt.xlabel('기준년월')
    # y축 : ylabel()
    plt.ylabel('관광객수')
    # x축 눈금 : xticks() => [ , , ]
    plt.xticks(['2010-01', '2011-01', '2012-01', '2013-01', '2014-01', '2015-01', '2016-01', '2017-01', '2018-01', '2019-01','2020-01']
)
    plt.show()


# 함수 처리 : 관광/ 유학/ 기타
def plot_test(cntry. t) :
    df_filter = df[df['국적'] == cntry]
    
    plt.figure(figsize=(12,4))
    plt.plot(df_filter['기준년월'], df_filter['관광'])

    # 타이틀: title()
    plt.title('{} 국적의 관광객 추이'.format(cntry))
    # x축 : xlabel()
    plt.xlabel('기준년월')
    # y축 : ylabel()
    plt.ylabel('관광객수')
    # x축 눈금 : xticks() => [ , , ]
    plt.xticks(yer)
    plt.show()


### 히트맵 ###
'''
매트릭스 형태에
값을 컬러로 표현하는 데이터 시각화 방법
장점: 전체 데이터를 한눈에 파악할 수 있다.

X축, Y축에 어떤 변수들을 사용할 지를 고민해야 한다.
'''

# x축 : 월(Month), Y축 : 연(Year)
# 데이터: 관광객 수
# => 연도와 월로 구분된 변수를 생성 : 기준년월 => 2010-01
df['기준년월'] = df['기준년월'].astype(str)  # 정수를 문자열로 변환
df['년도'] = df['기준년월'].str.slice(0, 4)  # 앞 4자리(년도)
df['월'] = df['기준년월'].str.slice(4, 6)    # 5~6번째 자리(월)

# 원하는 국적 데이터만 추출 : df_filter
condition = (df['국적'] == '중국')
df_filter = df[condition]
# df_filter 데이터를 매트릭스 형태로 변환 : pivot_table()
# index = '년도'
# columns = '월'
# values = '관광'

df_pivot = df_filter.pivot_table(index = '년도',
                                 columns='월',
                                 values='관광')

import seaborn as sns
plt.figure(figsize=(16 ,10))

sns.heatmap(df_pivot,
            annot=True,
            fmt='.0f',
            cmap= 'rocket_r')
plt.title('중국 관광객 히트맵')
plt.show()

























