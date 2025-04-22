# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 14:59:53 2025

@author: Admin
"""

import pandas as pd
import matplotlib.pyplot as plt

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
    
# youtube_rank xlsx
df = pd.read_excel('./files/youtube_rank.xlsx')

# 데이터
df.head()
df.tail()

# 구독자수 수 : 0부터 10개만
df['subscriber'][0:10]

# 만 => 0000
df['subscriber'].str.replace('만', '0000')
df['replaced_subscriber'] = df['subscriber'].str.replace('만','0000')
df.head()

df.info()

df['replaced_subscriber'] = df['replaced_subscriber'].astype('int')

# 구독자 수 => 파이차트 => 채널###
# category => 카테고리 갯수
# replaced_subscriber => 카테고리별로 더하기
# 구독자 수, 채널 수 피봇 테이블 생성
# 데이터프레임.pivot_table()
# index = 'category'
# values = 'replaced_subscriber
# aggfunc = ['sum','count']
pivot_df = df.pivot_table(index='category',
                          values='replaced_subscriber',
                          aggfunc=['sum','count'])

pivot_df.head()

# 데이터프레임의 컬럼명 변경
pivot_df.columns = ['subscriber_sum', 'category_count']
pivot_df.head()

# 데이터프레임의 인덱스 초기화
pivot_df = pivot_df.reset_index()

# 데이터프레임을 내림차순 정렬
pivot_df = pivot_df.sort_values(by='subscriber_sum', ascending=False)

plt.figure(figsize=(30,10))
plt.pie(pivot_df['subscriber_sum'],
                 labels=pivot_df['category'],
                 autopct='%1.1f%%')
plt.show()

# 카페고리별 채널 수 시각화
pivot_df = pivot_df.sort_values(by='category',ascending=False)

plt.pie(pivot_df['category_count'],
        labels=pivot_df['category'],
        autopct='%1.1f%%')
plt.show()








































