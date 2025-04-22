# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 14:22:51 2025

@author: Admin
"""

# 사용 모듈 import
import pandas as pd
import matplotlib.pyplot as plt

# 분석할 데이터 읽기
df = pd.read_csv("./data/csv/weather.csv", encoding="CP949")

# 날짜 컬럼을 datetime 형식으로 변환
df['일시'] = pd.to_datetime(df['일시'], errors='coerce')

# 'month' 컬럼 추가
df['month'] = df['일시'].dt.month

# 월별 데이터 저장 리스트 초기화
monthly = [None] * 12
monthly_wind = [0] * 12

# 1월 데이터 확인 (오류 수정 후 위치 변경)
monthly[0] = df[df['month'] == 1]
print(monthly[0].head())  # monthly[0]이 DataFrame일 때만 head() 호출

# 월별 데이터 분류 및 평균 풍속 계산
for i in range(12):
    monthly[i] = df[df['month'] == (i + 1)]
    monthly_wind[i] = monthly[i]['평균 풍속(m/s)'].mean()


plt.plot(monthly_wind, 'red')
plt.show()
