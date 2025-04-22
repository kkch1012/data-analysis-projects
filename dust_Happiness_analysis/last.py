# -*- coding: utf-8 -*-
"""
Created on Fri Feb 21 16:00:48 2025

@author: Admin
"""

import FinanceDataReader as fdr

df = fdr.StockListing('KRX') # NASDAQ

# DataReader('종목코드', 시작일자, 종료일자)
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 애플 주식 데이터 수집
df = fdr.DataReader('AAPL','2022')

# 주식 가격 시각화
plt.figure(figsize=(10, 6))
sns.lineplot(x=df.index,y=df['Close'])
plt.title('Apple Stock Price in 2022')
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()

# 샘플링 : 시간간격
# 다운 샘플링: 샘플링 데이터 축소
# 업 샘플링
# 데이터들이 매일 기록 => 시간 구간으로 묶어서 기존 데이터를 집약!
# resample('BM') : B 영업일 / M 월
# 한달 간격으로 다운 샘플링
df_month = df.resample("BM").mean()

# 수익률 = (매도가격-매수가격)/매수가격 : pct_change()
df_month['rtn'] = df_month['Close'].pct_change()

# 수익률 시각화
plt.figure(figsize=(10, 6))
sns.lineplot(x=df_month.index,y=df_month['rtn'])
plt.title('Apple Stock Price in 2022-2023')
plt.xlabel('Date')
plt.ylabel('Returns')
plt.show()

# 주가 흐름 파악
# 이동평균선 : 과거 주식 가격의 흐름을 바탕으로
# 미래 주식 가격을 예측하는데 사용되는 선
# 일정 기간동안 주식 가격의 흐름을 평균내어 선들을 연결!
# 예) 5일 이동평균선 : 최근 5일간의 주가를 종가기준으로 합하여
#                     5로 나누어 평균을 구하는 것
# rolling()
# 한달 단위로 집약된 데이터
# rolling(2).mean() => 2달씩 종가에 대한 평균

df_month['MA'] = df_month['Close'].rolling(2).mean()

df_month.iloc[:, [3,7]].plot(figsize=(15,8))
plt.show()

# 최근 종가를 이용하여 이동평균선과 비교 => 상승/하락 판단
# 이동평균선 60일 전 증가
last_close = df_month['MA'].iloc[-2]

# 오늘 종가
price = df_month['Close'].iloc[-1]

if price > last_close:
    print('상승 장')
elif price < last_close:
    print('하락 장')
else:
    print('변화 없음')
    
df_month.to_csv('apple_data.csv')

# ------------------------------------------
# ARIMA
from statsmodels.tsa.arima.model import ARIMA
import statsmodels.api as sm

df = pd.read_csv('apple_data.csv')

model = ARIMA(df['Close'].values, order=(0,1,2))
model_fit = model.fit()
model_fit.summary()
'''
<class 'statsmodels.iolib.summary.Summary'>
"""
                               SARIMAX Results                                
==============================================================================
Dep. Variable:                      y   No. Observations:                   39
Model:                 ARIMA(0, 1, 2)   Log Likelihood                -142.421
Date:                Fri, 21 Feb 2025   AIC                            290.843
Time:                        16:48:08   BIC                            295.755
Sample:                             0   HQIC                           292.591
                                 - 39                                         
Covariance Type:                  opg                                         
==============================================================================
                 coef    std err          z      P>|z|      [0.025      0.975]
------------------------------------------------------------------------------
ma.L1          0.1693      0.159      1.062      0.288      -0.143       0.482
ma.L2         -0.0327      0.189     -0.173      0.863      -0.404       0.339
sigma2       105.3289     29.705      3.546      0.000      47.109     163.549
===================================================================================
Ljung-Box (L1) (Q):                   0.01   Jarque-Bera (JB):                 0.59
Prob(Q):                              0.92   Prob(JB):                         0.74
Heteroskedasticity (H):               1.64   Skew:                            -0.04
Prob(H) (two-sided):                  0.38   Kurtosis:                         2.39
===================================================================================

Warnings:
[1] Covariance matrix calculated using the outer product of gradients (complex-step).
"""
'''
fig = model_fit.plot.predict()























