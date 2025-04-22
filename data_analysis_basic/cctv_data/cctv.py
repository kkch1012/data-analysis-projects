# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 17:07:35 2021
@author: Playdata


분석을 위해 사용되는 문법 및 사용 모듈
Python 기본 문법을 확인
Pandas와 Matplotlib의 기본적 사용법을 확인

분석 내용

국감브리핑 강남3구의 주민들이 
자신들이 거주하는 구의 체감 안전도를 높게 생각한다는 기사를 확인
http://news1.kr/articles/?1911504    

1. 서울시 각 구별 CCTV수를 파악하고, 
2. 인구대비 CCTV 비율을 파악해서 순위 비교
3. 인구대비 CCTV의 평균치를 확인하고 그로부터 CCTV가 과하게 부족한 구를 확인
4. 단순한 그래프 표현에서 한 단계 더 나아가 경향을 확인하고 시각화하는 기초 확인

"""
####################################################
### ------ 서울시 구별 CCTV 현황 분석하기  ----- ###
####################################################
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt
from matplotlib import font_manager, rc
#파이썬이실행되고 있는 운영체제 관련 모듈
import platform
# --------------------------------------------------#
# =========  분석작업을 위한 1차 전처리 =========== #
# --------------------------------------------------#


### 1. 엑셀파일 읽기 - 서울시 CCTV 현황
seoul_cctv = pd.read_csv("./data1/01. CCTV_in_Seoul.csv")

seoul_cctv.columns
seoul_cctv.columns[0]

seoul_cctv.head()

# 컬럼명 변경:
    '''
    DataFrame.rename(columns ={DataFrame.columns[컬럼index] : 변경컬럼명},
                     inplace=True)
    inplace=True: 변경사항을 바로 데이터프레임에 적용
    '''
seoul_cctv.rename(columns ={seoul_cctv.columns[0] : '구별'},
                 inplace=True)
seoul_cctv.head()

### 2. 엑셀파일 읽기 - 서울시 인구현황
pop_seoul = pd.read_excel("./data1/01. population_in_Seoul.xls",
                          header=2,
                          usecols="B, D, G, J, N")
pop_seoul.head()
#컬럼명 변경
pop_seoul.rename(columns={pop_seoul.columns[0]:'구별',
                          pop_seoul.columns[1]:'인구수',
                          pop_seoul.columns[2]:'한국인',
                          pop_seoul.columns[3]:'외국인',
                          pop_seoul.columns[4]:'고령자'}
                 ,inplace=True)
pop_seoul.columns

# 분석에 필요한 부분만 추출하여 읽기
# header = 엑셀 행(index)번호
# usecols="엑셀컬럼명, 엑셀컬럼명, ..."
# => B : 구이름, D: 인구수, G: 한국인, J: 외국인, N : 고령자(65세이상)




# --------------------------------------------------#
# ==== 분석작업을 위한 2차 전처리 : 데이터파악 ==== #
# --------------------------------------------------#

### 3. CCTV 데이터 파악하기
seoul_cctv.head()

seoul_cctv['최근증가율'] = (seoul_cctv['2014년']+seoul_cctv['2015년']
                       +seoul_cctv['2016년'])/seoul_cctv['2013년도 이전']*100
seoul_cctv.sort_values(by='최근증가율',ascending=False).head()
### 4. 서울시 인구 데이터 파악하기
# 합계에 해당하는 행 삭제
pop_seoul.head()
pop_seoul.drop([0],inplace=True)

# '구별' 컬럼의 유일한 값 확인
pop_seoul['구별'].unique()
'''
array(['종로구', '중구', '용산구', '성동구', '광진구', '동대문구', '중랑구', '성북구', '강북구',
       '도봉구', '노원구', '은평구', '서대문구', '마포구', '양천구', '강서구', '구로구', '금천구',
       '영등포구', '동작구', '관악구', '서초구', '강남구', '송파구', '강동구', nan],
      dtype=object)
'''
# 구별 컬럼에 결측치 여부 확인
pop_seoul[pop_seoul['구별'].isnull()]    
'''
 구별  인구수  한국인  외국인  고령자
26  NaN  NaN  NaN  NaN  NaN
'''
# 결측치 행 삭제
pop_seoul.drop([26],inplace=True)
pop_seoul.tail()

## '외국인 비율' '고령자 비율' 컬럼 추가
# '외국인 비율' = '외국인' / '인구수' * 100
pop_seoul['외국인비율'] = pop_seoul['외국인'] / pop_seoul['인구수'] * 100

pop_seoul['고령자비율'] = pop_seoul['고령자'] / pop_seoul['인구수'] * 100

pop_seoul.head()


pop_seoul.sort_values(by="인구수",ascending=False).head(5)
pop_seoul.sort_values(by="외국인",ascending=False).head(5)
pop_seoul.sort_values(by="외국인비율",ascending=False).head(5)
pop_seoul.sort_values(by="고령자",ascending=False).head(5)
pop_seoul.sort_values(by="고령자비율",ascending=False).head(5)





# --------------------------------------------------#
# =================== 분석작업 ==================== #
# --------------------------------------------------#

### 5. CCTV 데이터와 인구 데이터 합치고 분석하기
# Pandas.merge(seoul_cctv,pop_seoul,on='구별')
data_result = pd.merge(seoul_cctv,pop_seoul,on='구별')
data_result.head()
data_result.tail()
'''
만약, 두개의 데이터 프레임에 고통 컬럼명이 없을 경우,
left_on = 컬럼명, right_on = 컬럼명
'''
data_result.columns
'''
Index(['구별', '소계', '2013년도 이전', '2014년', 
       '2015년', '2016년', '최근증가율', '인구수',
       '한국인', '외국인', '고령자', 
       '외국인비율', '고령자비율'],
      dtype='object')
    '''
# 컬럼 제거: del DataFrame['컬럼명]
del data_result['2013년도 이전']
del data_result['2014년']
del data_result['2015년']
del data_result['2016년']
data_result.head()

# 인덱스를 구별로 바꿔치기
# '구별' 컬럼의 데이터를 index값으로 설정
data_result.set_index('구별',inplace=True)

# 고령자비율과 소계간의 상관관게 확인
np.corrcoef(data_result['고령자비율'],data_result['소계'])
# 인구수와 소게간의 상관관게 확인
np.corrcoef(data_result['인구수'],data_result['소계'])
'''

array([[ 1.        , -0.28078554],
       [-0.28078554,  1.        ]])



array([[1.        , 0.30634228],
       [0.30634228, 1.        ]])
'''



### 6. matplotlib를 이용하여 CCTV와 인구현황 그래프로 분석
# 한글 깨짐 방지를 위한 설정
plt.rcParams['axes.unicode_minus'] = False
# 운영 체제에 맞느 기본 폰트 설정
if platform.system() == 'Darwin':
    rc('font',family='AppleGothic')
elif platform.system() == 'Windows':
    path = 'c:/Windows/Fonts/malgun.ttf'
    font_name = font_manager.FontProperties(fname=path).get_name()
    rc('font',family=font_name)
else:
    print("~~~ sorry")
    
plt.figure()
data_result['소계'].plot(kind='barh',
                       grid=True,
                       figsize=(10,10))
plt.show()
    

    data_result['소계'].sort_values().plot(kind='barh',
                           grid=True,
                           figsize=(10,10))
    plt.show()
    
    
data_result['CCTV비율'] = data_result['소계']/data_result['인구수']* 100

data_result['CCTV비율'].sort_values().plot(kind='barh',
                       grid=True,
                       figsize=(10,10))
plt.show()
    
plt.figure(figsize=(6,6))
plt.scatter(data_result['인구수'], data_result['소계'],s=50)
plt.xlabel('인구수')
plt.ylabel('CCTV')
plt.grid()
plt.show()
    
plt.figure(figsize=(6,6))
plt.scatter(data_result['인구수'], data_result['CCTV비율'],s=100)
plt.xlabel('인구수')
plt.ylabel('CCTV')
plt.grid()
plt.show()
    
    
    
fp1 = np.polyfit(data_result['인구수'],data_result['소계'],1)
f1 = np.poly1d(fp1)
fx = np.linspace(100000,700000,100)

plt.figure(figsize=(10,10))
plt.scatter(data_result['인구수'],data_result['소계'],s=50)
plt.plot(fx,f1(fx),ls ='dashed',lw =3,color='r')  
# ls => line Style  
# lw => line width

plt.xlabel('인구수')
plt.ylabel('CCTV')
plt.grid()
plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
### 7. 보다 설득력 있는 자료 작업


fp1 = np.polyfit(data_result['인구수']
                 ,data_result['소계'],1)
f1 = np.poly1d(fp1)
fx = np.linspace(100000,700000,100)

# 오차 구하기

data_result['오차'] = np.abs(data_result['소계'] - f1(data_result['인구수']))

df_sort = data_result.sort_values(by='오차',ascending=False)

plt.figure(figsize=(14,10))
plt.scatter(df_sort['인구수'],df_sort['소계'],
            c = data_result['오차'],
            s=50)

plt.plot(fx,f1(fx), ls='dashed',lw=3,color='r')

for n in range(10):
    plt.text(df_sort['인구수'][n]*1.02,
             df_sort['소계'][n]*0.98,
             df_sort.index[n],
             fontsize=15)

plt.xlabel('인구수')
plt.ylabel('인구당비율')
plt.colorbar()
plt.grid()
plt.show()


