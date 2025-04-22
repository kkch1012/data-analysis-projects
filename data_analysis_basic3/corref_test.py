# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 11:46:31 2025

@author: Admin
"""

'''
상관관계
변수: 독립변수 / 종속변수
변수 => 키, 몸무게, 연간 소득과 같이 변하는 양을 표현 한것

상관관계(correlation)
=> 두 개의 변수들이 함께 변화하는 관계

상관게수(correlation coefficient)
=> 변수들 사이의 상관관계의 정도를 나타내는 수치

상관관계가 있는 두 변수가 있을 경우,
한 값이 증가할 때 다른 값도 증가할 경우,
양의 상관관계가 있다.
반대의 경우 음의 상관관계가 있다.

**** 상관관계와 인과관계는 다르다

0에서 9까지의 정수를 가지고 있는 x와,
0부터 18까지 x 원소 값의 두배 값을 가지는 y1 사이의 상관관계

'''

import numpy as np

np.random.seed(85)
# 동일한 결과를 얻기위해 85라는 초기값 사용


x = np.arange(0,10)


y1 = x*2

np.corrcoef(x, y1)

'''
array([[1, 1.],
       [1, 1.]])
'''

x = np.arange(0,10)
y2 = x ** 3

y3 = np.random.randint(0,100, size=10)

np.corrcoef(x, y2)

np.corrcoef((x,y2,y3))



# array([[ 1.        ,  0.90843373, -0.06472465],
#        [ 0.90843373,  1.        , -0.17528014],
#        [-0.06472465, -0.17528014,  1.        ]])
'''
array ([[x-x,x-y2,x-y3]
        [y2-x,y2-y2,y2-y3
         [y3-x,y3-y2,y3-y3]]])
'''

result = np.corrcoef((x,y2,y3))

import matplotlib.pyplot as plt

# plt.imshow(result) result 이미지를 보여주는 함수
plt.imshow(result)
plt.colorbar()
plt.show()



'''
seaborn 라이브러리
1. 맷플롯립을 기반
2. 맷플롯립에 비하여 높은 수준의 인터페이스 제공

3. import seaborn
'''

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# set_theme() : style = 테마명

sns.set_theme(style="darkgrid")
#  load_dataset() : 기본 제공 데이터 로드
tips = sns.load_dataset("tips")

# relplot() : 산점도

sns.relplot(data=tips,x = "total_bill",y="tip")

sns.relplot(x ="total_bill",data=tips,y="tip")

sns.relplot()

sns.relplot(data=tips,
            x = "total_bill",y="tip",
            col="time",
            hue="smoker",
            style="smoker",
            size="size")


plt.savefig('test.png')
plt.show()

tips.head()
tips.tail()

# distplot(컬럼명)
# kde = True: 가우시안 커널 밀도 추정 => 선으로 그려보고싶을때
# bins = 숫자 : 구간을 늘리거나 축소하는

sns.distplot(tips['tip'],
             kde=True,
             bins=10)
plt.show()

# 실제 식사 요금 대비 팁
# tip_pct = 100 *  tip / total_bill


tips['tip_pct'] = (100*tips['tip'])/ tips['total_bill']
sns.relplot(data=tips,
            x="tip_pct",y="tip",
            col="time",
            hue="smoker",
            size="size")
sns.distplot(tips['tip_pct'],
             kde=True,
             bins=10)
plt.show()

# 산점도에 선형회귀선 시각화 : regplot()
# data = 데이터프레임
 # x = 컬럼
 # y = 컬럼
#  단, 변수에 저장
# ax = sns.regplot()
# ax.set_title(')
# ax.set_xlabel(')
# ax.set_ylabel(')
# => ax : axes 객체
ax = sns.regplot(data=tips, x='total_bill', y='tip')
ax.set_xlabel('Total Bill')
ax.set_ylabel('Tip')
ax.set_title('Total Bill and Tip')

'''
pairplot()
여러 변수들 사이에 관계
수치형 데이터를 가지고 있는 컬럼(변수) 간의 관계

두 변수 a와 b의 상관도가 높을 경우,
a가 증가할 경우 => b도 증가

두 변수 a와 b의 상관도가 낮을 경우
b는 랜덤한 분포 형태/ 특정 값으로 쏠리는 형태

전체 데이터프레임을 사용할 수 있다.
'''
sns.pairplot(tips)
plt.show()

'''
Anscombe's quartet 데이터셋
'''
anscombe = sns.load_dataset("anscombe")
anscombe.head()
'''
lmplot() : 석형 회귀 직선을 구하는 기능

x=
y=
data = anscombe.query('dataset == I')
  dataset     x     y
0       I  10.0  8.04
1       I   8.0  6.95
2       I  13.0  7.58
3       I   9.0  8.81
4       I  11.0  8.33
'''

sns.lmplot(x="x",y="y",
           data=anscombe.query("dataset == 'I'"),
           ci =None
           ,scatter_kws={"s":80})
plt.show()


sns.lmplot(x="x",y="y",
           data=anscombe.query("dataset == 'II'"),
           ci =None
           ,scatter_kws={"s":80})

plt.show()



sns.lmplot(x="x",y="y",
           data=anscombe.query("dataset == 'III'"),
           ci =None
           ,scatter_kws={"s":80})

plt.show()

'''
flight 데이터셋
'''
flights = sns.load_dataset('flights')
sns.relplot(data=flights,x="year",y="passengers")
plt.show()
'''
x 축의 값은 년도(year) 값이므로 이산적으로 표시,
y 축의 값은 승객의 수
매년 전 세계의 비행기 이용자는 꾸준히 증가하는 추세
'''
sns.relplot(data=flights,x="year",y="passengers",kind="line")
plt.show()























