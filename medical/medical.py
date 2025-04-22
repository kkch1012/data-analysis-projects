# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 09:17:56 2025

@author: Admin
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

heart = pd.read_csv('./data/heart.csv')
heart.head()

heart.columns()
heart.head(3)

heart.info()
'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 918 entries, 0 to 917
Data columns (total 10 columns):
 #   Column          Non-Null Count  Dtype  
---  ------          --------------  -----  
 0   Age             918 non-null    int64  
 1   Sex             918 non-null    object 
 2   ChestPainType   918 non-null    object 
 3   RestingBP       891 non-null    float64
 4   Cholesterol     918 non-null    int64  
 5   FastingBS       827 non-null    float64
 6   RestingECG      918 non-null    object 
 7   MaxHR           918 non-null    int64  
 8   ExerciseAngina  918 non-null    object 
 9   HeartDisease    916 non-null    float64
dtypes: float64(3), int64(3), object(4)
memory usage: 71.8+ KB
'''

heart['HeartDisease'].head(5)

heart['HeartDisease'] == 1

# 조건 : 심장병이 있는 사람의 데이터만 추출 (True, False 반환)
# 논리형 인덱싱 : 조건의 결과가 True 값의 행만 추출
# heart 데이터 안에 조건을 넣어주면 True 값인 행만 추출하여
# 심장병이 있는 사람의 데이터만 추출할 수 있음? <= H
H = heart[heart["HeartDisease"] == 1]
H

# 심부전 데이터셋 결측치 처리
# => 결측치는 결측치가 차지하는 비율에 따라 다르게!
# 5% 미만 / 5% ~ 20% / 20% 이상
# =>  Chat GPT : 
    
# 결측치 비율 확인하기

'''
1. for 반복물을 통해 각 컬럼별 결측치 비율을 계싼하여 문자열로 출력
=> for i in heart.columns: 각 열에 대해 반복

2. heart 데이터 열(column)의 결측치 개수의 합 sum
    결측 값은 True 반환,그 외에는 False 반환 ; isna()
=> heart[i].isna().sum()
'''

for i in heart.columns:
    heart_beat = heart[i].isna().sum() / len(heart) * 100
    if heart_beat> 0:
        print("{0} null rate: {1}%".format(i,round(heart_beat,2)))

'''
RestingBP null rate: 2.94%      <= 혈압        : 대체
FastingBS null rate: 9.91%      <= 공복 혈당   : 대체
HeartDisease null rate: 0.22%   <= 심장병 여부 : 결측치만 삭제
'''

# 대체 : fillna() , replace()
# 삭제 : dropna() / drop()
# FastingBS : 대체 (120mg/dl < ?? ==> 1 : 0)

heart['FastingBS'] = heart['FastingBS'].fillna(0)

heart['RestingBP'].median()
heart['RestingBP'].mean()

heart['RestingBP'] = heart['RestingBP'].replace(np.nan,heart['RestingBP'].median())

heart.dropna(axis=0, inplace = True)
heart.info()


'''
V. 심부전 데이터셋 통계처리:
    ------------------------
1. MaxHR 평균값과 중앙값
2. ChestPainType : 열의 빈도수
3. 심부전 데이터 셋 Age, MaxDB, choiiesterol 열의 주요 통계량 요약
4. 그룹별 집계
4-1. 심부전 데이터셋의 HeartDisease, ChestPainType 열로 그룹화 후,
    MaxHR, Cholesterol 열의 평균
4-2 Sex열로 그룹화한 후, RestingBP 열의 평균
'''
print(heart['MaxHR'].mean())
# 136.83078602620088
print(heart['MaxHR'].median())
# 138.0

'''
평균값과 중앙값이 비슷하다는 것
=> 데이터에 극단적인 이상치가 없음
=> MaxHR 변수의 값들이 큰 왜곡없이 고르게 분포
'''
# ChestPainType 열의 빈도수
heart['ChestPainType'].value_counts()
'''
ChestPainType
ASY    496  <= 무증상
NAP    202  <= 비협심증, 흉통
ATA    172  <= 비전형적인 협심증
TA      46  <= 전형적인 협심증
Name: count, dtype: int64
==> 무증상이 다른 유형에 비해 훨씬 빈번하게 발생한다.

'''
# 'Age', 'MaxHR','Cholesterol' 통계량 요약
heart[['Age', 'MaxHR','Cholesterol']].describe()

'''
              Age       MaxHR  Cholesterol
count  916.000000  916.000000   916.000000
mean    53.533843  136.830786   198.728166
std      9.425923   25.447917   109.466452
min     28.000000   60.000000     0.000000
25%     47.000000  120.000000   173.000000
50%     54.000000  138.000000   223.000000
75%     60.000000  156.000000   267.000000
max     77.000000  202.000000   603.000000

Age mean : 53.73 <= 대부분 50대 초반
Cholesterol std : 109 / mean : 196 <= 
MaxHR max : 202 / min : 60 <=

'''

# 그룹별 집계
# 1. HeartDisease
# 2. ChestPainType
# => Age, MaxHR, Cholesterol => mean()

G = heart.groupby(['HeartDisease','ChestPainType'])  \
    [['Age','MaxHR','Cholesterol']].mean()

G

'''
                                  Age       MaxHR  Cholesterol
HeartDisease ChestPainType                                    
0.0          ASY            52.317308  138.548077   226.865385
             ATA            48.236486  152.621622   232.668919
             NAP            51.045802  150.641221   221.503817
             TA             54.692308  150.500000   222.730769
1.0          ASY            55.660714  125.806122   175.974490
             ATA            55.958333  137.500000   233.291667
             NAP            57.549296  129.394366   153.281690
             TA             55.000000  144.500000   186.700000
=> 정상적인(심장병이 없는) 사람의 나이, 시맙ㄱ수, 콜레스테롤 수치가
   각각 어떠한 흉통 유형에 따라 달라지는 지를 확인 할 수 있다!
예) 정상적인(심장병이 없는) 사람의 TA 유형과 심장병이 있는 TA 유형
나이 54.69 / 55.00
심박수: 150.5 / 144.5
콜레스테롤 222.72 / 186.7
             '''
SG = heart.groupby('Sex')['RestingBP'].mean()
SG
'''
Sex
F    132.119792
M    132.421271
Name: RestingBP, dtype: float64
-> 성별에 따른 건강관리 차이 미미함 
=> 성별에 따른 건강관리보단 개인에 따른 건강관리가 더 중요할 수 있다.
'''

'''
참고 :
    연령대 그룹
    콜레스테롤 구간분류
    복합 위험도 : 고혈압 / 콜레스테롤 / 고혈당 => 동시에 가진 환자의 위험도
    운동 관련 변수 : 운동시 심장 스트레스의 영향 : 이진 변수<= astype(int)
    심전도 결과 : 이진변환 => 심전도 결과와 심전도 발생 위험
    최대 심박수의 비율 계산 => 심장 운동능력 측정(나이)
'''


# 심부전 파이 차트: 'ChestPainType' 변수: 흉통 유형
# ratio : 'ChestPainType' 추출 <= 1차원으로 변경

# plt.pie 의 매개변수 설명
# labels : 부채꼴 조각 이름
# autopct : 부채꼴 안에 표시될 숫자 형식 지정
# 문자열에서 % 포맷팅으로 %0.f 형태로 사용하면 소수점 없이 정수
# 진짜 %를 표시하기 위해 %%로
# startangle : 부채꼴이 그려지는 시작 각도 설정, 90이면 12시 방향
# explode : 부채꼴이 파이 플롯의 중심에서 벗어나는 정도
# shadow : 그림자 효과

# plt.suptitle() : 전체 플롯의 제목
# plt.title() : 서브 플롯의 제목
# plt.show()

ratio = heart['ChestPainType'].value_counts()

'''
ChestPainType
ASY    496
NAP    202
ATA    172
TA      46
Name: count, dtype: int64
'''

plt.pie(x=ratio,
         labels=ratio.index,
         autopct='%0.f%%',
         startangle = 100,
         explode=[0.05,0.05,0.05,0.05],
         shadow=True,
         colors=['#003399','#0099ff','#00FFFF','#CCFFFF'])
plt.suptitle('Chest Pain Type')
plt.title("ASY, NAP, ATA, TA.")
plt.show()
'''
ASY(무증상) : 심장병 여부를 판단할 때 흉통 유형이 중요하다
      무증상 흉통을 고려!!
NAP, ATA(비전형적) :다양한 흉통 유형이 심잠병의 가능성이 높다
     심장병 판단 시, 흉통 유형을 세밀하게 분석해야 한다.
TA: 전형적인 흉통 증상이 드물다!
     즉, 다른 유형의 흉통이 더 흔할 수 있다. 
     
     
ChestPainType 변수 : 흉통 유형

plt.figure(figsize=(12,5))
countplot() : 각 범주(심장병 여부)에 속하는
 데이터(흉통 유형) 의 개수들 막대 시각화
countplot() 변수
data : countplot 에서 사용할 데이터 셋
x : HeartDisease
hue : 특정 열 데이터로 색상을 구분 ChestPainType
hue_order : 색상 순서를 수동 => "ASY", "NAP", "ATA", "TA"
'''
# x축 눈금 설정 : HeartDisease
# plt.xticks([0,1], ['~~', '~~~'])

# plt.tight_layout() : 겹치지않도록 최소한의 여백을 만들어주는 역할

plt.figure(figsize=(12, 5))

sns.countplot(data = heart,
              x='HeartDisease',
              hue='ChestPainType',
              hue_order=["ASY","NAP","ATA","TA"],
              palette=['#003399','#0099FF','#00FFFF','#CCFFFF']
    )
plt.suptitle('Chest Pain Types / Heart')
plt.title('ASY NAP ATA TA')

plt.xticks([0,1],['without nothing heart','heart disease'])
plt.tight_layout()
plt.show()


'''
심장병 O : ASY NAP ATA TA
=> 무증상 : 심장 질환과 흉통 유형 간의 강한 관련성
=> 심장병이 심각한 단계에 이를 때까지 증상이 나타나지 않을 수 있다!
=> 무증상 환자들에 대한 모니터링 강화.
=> 조기 검진을 통해 조기 발견이 중요!!

심장병 X : ATA NAP ASY TA
=> 비전형적인 흉통
=> 비환자들에게도 심장 건강에 대한 경각심!!!
=> 비환자들에게도 심장 검진이 필요!!
'''

# 심부전 데이터 영역 그래프
# 나이에 따른 심부전 여부 수치화
# groupby('Age)['HeartDisease'].value_counts()
# unstack()
Heart_Age = heart.groupby('Age')['HeartDisease'].value_counts().unstack(level='HeartDisease')

plt.figure(figsize=(12,5))

# plt.fill_between() : x축을 기준으로 그래프 영역을 채우는 함수
# x : 곡선을 정의하는 노드의 x좌표
# y1: 첫번째 곡선을 정의하는 노드의 y 좌표
# y2: 두번째 곡선을 정의하는 노드의 y좌표
# alpha : 투명도
# label = 범례에 표시할 문자열 입력
plt.fill_between(x =Heart_Age[0].index,
                 y1 =0,
                 y2 = Heart_Age[0],
                 color='#003399',
                 alpha =1,
                 label = 'Normal')
plt.show()

# x = Heart_Age[0].index : Heart_Age 데이터의 심장병에 없는 환자들의 나이
# y1 = 0 : y좌표 0부터 그래프 영역을 채워야해서 y1 = 0으로 설정
# y2 = Heart_Age[0] : Heart_Age 데이터의 심장병에 없는 환자들의 환자들의 나이별 숫자

# 심장병이 있는 환자들의 나이별 숫자 시각화

plt.fill_between(x =Heart_Age[1].index,
                 y1 =0,
                 y2 = Heart_Age[1],
                 color='#003399',
                 alpha =0.6,
                 label = 'Heart Disease')
plt.legend()
plt.xlabel('Age')
plt.ylabel('Count')
plt.suptitle('legend never die')
plt.title('Heart')
plt.show()

# 심부전 범주형 산점도 그래프 : sns.swarmplot()
# H_0 : 심장병이 없는 환자의 데이터 추출 HeartDisease == 0
# H_1 : 심장병이 없는 환자의 데이터 추출 HeartDisease == 1

H_0 = heart[heart['HeartDisease']==0]
H_1 = heart[heart['HeartDisease']==1]

# 그래프 객체 생성(figure에 2개의 서브 플롯을 생성)
fig = plt.figure(figsize=(15,5))
# axis1 = fig.add_subplot(1,2,1) 1행2열의 1
# axis2 = fig.add_subplot(1,2,2) 1행2열의 2
ax1 = fig.add_subplot(1,2,1)
ax2 = fig.add_subplot(1,2,2)

sns.swarmplot(x='RestingECG',
              y='Age',
              data=H_0,
              ax=ax1,
              hue='ExerciseAngina',
              palette=['#003399','#0099FF'],
              hue_order=['Y','N'])

# 심장병이 있는 환자의 데이터에서
# 나이별 안정된 상태에서 측정된 혈압 수치 시각화

sns.swarmplot(x='RestingECG',
              y='Age',
              data=H_1,
              ax=ax2,
              hue='ExerciseAngina',
              palette=['#003399','#0099FF'],
              hue_order=['Y','N'])

plt.suptitle('two graph')
ax1.set_title("Without heart disease")
ax2.set_title("with heart disease")
plt.show()


# 심부전 워드 클라우드
# pubmed 사이트의 심부전관련 논문의 제목 데이터 pubmed_title.csv

pubmed_title = pd.read_csv('./data/pubmed_title.csv')

from wordcloud import WordCloud
from PIL import Image

# text : 데이터 프레임을 list로 1차 변환시키고 str(문자열)로 2차 변환
# mask : 단어를 그릴 위치 설정
# cmap : 컬러맵 생성

text = str(list(pubmed_title['Title']))
mask = np.array(Image.open('./data/image.jpg'))
cmap =  plt.matplotlib.colors.LinearSegmentedColormap.from_list('',
                                                               ['#000666','#003399','#00FFFF'])

wordcloud = WordCloud(background_color = 'white',
                      width = 2500,height = 1400,
                      max_words= 170,
                      mask = mask,
                      colormap = cmap).generate(text)
plt.imshow(wordcloud)
plt.axis("off")
plt.suptitle('Heart Disease wordcloud')
plt.title('Pubmed site: Heart Failure')
plt.show()































