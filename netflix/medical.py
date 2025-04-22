# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 09:16:31 2025

@author: Admin
넷플릭스 데이터 분석 프로젝트: netflix_titles.csv / netflix_logo.jpg

사용 라이브러리 : 
    넘파이 : 수치 해석
    판다스 : 데이터 분석, 전처리
    맷프롯립 / 시본 : 데이터 시각화
    워드크라우드 : 특정 텍스트 강조
    
데이터 분석 목표:
    데이터를 빠르게 파악하고,
    전처리를 수행한 후
    여러 인사이드 도출 *****
    
데이터 전처리
    데이터 결측치 처리
    피처 엔지니어링 => 파생변수 생성(특정 데이터 피처를 데이터 프레임 변수로)
    
데이터 시각화 : 요청 기업의 브랜드 색상을 사용
    1. 브랜드 색상
        데이터 시각화 하기 전에 색상을 미리 정해주는 것이 중요.
        색상을 데이터에 성격에 맞게 선택
        중요도에 따라 강조 방법을 계획. => 시각화 효과를 극대화 가능.
        
    2. 파이 차트
        데이터의 카테고리별 비율을 시각화에 효과적.
        비율을 쉽게 비교할 수 있고,
        각 카테고리의 상대적으로 중요성을 한눈에 파악 가능.
        넷플릭스에서 영화와 TV 쇼의 비율을 시각화
        
    3. 막대 그래프
    데이터의 항목 간의 비교를 명확하게 시각화하는데 유용.
    각 장르의 빈도를 막대 그래프로 시각화.
    => 넷플릭스에서 어떤 장르가 가장 많이 등장하는 지를 파악.
    4. 히트맵
        데이터의 밀도나 강도를 색상으로 시각화하여
        복잡한 데이터셋에서 패턴, 트렌드 파악 용이!!
        나이 그룹별로 국가별 콘텐츠 비율을 시각화
        => 각 국가가 어떤 나이 그룹을 타겟으로 하는 콘텐츠가 많은 지를 분석
        => 각 콘텐츠를 통해 국가별 시청츨을 이해
        
    5. 워드크라우드
        텍스트 데이터에서 빈도가 높은 단어를 시각적으로 강조.
        => 데이터의 주요 주제, 키워드를 한눈에 파악
        넷플릭스에서 콘텐츠 설명에서 자주 등장하는 단어들을 시각화.
        => 어떤 주제나, 키워드가 자주 나오는 지를
        => 콘텐츠의 주요 테마 파악.
        => 마켓팅, 콘텐츠 기획, 전략, 사용자 분석 등 유용한 인사이트 파악
"""

'''
1. 넷플릭스 데이터 파악
'''
''' 1-1 데이터 분석 라이브러리 '''
#  numpy를 np로
# pandas를 pd로
# matplotlib.pyplot를 plt로
# seaborn를 sns로
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

''' 1-2 csv 로드 '''

# data 폴더의 csv파일을 로드하여 netflix 변수에 저장
# csv 파일 읽기
netflix = pd.read_csv('./data/netflix_titles.csv')
netflix.head() # 확인

''' 1-3. 데이터 내용 확인 : 컬럼(변수) 확인 '''
# 데이터프레임.columns
netflix.columns
'''
Out[4]: 
Index(['show_id', 'type', 'title', 'director', 'cast', 'country', 'date_added',
       'release_year', 'rating', 'duration', 'listed_in', 'description'],
      dtype='object')
'''


'''
2. 넷플릭스 데이터셋 결측치 처리
'''

'''
넷플릭스 결측치 비율 확인하고 처리
일반적으로 
    결측치 비율이 5% 미만
    => 일부만 존재 : 삭제
    => 데이터 손실 최소화, 분석의 신뢰성 영향 X
    결측치 비율이 5% ~ 20%미만
    => 결측치 비중이 꽤 큰편이므로 대체 방향을 모색.
    => 평균 / 중간값 / 최빈값
    결측치 비율이 20% 이상
    => 열전체를 삭제
    => 데이터 손실이 커지기 때문에 신중한 판단이 필요!!!!
    => 특히, 데이터셋이 작거나, 해당 변수가 중요한 역할을 할 경우,
    => 모델 기반 대체나, 예측 모델을 통해 결측치 보완!!!
    * 결측치가 20% 이상일 경우
      변수에 대한 중요성, 분석목적, 데이터의 양을 종합적으로 고려!!!
      
      
결측치 : 데이터프레임.isna()
결측치 갯수 : 데이터프레임.isna().sum()
결측치 비율 : 데이터프레임.isna().sum() / len(데이터프레임) * 100
'''
netflix.info()

for i in netflix.columns :
    missingValueRate = netflix[i].isna().sum() / len(netflix) * 100
    
    if missingValueRate >0 :
        print("{} null rate: {}%".format(i,round(missingValueRate, 2)))
'''
director null rate: 29.91%  <= 제거(컬럼)
cast null rate: 9.37%       <= 대체(컬럼의 해당값)
country null rate: 9.44%    <= 대체(컬럼의 해당값) 
date_added null rate: 0.11% <= 제거(컬럼의 해당 행)
rating null rate: 0.05%     <= 제거(컬럼의 해당 행)
duration null rate: 0.03%   <= 제거(컬럼의 해당 행)
'''
''' 2-1. country의 결측치(9.44%) 대체: fillna('No Data') '''
netflix['country'] = netflix['country'].fillna('No Data')

netflix['country']
'''
0       United States
1        South Africa
2             No Data
3             No Data
4               India
     
8802    United States
8803          No Data
8804    United States
8805    United States
8806            India
Name: country, Length: 8807, dtype: object
'''
''' 2-2 director/cast 대체: replace(np.nan,'No data')'''

netflix['director'] = netflix['director'].replace(np.nan,'No Data')
'''
0       Kirsten Johnson
1               No Data
2       Julien Leclercq
3               No Data
4               No Data
      
8802      David Fincher
8803            No Data
8804    Ruben Fleischer
8805       Peter Hewitt
8806        Mozez Singh
Name: director, Length: 8807, dtype: object
'''
netflix['cast'] = netflix['cast'].replace(np.nan,'No Data')
'''
0                                                 No Data
1       Ama Qamata, Khosi Ngema, Gail Mabalane, Thaban...
2       Sami Bouajila, Tracy Gotoas, Samuel Jouy, Nabi...
3                                                 No Data
4       Mayur More, Jitendra Kumar, Ranjan Raj, Alam K...
                       
8802    Mark Ruffalo, Jake Gyllenhaal, Robert Downey J...
8803                                              No Data
8804    Jesse Eisenberg, Woody Harrelson, Emma Stone, ...
8805    Tim Allen, Courteney Cox, Chevy Chase, Kate Ma...
8806    Vicky Kaushal, Sarah-Jane Dias, Raaghav Chanan...
Name: cast, Length: 8807, dtype: object
'''
''' 2-3. data_added/rating/duration 결측치를 가진 행을 제거: dropna() '''
# dropna() = axis=0 - 0=행 / inplace=True -실제로 제거
netflix.dropna(axis=0,inplace=True)
netflix.info() # 확인
'''
<class 'pandas.core.frame.DataFrame'>
Index: 8790 entries, 0 to 8806
Data columns (total 12 columns):
 #   Column        Non-Null Count  Dtype 
---  ------        --------------  ----- 
 0   show_id       8790 non-null   object
 1   type          8790 non-null   object
 2   title         8790 non-null   object
 3   director      8790 non-null   object
 4   cast          8790 non-null   object
 5   country       8790 non-null   object
 6   date_added    8790 non-null   object
 7   release_year  8790 non-null   int64 
 8   rating        8790 non-null   object
 9   duration      8790 non-null   object
 10  listed_in     8790 non-null   object
 11  description   8790 non-null   object
dtypes: int64(1), object(11)
memory usage: 892.7+ KB
'''
''' 2-4. 결측치 갯수로 재확인 : isnull().sum() / .isna().sum()'''

netflix.isna().sum()
'''
show_id         0
type            0
title           0
director        0
cast            0
country         0
date_added      0
release_year    0
rating          0
duration        0
listed_in       0
description     0
dtype: int64
'''

'''
3. 넷플릭스 피처 엔지니어링

피처 엔지니어링: 데이터프레임의 기존 변수를 조합하거나, 
                새로운 변수를 만드는 것을 의미.
데이터 분석/머신러닝 모델을 학습시킬 때 매우 중요!!
현업 : 예측 모델이 데이터의 패턴을 잘 이해하고 학습할 수 있는 기준!!!

피처 엔지니어링을 명확하고 의미있는 피처를 만들어 사용하면,
모델의 결과를 쉽게 해석할 수 있다!!!

데이터의 다양한 측면을 고려하여 더 정확한 분석을 할 수 있다!!

'''
'''
 3-1 넷플릭스 시청 등급 변수
 넷플릭스 데이터프레임 rating 변수를 이용하여 age_group(시청등급)
 All(모든) / Older Kids(어린이) / Teens(청소년 초반) 
 / Adults(성인) / Young Adults(청소년 후반)
 '''
netflix['age_group'] = netflix['rating']

age_group_dic = {
    'G': 'All', # 전체
    'TV-G': 'All',
    'TV-Y': 'All',
    'PG': 'Older Kids', # 7세 이상
    'TV-Y7': 'Older Kids',
    'TV-Y7-FV': 'Older Kids',
    'TV-PG': 'Older Kids',
    'PG-13': 'Teens', # 13세이상
    'TV-14': 'Young Adults', #16세 이상
    'NC-17': 'Adults',  #등급 보류
    'NR': 'Adults',     #무삭제 등급
    'UR': 'Adults',
    'R': 'Adults',
    'TV-MA': 'Adults'}

netflix['age_group'] = netflix['age_group'].map(age_group_dic)

netflix.head(2)

# => 피처 엔지니어링을 통해 데이터의 가독성, 효율성을 높일 수 있다.

''' 3-2. 전처리가 완료된 데이터를 csv 파일로 저장 : to_csv() '''
# to_csv('파일명', index=False)
netflix.to_csv('./result/final.csv',index=False)

'''
4. 넷플릭스 시각화하기
전처리된 넷플릭스 파일 읽기

시각화 라이브러리

'''
''' 4-1. 시각화 라이브러리'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

''' 4-2. 전처리된 데이터 읽기'''
netflix = pd.read_csv('./result/final.csv')
netflix.info()

''' 4-3. 넷플릭스 브랜드 색상 시각화 : sns.palplot('색상','색상')'''
sns.palplot(['#221f1f','#b20710','#e50914','#f5f5f1'])
plt.title('Netflix brand palette', #제목명
          loc ='left',  # 정렬기준
          fontfamily='serif',   #글꼴
          fontsize=15,  #글씨 크기(pt단위)
          y=1.2)    #제목의 y축 위치값
plt.show()


''' 4-4. 넷플릭스 파이 차트: Movies & TV shows'''
# netflix['type'] => value_counts()
type_counts = netflix['type'].value_counts()
'''
type
Movie      6126
TV Show    2664
Name: count, dtype: int64
'''
plt.figure(figsize=(5,5))
plt.pie(type_counts,
        labels =type_counts.index,
        autopct='%0.f%%', #자동으로 %만들어줌
        startangle=100,
        explode=[0.05,0.05],
        shadow=True,
        colors=['#b20710','#221f1f']
        )
plt.suptitle('TV show vs Movie', fontfamily ='serif',fontsize=15,fontweight='bold')
plt.title("how many netflix content..?",fontfamily='serif',fontsize=12)
plt.show()

'''
4-5 넷플릭스 막대 그래프
어떤 장르가 인기가 가장 많은지
listed_in

'''
netflix.head(3)
netflix['listed_in']
'''
Out[47]: 
0                                           Documentaries
1         International TV Shows, TV Dramas, TV Mysteries
2       Crime TV Shows, International TV Shows, TV Act...
3                                  Docuseries, Reality TV
4       International TV Shows, Romantic TV Shows, TV ...
                       
8802                       Cult Movies, Dramas, Thrillers
8803               Kids' TV, Korean TV Shows, TV Comedies
8804                              Comedies, Horror Movies
8805                   Children & Family Movies, Comedies
8806       Dramas, International Movies, Music & Musicals
Name: listed_in, Length: 8790, dtype: object
'''

# 넷플릭스 데이터셋의 장르별 등장 횟수 계산(빈도 계산)
genres = netflix['listed_in'].str.split(', ',expand=True) \
                             .stack()                     \
                             .value_counts()
# expand=True : 분할된 결과를 확장하여 여러 열로 반환
# 분할된 문자가 개별적인 열로 배치되어 데이터프레임 생성

'''
Out[65]: 
International Movies            2752
Dramas                          2426
Comedies                        1674
International TV Shows          1349
Documentaries                    869
Action & Adventure               859
TV Dramas                        762
Independent Movies               756
Children & Family Movies         641
Romantic Movies                  616
Thrillers                        577
TV Comedies                      573
Crime TV Shows                   469
Kids' TV                         448
Docuseries                       394
Music & Musicals                 375
Romantic TV Shows                370
Horror Movies                    357
Stand-Up Comedy                  343
Reality TV                       255
British TV Shows                 252
Sci-Fi & Fantasy                 243
Sports Movies                    219
Anime Series                     174
Spanish-Language TV Shows        173
TV Action & Adventure            167
Korean TV Shows                  151
Classic Movies                   116
LGBTQ Movies                     102
TV Mysteries                      98
Science & Nature TV               92
TV Sci-Fi & Fantasy               83
TV Horror                         75
Anime Features                    71
Cult Movies                       71
Teen TV Shows                     69
Faith & Spirituality              65
TV Thrillers                      57
Stand-Up Comedy & Talk Shows      56
Movies                            53
Classic & Cult TV                 26
TV Shows                          16
Name: count, dtype: int64
'''
# 1단계 
# netflix['listed_in'].str.split(', ',expand=True)
# 2단계 
# netflix['listed_in'].str.split(', ',expand=True).stack()
# 3단계 
# netflix['listed_in'].str.split(', ',expand=True).stack().value_counts()

plt.figure(figsize=(12,6))

sns.barplot(x=genres.values,
            y=genres.index,
            hue=genres.index,
            palette='RdGy')
plt.title('what do you see on netflix..?',fontsize=16)
plt.xlabel('Count', fontsize=14)
plt.ylabel('Genre', fontsize=14)

plt.grid(axis='x')
plt.show()
'''
넷플릭스의 전체적인 콘텐츠 전략을 확인할 수 있다.
넷플릭스는 드라마와 국제 영화에 집중
국제 영화 드라마 코미디 국제 티비쇼까지 상당히 많은수가 나타나고
다큐멘터리부터 떨어지는것으로 추정

'''

'''
4-3 넷플릭스 히트맵
넷플릭스 데이터셋을 이용하여 각 나라의 콘텐츠 수를 집계
각 나라에서 어느 나이 그룹이 어떠 콘텐츠를 소비하는지 분석
특정 나이층의 시청 선호도를 파악하여 마케팅 전략을 세우고자..
특정 나라에서 특정 나이 그룹을 위한 컨텐츠가 부족하다면 
해당 연령층을 겨냥한 새로운 컨텐츠를 개발!
country / age_group / /
genres
'''
# 1. 넷플릭스 데이터의 title => 'Sankofa' 인 행의 데이터확인
netflix[ netflix['title'].str.contains('Sankofa',na=False,case=False)]
'''
Out[68]: 
  show_id   type  ...                                        description age_group
7      s8  Movie  ...  On a photo shoot in Ghana, an American model s...    Adults

[1 rows x 13 columns]
'''
# 2. 'country' 열의 값을 , 기준으로 구분 => List
# 출력할 최대 행 수를 None으로 설정해서 모두 출력
pd.set_option('display.max_rows', None)
# pd.set_option() : 판다스 라이브러리의 출력 옵션을 설정하는 함수
# display.max_rows , None : 전체 행을 생략없이 출력 가능



# 쉼표로 country 열의 값을 파이썬 리스트
netflix['country'] = netflix['country'].str.split(', ')

# 3. 파이썬 리스트로 바꾼 country 열의 값에 explode() 함수를 적용하여
#    개별 행으로 분리
netflix_age_country = netflix.explode('country')

# 4. 확인 : title열의 값이 'Sankofa'인 행 전체를 확인하여
     # country 열과 age_group 열의 값이 어떻게 이루어져 있는지 확인
netflix_age_country[netflix_age_country['title'].str.contains('Sankofa',na=False,case=False)]

# 5. 각 나이 그웁에 따른 국가별 넷플릭스 컨텐츠 수
netflix_age_country_unstack = netflix_age_country.groupby('age_group')\
                              ['country'].value_counts().unstack()
# unstack() : 그룹화돈 데이터들 ㄹㅇ서

# 6. 특정 나이 그룹에 따른 특정 나라별 콘텐츠로 필터링
# 6-1 연령
age_order = ['All','Older Kids','Teens','Adults']
# 6-2 국가
country_order = ['United States','India','United Kingdom','Canada',
                 'Japan','France','South Korea','Spain','Mexico','Turkey']

netflix_age_country_unstack = netflix_age_country_unstack.loc[age_order,country_order]
# 6-4 결측치 처리 : 0
netflix_age_country_unstack = netflix_age_country_unstack.fillna(0)

# 6-5 나이 그룹에 따른 국가별 콘텐츠 비율
#     각 열의 값을 각 열의 합으로 나누기: div()
#     axis = 1
netflix_age_country_unstack = netflix_age_country_unstack.div(
                              netflix_age_country_unstack.sum(axis=0),
                              axis=1)
# div(어떤 값을, 무엇으로) 나누기
# div(열의 합을, 각 열의 값으로) 나누기
# div(~~.sum(axis=0) ,axis=1)
# U.S => 열의 합 :  255(All) + 694(O.K) + ~~~~~
# => 데이터 정규화 : 단위가 큰수와 단위가 아주 작은 수=> 0~1 실수

plt.figure(figsize=(15,5))

# 사용자 정의 컬러맵
cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list('',
                    ['#221f1f','#b20710','#f5f5f1'])
# from_list('색팔레트이름',['색상','색상','색상', .....])
# #221f1f # : 뒤의 수가 16진수, 색상값일 경우

sns.heatmap(netflix_age_country_unstack,
            cmap = cmap
            ,linewidth=2.5,
            annot = True,
            fmt='.0%')
plt.suptitle('Total content by country', fontweight='bold',
             fontfamily='serif',fontsize=15)
plt.title('what', fontsize=12,fontfamily='serif')

plt.show()

# 워드 클라우드를 원하는 형태로 그리기 위해 그림을 불러오는 패키지
from PIL import Image

plt.figure(figsize=(15,5))

# netflix['description']
# wordcloud 에서 작동할 수 있도록 데이터 프레임을
# list 1차 변환시키고 ,str(문자열)로 2차 변환
text = str(list(netflix['description']))

# 로고 이미지 열고 : 'netflix_logo.jpg' => Image.open()
# 넘파이 배열로 변환: np.array()
mask = np.array(Image.open('./data/netflix_logo.jpg'))

# 워드 클라우드 색상맵
cmap = plt.matplotlib.colors.LinearSegmentedColormap.from_list('',
                                                               ['#221f1f','#b20710'])

# 워드클라우드 생성: WordCloud().generate(text)
# WordCloud()
# backgroud_color = 배경색
# width = px단위 #1400
# height = px단위 # 1400
# max_words = #노출빈도가 가장 큰 단어부터 최대 ? 단어만 출력
# mask =  #이미지를 이용한 경우에만, 넘파이 배열
# colormap = # 각 단어의 색상
from wordcloud import WordCloud

wordcloud = WordCloud(background_color = 'white',
                      width = 1400,height = 1400,
                      max_words= 170,
                      mask = mask,
                      colormap = cmap).generate(text)

plt.suptitle('Movies and TV Shows',
             fontweight='bold', fontfamily='serif',fontsize=15)

# 워드크라우드 결과를 Plots 창에 나타내기 : plt.imshow(워드크라우드 객체명)
plt.imshow(wordcloud)
plt.axis('off')
plt.show()












