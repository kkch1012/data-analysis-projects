# -*- coding: utf-8 -*-
"""
Created on Wed Mar 12 09:14:52 2025

@author: Admin
"""

import numpy as np
from numpy import dot
from numpy.linalg import norm

def cos_sim(A, B):
 return dot(A, B)/(norm(A)*norm(B))

doc1 = np.array([0,1,1,1])
doc2 = np.array([1,0,1,1])
doc3 = np.array([2,0,2,2])

print('문서 1과 문서2의 유사도 :',cos_sim(doc1, doc2))
print('문서 1과 문서3의 유사도 :',cos_sim(doc1, doc3))
print('문서 2와 문서3의 유사도 :',cos_sim(doc2, doc3))

'''
문서 1: 저는 사과 좋아요
문서 2: 저는 바나나 좋아요
문서 3: 저는 바나나 좋아요 저는 바나나 좋아요

띄어쓰기 기준 토큰화를 진행했다고 가정.

from numpy import dot
dot 함수는 두 배열
'''

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = pd.read_csv('./movies_metadata.csv', low_memory=False)
data.head(2)

## 코사인 유사도에 사용할 데이터
## 영화 제목에 해당하는 title / 줄거리에 해당하는 overview
# => 좋아하는 영화를 입력 => 해당 영화의 줄거리와 유사한 줄거리의 영화

# 훈련 데이터 : 20000
data = data.head(20000)

# TF - IDF를 연산할 때 데이터에 Null 값이 들어있으면 에러가 발생
# data의 overview 열에 결측값에 해당하는 Null 값이 있는지 확인

print('overview 열의 결측값의 수:', data['overview'].isnull().sum())

## 결측값을 가진 행을 제거하는 pandas의 dropna()
## 결측값이 있던 행에 특정값으로 채워넣는 pandas의 fillna()
# => 빈 값(empty value)으로 대체 : ''

data['overview'] = data['overview'].fillna('')

## overview 열에 대해서 TF-IDF 행렬
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(data['overview'])

## 행렬의 크기
print('TF-IDF 행렬의 크기(shape): ',tfidf_matrix.shape)
# TF-IDF 행렬의 크기(shape):  (20000, 47487)
# 20,000의 행을 가지고 47847의 열을 가지는 행렬
# 20,000개의 영화를 표현하기 위해서 총 47,487개의 단어가 사용
# 47,847차원의 문서 벡터가 20,000개가 존재

## 20,000개의 문서 벡터에 대해서 상호 간의 코사인 유사도
# => consine_similarity(tfidf_matrix, tfidf_matrix)
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

print('코사딘 유사도 연산 결과:', cosine_sim.shape)
# 코사딘 유사도 연산 결과: (20000, 20000)
# 20,000개의 각 문서 벡터(영화 줄거리 벡터)와
# 자기 자신을 포함한 20,000개의 문서 벡터 간의 유사도가 기록된 행렬.

'''
기존 데이터프레임으로부터
영화의 타이틀을 key,
영화의 인덱스를 value로 하는 딕셔너리 title_to_index를 만들기

'''

title_to_index = dict(zip(data['title'], data.index))
# 영화 제목 Father of the Bride Part II의 인덱스를 리턴
idx = title_to_index['Father of the Bride Part II']
print(idx)

'''
선택한 영화의 제목을 입력하면 : title_to_index
코사인 유사도를 통해: cosine_sim
가장 overview가 유사한 10개의 영화를 찾아내는 함수
'''
def get_recommendations(title, cosine_sim=cosine_sim):
    # 선택한 영화의 타이틀로부터 해당 영화의 인덱스
    idx = title_to_index[title]
    
    # 해당 영화와 모든 영화와의 유사도
    sim_scores = list(enumerate(cosine_sim[idx]))
    
    # 유사도에 따라 영화들을 정렬
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    # 가장 유사한 10개의 영화
    sim_scores = sim_scores[1:11]
    
    # 가장 유사한 10개의 영화의 인덱스
    movie_indices = [idx[0] for idx in sim_scores]
    
    # 가장 유사한 10개의 영화의 제목을 리턴
    return data['title'].iloc[movie_indices]

get_recommendations('The Dark Knight Rises')

'''
Out[32]: 
12481                            The Dark Knight
150                               Batman Forever
1328                              Batman Returns
15511                 Batman: Under the Red Hood
585                                       Batman
9230          Batman Beyond: Return of the Joker
18035                           Batman: Year One
19792    Batman: The Dark Knight Returns, Part 1
3095                Batman: Mask of the Phantasm
10122                              Batman Begins
Name: title, dtype: object
'''

# 유클리드 거리
import numpy as np
def dist(x,y):
 return np.sqrt(np.sum((x-y)**2))

doc1 = np.array((2,3,0,1))
doc2 = np.array((1,2,3,1))
doc3 = np.array((2,1,2,2))
docQ = np.array((1,1,0,1))
print('문서1과 문서Q의 거리 :',dist(doc1,docQ))
print('문서2과 문서Q의 거리 :',dist(doc2,docQ))
print('문서3과 문서Q의 거리 :',dist(doc3,docQ))

# 유클리드 거리의 값이 가장 작다는 것은 문서 간 거리가 가장 가깝다는 것을 의미.
# 즉, 문서1이 문서Q와 가장 유사하다고 볼 수 있다.

doc1 = "apple banana everyone like likey watch card holder"
doc2 = "apple banana coupon passport love you"
# 토큰화
tokenized_doc1 = doc1.split()
tokenized_doc2 = doc2.split()
print('문서1 :',tokenized_doc1)
print('문서2 :',tokenized_doc2)
'''
문서1 : ['apple', 'banana', 'everyone', 'like', 'likey', 'watch', 'card', 'holder']
문서2 : ['apple', 'banana', 'coupon', 'passport', 'love', 'you']
'''
union = set(tokenized_doc1).union(set(tokenized_doc2))
print('문서1과 문서2의 합집합 :',union)
'''
문서1과 문서2의 합집합 : {'banana', 'coupon', 'watch', 'holder', 
'like', 'likey', 'you', 'apple', 'everyone', 'passport', 'love', 'card'}
'''
intersection = set(tokenized_doc1).intersection(set(tokenized_doc2))
print('문서1과 문서2의 교집합 :',intersection)
# 문서1과 문서2의 교집합 : {'banana', 'apple'}

print('자카드 유사도 :',len(intersection)/len(union))
# 자카드 유사도 : 0.16666666666666666




