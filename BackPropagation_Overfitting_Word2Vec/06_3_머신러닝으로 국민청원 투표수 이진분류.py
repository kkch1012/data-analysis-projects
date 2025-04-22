# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 09:19:44 2025

@author: Admin
"""

import os
import platform
import re
import pandas as pd
import numpy as np 

base_path = "data"
file_name = "petition.csv"


petitions = pd.read_csv(f"{base_path}/petition.csv",
                 index_col="article_id",
                 parse_dates=['start','end'])

petitions.shape
# Out[62]: (395547, 7)
petitions.describe()



#                                start  ...         votes
# count                         395547  ...  3.955470e+05
# mean   2018-05-26 13:51:50.767620352  ...  1.501188e+02
# min              2017-08-19 00:00:00  ...  0.000000e+00
# 25%              2018-01-26 00:00:00  ...  2.000000e+00
# 50%              2018-05-30 00:00:00  ...  5.000000e+00
# 75%              2018-09-20 00:00:00  ...  1.500000e+01
# max              2019-02-04 00:00:00  ...  1.192049e+06
# std                              NaN  ...  4.802583e+03


petition_remove_outlier = petitions.loc[(petitions['votes']>500) & (petitions['votes'] < 200000)]

petition_remove_outlier.shape
# Out[65]: (5308, 7)

df = petition_remove_outlier.copy()
df.describe()
'''
Out[69]: 
                               start  ...          votes
count                           5308  ...    5308.000000
mean   2018-06-05 11:19:34.679728640  ...    5664.046345
min              2017-08-19 00:00:00  ...     501.000000
25%              2018-02-23 00:00:00  ...     772.000000
50%              2018-06-01 00:00:00  ...    1452.000000
75%              2018-09-27 00:00:00  ...    3783.500000
max              2019-02-04 00:00:00  ...  197343.000000
std                              NaN  ...   14361.883793

'''

df.loc[df['answered'] == 1].shape
# Out[70]: (0, 7)
# 20만 건 이상 투표한 데이터는 모두 제외했으므로, votes 에서 답변 대상인 건은 0 

import matplotlib.pyplot as plt
df['votes'].plot.hist()
plt.show()


# 기본값을 0으로 설정
df['votes_pas_neg'] = 0 

# 평균 투표수
votes_mean = df['votes'].mean()

# 투표수가 평균을 넘으면 1로
df['votes_pas_neg'] = (df['votes'] > votes_mean) == 1

df['votes_pas_neg'].dtypes

# 타입을 boolean 에서 int 로 변경
df['votes_pas_neg'] =  df['votes_pas_neg'].astype(int)

df[['votes','votes_pas_neg']].head()
'''
            votes  votes_pas_neg
article_id                      
28           2137              0
34            679              0
43          11293              1
46           1933              0
50           1251              0
'''
sample_index = 28
sample_title = df.loc[sample_index, 'title']
sample_content = petitions.loc[sample_index, 'content']

def preprocessing(text):
    # 개별 문자 제거
    text = re.sub('\\\\n', ' ',text)
    
    # 특수문자 이모티콘 제거
    # text = re.sub('[?.,;:|\)*~`’!^\-_+<>@\#$%&-=#}※]', '',text)
    
    # 한글, 영문, 숫자만 남기고 모두 제거
    # text = re.sub('[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z0-9]',' ',text)
    # 한글, 영문만 남기고 모두 제거
    text = re.sub('[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z]',' ',text)
    #중복으로 생성된 공백값을 제거
    text = re.sub(' +', ' ',text)
    
    return text


def remove_stopwords(text):
    tokens = text.split(' ')
    
    stops = ['하지만','수','현','있는',
     '그리고',
     '그런데',
     '저는',
     '제가',
     '그럼',
     '이런',
     '저런',
     '합니다',
     '많은',
     '많이',
     '정말',
     '너무']
    
    meaningful_words = [w for w in tokens if not w in stops]
    
    return ' '.join(meaningful_words)


# 샘플 데이터를 적용
pre_sample_content = preprocessing(sample_content)
pre_sample_content = remove_stopwords(pre_sample_content)

df['content_preprocessing'] = df['content'].apply(preprocessing)
df['content_preprocessed'] = df['content_preprocessing'].apply(remove_stopwords)

### 학습 데이터와 테스트 세트를 7:3 비율
df.shape # Out[100]: (5308, 10)

split_count = int(df.shape[0] * 0.7)

df_train = df[:split_count].copy()
df_train.shape # Out[103]: (3715, 10)


df_test = df[split_count:].copy()
df_test.shape # Out[105]: (1593, 10)

from sklearn.feature_extraction.text import CountVectorizer

vectorizer = CountVectorizer(analyzer='word',
                             tokenizer= None,
                             preprocessor=None,
                             stop_words=None,
                             min_df=2,
                             ngram_range=(1,3),
                             max_features = 2000)


train_feature_vector = vectorizer.fit_transform(df_train['content_preprocessed'])
train_feature_vector.shape # Out[110]: (3715, 2000)

test_feature_vector = vectorizer.transform(df_test['content_preprocessed'])
test_feature_vector.shape #Out[113]: (1593, 2000)


vocab = vectorizer.get_feature_names_out()
print(len(vocab))
vocab[:10]
'''
Out[116]: 
array(['aid', 'and', 'article', 'articleview', 'articleview html',
       'articleview html idxno', 'be', 'cctv를', 'co', 'co kr'],
      dtype=object)
'''
dist = np.sum(train_feature_vector, axis=0)

pd.DataFrame(dist,columns=vocab)
'''
   aid  and  article  articleview  articleview html  ...  힘들어  힘듭니다  힘없는   힘을   힘이
0  138  124       83          111                87  ...   64    62   58  124  111

[1 rows x 2000 columns]
'''


from sklearn.feature_extraction.text import TfidfTransformer

transformer = TfidfTransformer(smooth_idf=False)
'''
smooth_idf=False
 True일 때는 피처를 만들 때 0으로 나오는 항목에 대해
 작은 값을 더해서 피처를 만들고
 False일 때는 더하지 않음
 
TF-IDF 가중치를  
 
 
 
 
 
 
'''


train_feature_tfidf = transformer.fit_transform(train_feature_vector)
train_feature_tfidf.shape

test_feature_tfidf = transformer.transform(test_feature_vector)
test_feature_tfidf.shape



from lightgbm import LGBMClassifier

model = LGBMClassifier(random_state=42, n_jobs=1)

y_label = df_train['votes_pas_neg']
model = model.fit(train_feature_tfidf,y_label)

### 평

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

k_fold = KFold(n_splits=5,
               shuffle=True,
               random_state=42)

scoring = 'accuracy'
score = cross_val_score(model,
                        train_feature_tfidf,
                        y_label,
                        cv=k_fold,
                        n_jobs=1,
                        scoring=scoring
                        )

score
'''
Out[134]: array([0.78600269, 0.80888291, 0.81965007, 0.78600269, 0.79407806])
'''


round(np.mean(score)*100,2)
# Out[135]: 79.89
y_pred = model.predict(test_feature_tfidf)
y_pred[:10]
# Out[137]: array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])

output = pd.DataFrame(data={'votes_pos_neg_pred': y_pred})
output.head()
'''
   votes_pos_neg_pred
0                   0
1                   1
2                   0
3                   0
4                   0
'''
# 0과 1이 어떻게 집계 되었는지 확인
output['votes_pos_neg_pred'].value_counts()
'''
votes_pos_neg_pred
0    1550
1      43
Name: count, dtype: int64
'''
df_test.columns
df_test['votes_pas_neg_pred'] = y_pred


df_test['pred_diff'] = np.abs(df_test['votes_pas_neg'] - df_test['votes_pas_neg_pred'])

df_test[['title','votes','votes_pas_neg','votes_pas_neg_pred','pred_diff']].head()

pred_diff = df_test['pred_diff'].value_counts()


print(f"전체{y_pred.shape[0]}건의 데이터 중 {pred_diff[0]}건 예측")
# 전체1593건의 데이터 중 1273건 예측

acc = (pred_diff[0] / y_pred.shape[0]) *100
print(f'정확도 {acc:.6f}')
# 정확도 79.912116




