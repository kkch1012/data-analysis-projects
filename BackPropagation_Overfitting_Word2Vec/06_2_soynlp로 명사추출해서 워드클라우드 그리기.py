# -*- coding: utf-8 -*-
"""
Created on Tue Mar 18 09:19:25 2025

@author: Admin
"""

import pandas as pd
import numpy as np
import re
import os
import platform

### 데이터 불러오기 ###
base_path = "data"
file_name = "petition.csv"

df = pd.read_csv(f"{base_path}/petition.csv",
                 index_col="article_id",
                 parse_dates=['start','end'])


df.shape
df.tail()


'''
자신의 관심사에 있는 단어로 데이터를 가져오려면
파이썬에서 re(정규식) 모듈에서 제공하는 match() 함수

match() 함수 
문자열이 정규식에서 지정된 패턴과 일치하는지 확인하기 위해
처음부터 문자열을 검색한 후 일치하는 단어를 반환


'''
# 돌봄 육아 초등 보육 등의 키워드가 들어 있는 타이틀을 추출
p = r'.*(돌봄|육아|초등|보육).*'

care = df[df['title'].str.match(p) | df['content'].str.match(p, flags=re.MULTILINE)] 

care.shape
care.head(2)

sample_index= 24
sample_title = care.loc[sample_index, 'title']
sample_content = care['content'][sample_index]


from soynlp.tokenizer import RegexTokenizer

tokenizer = RegexTokenizer()
tokened_title = tokenizer.tokenize(sample_title)

tokened_content = tokenizer.tokenize(sample_content)


content_text = care['content'].str.replace("\\\\n"," ",regex=True)

# 한글과 영문자가 아닌 불필요한 문자들도 삭제
content_text = content_text.str.replace("[^ㄱ-하-ㅣ가-힣a-zA-Z]"," ",regex=True)

tokens = content_text.apply(tokenizer.tokenize)
tokens[:3]
tokens[sample_index][:10]

from wordcloud import WordCloud
import matplotlib.pyplot as plt

stopwords = ['하지만','그리고','그런데','저는','제가','그럼','이런','저런','합니다','많은','많이','정말','너무']

def display_word_cloud(data, width=1200, height=500):
    word_draw = WordCloud(
        font_path=r"/Library/Fonts/NanumGothic.ttf",
        width=width, height=height,
        stopwords=stopwords,
        background_color="white",
        random_state=42
        )
    word_draw.generate(data)
    
    plt.figure(figsize=(15,7))
    plt.imshow(word_draw)
    plt.axis("off")
    plt.show()
    
display_word_cloud(' '.join(content_text))


# 명사만 추출하여 시각화


from soynlp.noun import LRNounExtractor

noun_extractor = LRNounExtractor(verbose=True)

noun_extractor.train(content_text)

nouns = noun_extractor.extract()

nouns_text = " ".join(list(nouns.keys()))
nouns_text[:100]

display_word_cloud(nouns_text)




















