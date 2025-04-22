# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 09:28:00 2025

@author: Admin
"""

import pandas as pd

df = pd.read_csv("movies04293.csv")
df1 = pd.read_csv("wiki_movie_plots_deduped.csv")
from tensorflow.keras.datasets import imdb

# num_words=10000: 가장 많이 등장한 10,000개의 단어만 사용
(x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=10000)
print(f"리뷰 개수: {len(x_train)}")
print(f"첫 번째 리뷰의 라벨: {y_train[0]}")
print(f"첫 번째 리뷰의 원시 데이터 (정수 인덱스): {x_train[0]}")
