# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 12:10:03 2025

@author: Admin


"""

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer

preprocessed_sentences = [['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 'person'], ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'], ['huge', 'secret'], ['barber', 'kept', 'word'], ['barber', 'kept', 'word'], ['barber', 'kept', 'secret'], ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'], ['barber', 'went', 'huge', 'mountain']]


tokenizer = Tokenizer()
tokenizer.fit_on_texts(preprocessed_sentences)
encoded = tokenizer.texts_to_sequences(preprocessed_sentences)
print(encoded)

max_len =max(len(item) for item in encoded)
print('최대 길이 :',max_len)

for sentence in encoded:
    while len(sentence) < max_len:
        sentence.append(0)

padded_np = np.array(encoded)
padded_np



from tensorflow.keras.preprocessing.sequence import pad_sequences

encoded = tokenizer.texts_to_sequences(preprocessed_sentences)
print(encoded)

padded = pad_sequences(encoded)

padded = pad_sequences(encoded, padding='post')
padded
'''
array([[ 1,  5,  0,  0,  0,  0,  0],
       [ 1,  8,  5,  0,  0,  0,  0],
       [ 1,  3,  5,  0,  0,  0,  0],
       [ 9,  2,  0,  0,  0,  0,  0],
       [ 2,  4,  3,  2,  0,  0,  0],
       [ 3,  2,  0,  0,  0,  0,  0],
       [ 1,  4,  6,  0,  0,  0,  0],
       [ 1,  4,  6,  0,  0,  0,  0],
       [ 1,  4,  2,  0,  0,  0,  0],
       [ 7,  7,  3,  2, 10,  1, 11],
       [ 1, 12,  3, 13,  0,  0,  0]])
'''
(padded == padded_np).all()

padded = pad_sequences(encoded, padding='post', maxlen=5)

'''
array([[ 0,  0,  0,  0,  0],
       [ 5,  0,  0,  0,  0],
       [ 5,  0,  0,  0,  0],
       [ 0,  0,  0,  0,  0],
       [ 3,  2,  0,  0,  0],
       [ 0,  0,  0,  0,  0],
       [ 6,  0,  0,  0,  0],
       [ 6,  0,  0,  0,  0],
       [ 2,  0,  0,  0,  0],
       [ 3,  2, 10,  1, 11],
       [ 3, 13,  0,  0,  0]])
'''

padded = pad_sequences(encoded, padding='post', truncating='post', maxlen=5)
padded
'''
array([[ 1,  5,  0,  0,  0],
       [ 1,  8,  5,  0,  0],
       [ 1,  3,  5,  0,  0],
       [ 9,  2,  0,  0,  0],
       [ 2,  4,  3,  2,  0],
       [ 3,  2,  0,  0,  0],
       [ 1,  4,  6,  0,  0],
       [ 1,  4,  6,  0,  0],
       [ 1,  4,  2,  0,  0],
       [ 7,  7,  3,  2, 10],
       [ 1, 12,  3, 13,  0]])
'''

last_value = len(tokenizer.word_index) + 1 # 단어 집합의 크기보다 1 큰 숫자를 사용
print(last_value)


padded = pad_sequences(encoded, padding='post', value=last_value)
padded
'''
array([[ 1,  5,  0,  0,  0,  0,  0],
       [ 1,  8,  5,  0,  0,  0,  0],
       [ 1,  3,  5,  0,  0,  0,  0],
       [ 9,  2,  0,  0,  0,  0,  0],
       [ 2,  4,  3,  2,  0,  0,  0],
       [ 3,  2,  0,  0,  0,  0,  0],
       [ 1,  4,  6,  0,  0,  0,  0],
       [ 1,  4,  6,  0,  0,  0,  0],
       [ 1,  4,  2,  0,  0,  0,  0],
       [ 7,  7,  3,  2, 10,  1, 11],
       [ 1, 12,  3, 13,  0,  0,  0]])
'''

from konlpy.tag import Okt

okt =Okt()

tokens = okt.morphs("나는 자연어 처리를 배운다")

word_to_index = {word : index for index, word in enumerate(tokens)}
'''
Out[180]: {'나': 0, '는': 1, '자연어': 2, '처리': 3, '를': 4, '배운다': 5}
'''
# 특정 단어에 대한 원-핫-인코딩

def one_hot_encoding(word, word_to_index):
    
    one_hot_vector = [0]*(len(word_to_index))
    index = word_to_index[word]
    one_hot_vector[index] = 1
    return one_hot_vector

one_hot_encoding("처리",word_to_index)

# 케라스를 이용한 원-핫 인코딩 : to_categorical()
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical

text = "나랑 점심 먹으러 갈래 점심 메뉴는 햄버거 갈래 갈래 햄버거 최고야"

tokenizer = Tokenizer()
tokenizer.fit_on_texts([text])

print('단어 집합: ',tokenizer.word_index)
'''
단어 집합:  {'갈래': 1, '점심': 2, '햄버거': 3, '나랑': 4, '먹으러': 5, '메뉴는': 6, '최고야': 7}
'''

# 케라스는 정수 인코딩 된 결과로부터
# 원-핫 인토딩을 수행하는 to_categorical()를 지원

sub_text = "점심 먹으러 갈래 메뉴는 햄버거 최고야"
encoded = tokenizer.texts_to_sequences([sub_text])[0]
'''
Out[195]: [2, 5, 1, 6, 3, 7]
'''

one_hot = to_categorical(encoded)
'''
array([[0., 0., 1., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 1., 0., 0.],
       [0., 1., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 1., 0.],
       [0., 0., 0., 1., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 1.]])
'''









































