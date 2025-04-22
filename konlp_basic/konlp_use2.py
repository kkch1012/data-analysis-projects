# -*- coding: utf-8 -*-
"""
Created on Fri Mar  7 09:15:13 2025

@author: Admin
"""

import re

# 기호
r = re.compile("a.c")
r.search("kkk")
r.search("abc")

# #기호 : ?앞의 문자가 존재할 수도 있고 존재하지 않을 수도 있는 경우
r = re.compile("ab?c")
r.search("abc")
r.search("ac")
'''
정규 표현식에서의 b는 있다고 취급할 수도 있고, 없다고 취급할 수도 있다.
즉, abc와 ac 모두 매치할 수 있다.
'''

# *기호: *은 바로 앞의 문자가 0개 이상일 경우
r = re.compile("ab*c")
r.search("a") #
r.search("ac")
r.search("abc")
r.search("abbbbc")
r.search("ffbbc") #

# +기호 : *와 유사 / 앞의 문자가 최소 1개 이상
r = re.compile("ab+c")
r.search("ac")
r.search("abc")
r.search("abbbbc")

# ^기호: ^는 시작되는 문자열을 지정
r = re.compile("^ab")
r.search("bbc")
r.search("abz")


# {숫자} 기호: 문자에 해당 기호를 붙이면, 해당 문자를 숫자만큼 반복한 것
r = re.compile("ab{2}c")
r.search("ac")
r.search("abc")
r.search("abbbbbc")
r.search("abbc")

# {숫자1, 숫자2} 기호: 해당 문자를 숫자1 이상 숫자2 이하만큼
r = re.compile("ab{2,8}c")
r.search("ac")
r.search("abc")
r.search("abbbbbc")
r.search("abbc")



r = re.compile("[abc]")
r.search("zzz")
r.search("a")
r.search("aaaaaaa")
r.search("baac")

r = re.compile("[a-z]")
r.search("AAA")
r.search("111")
r.search("aBC")

# [^문자]기호: ^기호 뒤에 붙은 문자들을 제외한 모든 문자
r = re.compile("[^abc]")
r.search("a")
r.search("ab")
r.search("d")
r.search("1")


'''
search()는 정규 표현식 전체에 대해서 문자열이 매치하는지
match()는 문자열의 첫 부분부터 정규 표현식과 매치하는지
'''

# re.split()
text = "사과 딸기 수박 메론 바나나"
re.split(" ", text)
text="""사과
딸기
수박
메론
바나나"""
re.split("\n",text)

text = """이름 : 김철수
전화번호 : 010 -1234 - 1234
나이 : 30
성별 : 남"""

re.findall("\d+",text)

re.findall("\d+","문자열입니다.")

# re.sub() : replace() 동일
# 정제 작업에 많이 사용되는데
# 영어 문장에 각주 등과 같은 이유로 


text = "Regular expression : A regular expression, regex or regexp[1] (sometimes called a rational expression)[2][3] is, in theoretical computer science and formal language theory, a sequence of characters that define a search pattern."
preprocessed_text = re.sub('[^a-zA-Z]',' ',text)
print(preprocessed_text)

preprocessed_text = re.split(' ', preprocessed_text)
preprocessed_text

text = """100 John PROF
101 James STUD
102 Max STUD"""

re.split('\s+',text)

# \d는 숫자에 해당되는 정규표현식
re.findall('\d+',text)

# 텍스트로부터 대문자인 행의 값만
re.findall('[A-Z]',text)

# 대문자가 연속으로 네 번 등장하는 경우라는 조건을 추가
re.findall('[A-Z]{4}',text)

# 처음에 대문자가 등장한 후에 소문자가 여러번 등장하는 경우
re.findall('[A-Z][a-z]+',text)


# 정규 표현식을 이용한 토큰화
# NLTK에서는 정규 표현식을 사용해서 단어 토큰화를 수행하는 RegexTokenizer를 지원
# RegexpTokenizer()에서 괄호 안에ㅔ 하나의 토큰으로 규정하기를 원하는 정규 표현식을 넣어서
# 토큰화를 수행
from nltk.tokenize import RegexpTokenizer

text = "Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."

# 문자 또는 숫자가 1개 이상인 경우 : \w+
tokenizer1 = RegexpTokenizer("[\w]+")

# 공백을 기준으로 토큰화
tokenizer2 = RegexpTokenizer("[\s]+",gaps=True)

print(tokenizer1.tokenize(text))
'''
['Don', 't', 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', 'Mr', 'Jone', 's', 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop']
'''
print(tokenizer2.tokenize(text))
'''
["Don't", 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name,', 'Mr.', "Jone's", 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop.']
'''




import nltk
nltk.download('punkt')
nltk.download('stopwords')

# dictionay 사용하기
from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

raw_text = "A barber is a person. a barber is good person. a barber is huge person. he Knew A Secret! The Secret He Kept is huge secret. Huge secret. His barber kept his word. a barber kept his word. His barber kept his secret. But keeping and keeping such a huge secret to himself was driving the barber crazy. the barber went up a huge mountain."

# 문장 토큰화
sentences = sent_tokenize(raw_text)
'''
['A barber is a person.',
 'a barber is good person.',
 'a barber is huge person.',
 'he Knew A Secret!',
 'The Secret He Kept is huge secret.',
 'Huge secret.',
 'His barber kept his word.',
 'a barber kept his word.',
 'His barber kept his secret.',
 'But keeping and keeping such a huge secret to himself was driving the barber crazy.',
 'the barber went up a huge mountain.']'''

'''
2.단어의 개수를 통일
3. 불용어와 단어 길이가 2이하인 경우 제외
4. 텍스트를 수치화

'''
vocab = {}
preprocessed_sentences = []
stop_words = set(stopwords.words('english'))

for sentence in sentences:
    tokenized_sentence = word_tokenize(sentence)

    result = []  # ✅ 올바른 위치에 배치

    for word in tokenized_sentence:  # ❌ 기존 코드보다 들여쓰기 한 칸 줄임
        word = word.lower()

        if word not in stop_words:
            if len(word) > 2:
                result.append(word)

                if word not in vocab:
                    vocab[word] = 0

                vocab[word] += 1

    preprocessed_sentences.append(result)  # ✅ for 문과 같은 레벨에 있어야 함


print(vocab)
'''
{'barber': 8, 'person': 3, 'good': 1, 'huge': 5, 'knew': 1, 'secret': 6, 'kept': 4, 'word': 2, 'keeping': 2, 'driving': 1, 'crazy': 1, 'went': 1, 'mountain': 1}
'''

vocab_sorted = sorted(vocab.items(),
                      key = lambda x:x[1],
                      reverse = True)
vocab_sorted
'''
[('barber', 8),
 ('secret', 6),
 ('huge', 5),
 ('kept', 4),
 ('person', 3),
 ('word', 2),
 ('keeping', 2),
 ('good', 1),
 ('knew', 1),
 ('driving', 1),
 ('crazy', 1),
 ('went', 1),
 ('mountain', 1)]
'''

word_to_index = {}
i = 0

for (word, frequency) in vocab_sorted:
    if frequency >1:
        i=i+1
        word_to_index[word] = i
        

print(word_to_index)



vocab_size = 5

words_frequency = [word for word, index in word_to_index.items() 
                   if index >= vocab_size + 1]

for w in words_frequency:
    del word_to_index[w]
print(word_to_index)
'''
{'barber': 1, 'secret': 2, 'huge': 3, 'kept': 4, 'person': 5}
'''

word_to_index['OOV'] = len(word_to_index) + 1

encoded_sentences = []
for sentence in preprocessed_sentences:
    encoded_sentence = []
    
    for word in sentence:
        try:
# 단어 집합에 있는 단어라면 해당 단어의 정수를 리턴
            encoded_sentence.append(word_to_index[word])
        except KeyError:
# 만약 단어 집합에 없는 단어라면 ' 의 정수를 리턴
            encoded_sentence.append(word_to_index['OOV'])
    encoded_sentences.append(encoded_sentence)

print(encoded_sentences)


from collections import Counter

print(preprocessed_sentences)

all_words_list = sum(preprocessed_sentences, [])
print(all_words_list)

vocab = Counter(all_words_list)
print(vocab)
vocab_size = 5
vocab = vocab.most_common(vocab_size) #

word_to_index = {}
i = 0

for (word, frequency) in vocab :
    i = i + 1
    word_to_index[word] = i
print(word_to_index)


word_to_index['OOV'] = len(word_to_index) + 1

encoded_sentences = []
for sentence in preprocessed_sentences:
    encoded_sentence = []
    
    for word in sentence:
        try:
# 단어 집합에 있는 단어라면 해당 단어의 정수를 리턴
            encoded_sentence.append(word_to_index[word])
        except KeyError:
# 만약 단어 집합에 없는 단어라면 ' 의 정수를 리턴
            encoded_sentence.append(word_to_index['OOV'])
    encoded_sentences.append(encoded_sentence)

print(encoded_sentences)



from nltk import FreqDist
import numpy as np

# np.hstack으로 문장 구분을 제거
vocab = FreqDist(np.hstack(preprocessed_sentences))

print(vocab["barber"]) # 'barber'

vocab_size = 5
vocab = vocab.most_common(vocab_size)
# 등장 빈도수가 높은 상위 5 개의 단어만 저장
print(vocab)

word_to_index = {word[0] : index + 1 for index, word in enumerate(vocab)}
print(word_to_index)



test_input = ['a', 'b', 'c', 'd', 'e']
for index, value in enumerate(test_input): #입력의 순서대로 0 부터 인덱스를 부여함
    print("value : {}, index: {}".format(value, index))


from tensorflow.keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
preprocessed_sentences = [['barber', 'person'], ['barber', 'good', 'person'], ['barber', 'huge', 'person'], ['knew', 'secret'], ['secret', 'kept', 'huge', 'secret'], ['huge', 'secret'], ['barber', 'kept', 'word'], ['barber', 'kept', 'word'], ['barber', 'kept', 'secret'], ['keeping', 'keeping', 'huge', 'secret', 'driving', 'barber', 'crazy'], ['barber', 'went', 'huge', 'mountain']]

tokenizer.fit_on_texts(preprocessed_sentences)

print(tokenizer.word_index)

print(tokenizer.word_counts)

print(tokenizer.texts_to_sequences(preprocessed_sentences))


vocab_size = 5

tokenizer = Tokenizer(num_words = vocab_size + 1)
# 상위 5 개 단어만 사용
tokenizer.fit_on_texts(preprocessed_sentences)

print(tokenizer.word_index)
print(tokenizer.word_counts)
print(tokenizer.texts_to_sequences(preprocessed_sentences))


tokenizer = Tokenizer()
tokenizer.fit_on_texts(preprocessed_sentences)
vocab_size = 5
words_frequency = [word for word, index in tokenizer.word_index.items() if index >= vocab_size + 1]
#인덱스가 5 초과인 단어 제거
for word in words_frequency:
    del tokenizer.word_index[word]
    del tokenizer.word_counts[word] # 해당 단어에 대한 카운트 정보를 삭제


print(tokenizer.word_index)

print(tokenizer.word_counts)

print(tokenizer.texts_to_sequences(preprocessed_sentences))

import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer












