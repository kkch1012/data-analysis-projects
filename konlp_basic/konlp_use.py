# -*- coding: utf-8 -*-
"""
Created on Thu Mar  6 11:40:42 2025

@author: Admin
"""

from nltk.tokenize import word_tokenize
from nltk.tokenize import WordPunctTokenizer
from tensorflow.keras.preprocessing.text import text_to_word_sequence
import nltk

nltk.download('punkt_tab')



from konlpy.tag import Okt
from konlpy.tag import Kkma

okt = Okt()
kkma = Kkma()

# word_tokenize
print('단어 토큰화1:',word_tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))
'''
단어 토큰화1: ['Do', "n't", 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', ',', 'Mr.', 'Jone', "'s", 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop', '.']
'''

# WordPunctTokenizer
print('단어 토큰화2 :', WordPunctTokenizer().tokenize("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))
'''
단어 토큰화2 : ['Don', "'", 't', 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', ',', 'Mr', '.', 'Jone', "'", 's', 'Orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop', '.']
'''

# text_to_word_sequence
print('단어 토큰화3 :', text_to_word_sequence("Don't be fooled by the dark sounding name, Mr. Jone's Orphanage is as cheery as cheery goes for a pastry shop."))
'''
단어 토큰화3 : ["don't", 'be', 'fooled', 'by', 'the', 'dark', 'sounding', 'name', 'mr', "jone's", 'orphanage', 'is', 'as', 'cheery', 'as', 'cheery', 'goes', 'for', 'a', 'pastry', 'shop']
'''

'''
토큰화에서 고려해야할 사항

'''
from nltk.tokenize import TreebankWordTokenizer

tokenizer = TreebankWordTokenizer()

text = "Starting a home-based restaurant may be an ideal. it doesn't have a food chain or restaurant of their own"

print('트리뱅크 워드토크나이저', tokenizer.tokenize(text))

### 문장 토큰화(Sentence Tokenization) ###

'''
문장 분류
=> 코퍼스 내에서 문장 단위로 구분하는 작업
=> 코퍼스는 문장 단위로 구분되어 있지 않아서
=> 사용하고자 하는 용도에 맞게 문장 토큰화

?나 마침표나 !기준으로 문장을 잘라내면 되지 않을까
=> IP 192.168.56.31 서버에 들어가서 로그 파일 저장해서 aaa@gmail.com로 결과 좀 보내줘.
그 후에 점심 먹으러 가자

사용하는 코퍼스가 어떤 국적의 언어인지
해당 코퍼스 내에서 특수문자들이 어떻게 사용되고 있는지

NLTK에서는 영어 문장의 토큰화를 수행하는 sent_tokenize 지원
'''

from nltk.tokenize import sent_tokenize

text = "His barber kept his word. But keeping such a huge secret to himself was driving him crazy. Finally, the barber went up a mountain and almost to the edge of a cliff. He dug a hole in the midst of some reeds. He looked about, to make sure no one was near."
print('문장 토큰화1 : ',sent_tokenize(text))



text = "I am actively looking for Ph.D. students. and you are a Ph.D student."
print('문장 토큰화1 : ',sent_tokenize(text))

text = '딥 러닝 자연어 처리가 재미있기는 합니다. 그런데 문제는 영어보다 한국어로 할 때 너무 어렵습니다. 이제 해보면 알걸요?'
print('문장 토큰화1 : ',sent_tokenize(text))

# 한국어의 경우 : KSS(Korean Sentence Splitter)

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag

text = "I am actively looking for Ph.D. students. and you are a Ph.D student."

tokenized_sentence = word_tokenize(text)

print('단어 토큰화:',tokenized_sentence)
'''
단어 토큰화: ['I', 'am', 'actively', 'looking', 'for', 'Ph.D.', 'students', '.', 'and', 'you', 'are', 'a', 'Ph.D', 'student', '.']
'''

from konlpy.tag import Okt
from konlpy.tag import Kkma

okt =Okt()
kkma = Kkma()

print('OKT 형태소 분석 :',okt.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))

print('OKT 품사 태깅:',okt.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
'''
OKT 품사 태깅: [('열심히', 'Adverb'), ('코딩', 'Noun'), ('한', 'Josa'), ('당신', 'Noun'), (',', 'Punctuation'), ('연휴', 'Noun'), ('에는', 'Josa'), ('여행', 'Noun'), ('을', 'Josa'), ('가봐요', 'Verb')]
'''

print('OKT 형태소 분석 :',okt.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))

print('OKT 형태소 분석 :',kkma.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))
'''
OKT 형태소 분석 : ['열심히', '코딩', '하', 'ㄴ', '당신', ',', '연휴', '에', '는', '여행', '을', '가보', '아요']
'''
print('OKT 품사 태깅:',kkma.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))


print('OKT 품사 태깅:',kkma.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요"))

import nltk
nltk.download('wordnet')
nltk.download('punkt')

from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()

words = ['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
print('표제어 추출 전:',words)
print('표제어 추출 후:',[lemmatizer.lemmatize(word) for word in words])

'''
표제어 추출기(lemmatizer)가 본래 단어의 품사 정보를 알아야만
정확한 결과를 얻을 수 있기 때문
'''

lemmatizer.lemmatize('dies','v')
lemmatizer.lemmatize('watched','v')
lemmatizer.lemmatize('has','v')

'''
표제어 추출
'''

from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

stemmer = PorterStemmer()

sentence = "This was not the map we found in Billy Bones's chest, but an accurate copy, complete in all things--names and heights and soundings--with the single exception of the red crosses and the written notes."

tokenized_sentence = word_tokenize(sentence)
print('어간 추출 전: ',tokenized_sentence)
print('어간 추출 후: ',[stemmer.stem(word) for word in tokenized_sentence])

'''
정해진 규칙 기반의 접근
=> 어간 추출 후의 결과에는 사전에 없는 결과들을 포함

formalize -> fomral
allowance -> allow
electricical -> electric
'''
words = ['formalize','allowance','electricical']
print('어간 추출 전: ',words)
print('어간 추출 후:',[stemmer.stem(word) for word in words])

### 포터 알고리즘과 랭커스터 스태머 알고리즘
from nltk.stem import LancasterStemmer

porter_stemmer = PorterStemmer()
lancaster_stemmer = LancasterStemmer()

words = ['policy', 'doing', 'organization', 'have', 'going', 'love', 'lives', 'fly', 'dies', 'watched', 'has', 'starting']
print(words)
print('포터 스테머의 어간 추출 후:',[porter_stemmer.stem(w) for w in words])
'''
포터 스테머의 어간 추출 후: ['polici', 'do', 'organ', 'have', 'go', 'love', 'live', 'fli', 'die', 'watch', 'ha', 'start']
'''
print('렝커 스테머의 어간 추출 후:',[lancaster_stemmer.stem(w) for w in words])
'''
렝커 스테머의 어간 추출 후: ['policy', 'doing', 'org', 'hav', 'going', 'lov', 'liv', 'fly', 'die', 'watch', 'has', 'start']
'''


'''
한국어에서의 어간 추출

체언: 명사, 대명사, 수사
수식언: 관형사, 부사
관계언: 조사
독립언: 감탄사
용언: 동사, 형용사

용언 => 동사,헝용사는 어간 + 어미의 결합

1) 활용: 용언의 어간이 어미를 가지는 일
한국어에서만 가지는 특징

어간(stem) / 어미(ending)

2) 규칙 활용: 어간이 어미를 취할 때, 어간의 모습이 일정

3) 불규칙 활용
어간이 어미를 취할 때 어간의 모습이 바뀌거나 취하는 어미가 특수한 어미일 경우

'''

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from konlpy.tag import Okt


nltk.download('stopwords')
nltk.download('punkt')

# NLTK에서 불용어 확인
stop_word_list = stopwords.words('english')
print('불용어 개수: ',len(stop_word_list))
print('불용어 10개 출력: ',stop_word_list[:10])
'''
불용어 10개 출력:  ['a', 'about', 'above', 'after', 'again', 'against', 'ain', 'all', 'am', 'an']
'''
example = "Family is not an important thing. It's everything."
stop_words = set(stopwords.words('english'))

word_tokens = word_tokenize(example)
print('불용어 제거 전:',word_tokens)
'''
불용어 제거 전: ['Family', 'is', 'not', 'an', 'important', 'thing', '.', 'It', "'s", 'everything', '.']
'''

result = []
for word in word_tokens:
    if word not in stop_words:
        result.append(word)

print('불용어 제거 후:',result)    

okt = Okt()    
    
example = "고기를 아무렇게나 구우려고 하면 안 돼. 고기라고 다 같은 게 아니거든. 예컨대 삼겹살을 구울 때는 중요한 게 있지."
stop_words = "를 아무렇게나 구 우려 고 안 돼 같은 게 구울 때 는"

stop_words = set(stop_words.split(' '))    
word_tokens = okt.morphs(example)    
print('불용어 제거 전:',word_tokens)    

result = []
for word in word_tokens:
    if word not in stop_words:
        result.append(word)
print('불용어 제거 후: ',result)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    














