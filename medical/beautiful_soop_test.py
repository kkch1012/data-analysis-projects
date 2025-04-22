# -*- coding: utf-8 -*-
"""
Created on Mon Feb 17 14:21:23 2025

@author: Admin
"""

# 웹 스크래핑을 하기 위해 HTTP 요청에 사용할 리퀘스트
# HTML 구조 파싱에 사용할 뷰티플수프 라이브러리
import requests
from bs4 import BeautifulSoup

# 문자열을 줄바꿈해서 ㅈ
# HTML 코드
html = '''
<html>
     <body>
          <h1 id="title">파이썬 데이터 부석가 되기<h1>
          <p id ='body'>오늘의 주제는 웹 데이터 수집</p>
          <p class='scraping'>삼성전자 일별 시세 불러오기</p>
          <p class='scraping'>이해쏙쏙</p>
        <.plt.v
</html>
'''
'''
1. BeautifulSoup(읽은 html 문자열, 파싱하는 parser )
   파싱하는 parser : html / xml
2. BeautifulSoup 객체를 변수 저장!
   BeautifulSoup 객체의 함수를 이용하여 데이터 추출 가능
'''

# html.parser로 앞에서 입력한 HTML 코드를 파싱하여 
# 그 결과를 soup에 저장
soup = BeautifulSoup(html, 'html.parser')


# html : 단순 문장열
# Soup = 각각의 요소

for stripped_test in soup.stripped_strings:
    print(stripped_test)

'''
파이썬 데이터 분석가 되기
오늘의 주제는 웹 데이터 수집
삼성전자 일별 시세 불러오기
이해 쏙쏙 selena!'''

# 태그명으로검색 : find('태그명') / find_all('태그명')
# find('태그명') : 동일 태그가 여러 번 반복해 있을 경우, 맨 처음 1개
# find_all('태그명') : 동일 태그가 여러 번 반복해 있을 경우, 모두 ..

first_p = soup.find('p')

all_p = soup.find_all('p')

# id 값이 title인 조건에 해당하는 첫 번째 정보만 검색
title = soup.find(id='title') # = ==> ==

scraping = soup.find(class_= 'scraping')
'''
 <p class="scraping">삼성전자 일별 시세 불러오기</p>
 '''
# class 값이 scraping인 조건에 해당하는 모든 정보 검색
scraping_all = soup.find_all(class_='scraping')
'''
[<p class="scraping">삼성전자 일별 시세 불러오기</p>, <p class="scraping">이해쏙쏙</p>]
'''

# attrs 매개변수 사용 => 속성으로 검색 attrs ={속성명: 값}
# class 속성이 scraping인 첫번째 요소 검색
first_scraping = soup.find(attrs={'class': 'scraping'})

# id 속성이 body인 요소 검색
body_element = soup.find(attrs={'id': 'body'})


### 야후 파이낸스 주가 데이터 웹 스크래핑 ###
# stock_url 변수에 URL 저장
stock_url = 'https://finance.yahoo.com/quote/005930.KS/history/'


# 웹 페이지 요청 : request.get()
res = requests.get(stock_url)
# => <Response [429]>

# 응답에서 HTML 문서만 가져오기 : request.text
html = res.text
# Out[110]: 'Edge: Too Many Requests'

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/130.0.0.0 Safari/537.36',
    'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7'
    }


res = requests.get(stock_url, headers=headers)
html = res.text

soup = BeautifulSoup(html, 'html.parser')

# 뷰티플수프로 tr 요소 모두 찾아오기
print(soup.find_all('tr'))

# 'tr'요소가 있을 경우, 클래스 정보를 가져옴
first_tr = soup.find('tr')
first_class = first_tr.get('class')[0]


# tr 태그에 있는 td 태그중
# class가 first_class('yf-1jecxey)

# 2024년 11월 17일
# 52,000원

# YYYY년 MM월 DD일인 형식으로 날짜(Data) 처리
# strftime('%년 %m월 ,%)
dataf = pd.to_datetime(soup.find_all("td",class_=first_class)[0].text).strftime('%Y년 %m월 %d일') 

# .00을 '원'으로 대체하여 증가(Close) 처리
# replace('.00','원')
closeP = soup.find_all("td",class_=first_class)[4].text.replace('.00','원')

# 55900 원

'''
for문으로 순회하면서 전체 날짜, 종가 데이터 가져오기
출력 결과: 날짜: 2024년 11월 13일 / 종가 : 51,900원
'''

# rows = soup.find_all("tr")  

# for row in rows:
#     cols = row.find_all("td")  
#     if len(cols) >= 2: 
#         date_text = cols[0].text.strip()  
#         price_text = cols[4].text.strip() 
#         date_formatted = pd.to_datetime(date_text).strftime('%Y년 %m월 %d일')
#         close_price = price_text.replace('.00', '원')
#         print(f"날짜: {date_formatted} / 종가: {close_price}")



'''
1. tr 태그 조건에 해당하는 모든 정보 검색
2. 첫 번째 tr 태그(헤더)를 제외하고 순회
    2-1 각 tr 태그 내의 모든 td 태그를 찾기
    2-2 날짜 형식 변환
    2-3. .00 => 원으로 반환
    2-4. 출력
'''


rows = soup.find_all("tr")  # <tr>~~~~~~~~~~~~~~~~</tr>

dates = []
prices = []
for i in range(1, len(rows)):
    cells = rows[i].find_all('td') # <td></td>.......<td></td>
    
    
    if len(cells) == 7:
        # date = pd.to_datetime(cells[0].text).strftime('%Y년 %m월 %d일')
        date = pd.to_datetime(cells[0].text, format='%b %d, %Y')
        close_price = cells[4].text.replace(',','').replace('.00','')

        dates.append(date) # 2025년 02월 17일
        prices.append(int(close_price)) # 56,000원
        
        print(f"날짜 : {date} / 종가는 {close_price}")

'''
cells의 구조

'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

stock_data = pd.DataFrame({'date': dates, 'price': prices})

# y축 눈금 간격 설정
min_price = min(stock_data['price'])
max_price = max(stock_data['price'])
y_ticks = range(min_price,max_price, 3000)

plt.figure(figsize=(10, 5))
plt.plot(stock_data['date'],
         stock_data['price'],
         marker='o',
         label='price')
plt.xlabel('Date')
plt.ylabel('Closing Price (KRW)')
plt.title('KaKao stock Price')
plt.grid(True)
plt.yticks(y_ticks)
plt.show()

# HTML 페이지에서 데이터를 읽어온 후 , 데이터 프레임으로...
from io import StringIO

# 빈 데이터프레임을 생성
stock_data = pd.DataFrame()

response = requests.get(stock_url, headers=headers)

# pd.read_html()
stock_data = pd.read_html(StringIO(str(response.text)),header=0)[0]

stock_data
# 표 형태로 되어잇는 웹페이지를 읽어서 데이터프레임화 시킬때만 사용

# 데이티프레임의 컬럼명 수정

# 날짜 컬럼을 datetime 형식으로 변환

# 결측값 행 제거








































