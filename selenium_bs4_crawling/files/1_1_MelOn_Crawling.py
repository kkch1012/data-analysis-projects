# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 09:20:03 2025

@author: Admin

MelOn Crawling => Excel 파일로 저장
"""

from selenium import webdriver
from bs4 import BeautifulSoup

driver = webdriver.Chrome()
url = 'http://www.melon.com/chart/index.htm'
driver.get(url)
html = driver.page_source

soup = BeautifulSoup(html, 'html.parser')

# --------------------------------------
# 반복문을 이용해 곡과 가수명을 song_data에 저장
song_data = []

rank = 1

songs = soup.select('table > tbody > tr')
len(songs)
songs[0]
# 100
for song in songs:
    title = song.select('div.rank01 > span > a')[0].text
    singer = song.select('div.rank02 > span > a')[0].text
    
    song_data.append(['Melon',rank, title, singer])
    
    rank = rank + 1
    
song_data[0]
'''Out[83]: ['Melon', 1, 'REBEL HEART', 'IVE (아이브)']'''

import pandas as pd

columns = ['서비스','순위','타이틀','가수']

pd_data = pd.DataFrame(song_data, columns= columns)
pd_data.head()
pd_data.tail()

# 크롤링 결과를 엑셀파일로 저장
pd_data.to_excel('./files/melon.xlsx', index=False)






url = "https://music.bugs.co.kr/chart"
driver.get(url)
html = driver.page_source

soup = BeautifulSoup(html, 'html.parser')
song_bugs_data = []

rank =1
songs_bugs = soup.select('table.byChart > tbody > tr')
len(songs_bugs)
songs_bugs[0]
for song in songs_bugs:
    title_bugs = song.select('p.title >  a')[0].text
    singer_bugs = song.select('p.artist > a')[0].text
    
    song_bugs_data.append(['Bugs',rank,title_bugs,singer_bugs])
    rank = rank +1

import pandas as pd

columns = ['서비스','순위','타이틀','가수']
pd_data = pd.DataFrame(song_bugs_data, columns= columns)
pd_data.head()
pd_data.tail()
pd_data.to_excel('./files/bugs.xlsx',index=False)









from selenium import webdriver
from bs4 import BeautifulSoup

driver = webdriver.Chrome()

# 1 ~50 위 가지 
url ='https://www.genie.co.kr/chart/top200'
# genie.xlsx
driver.get(url)
html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')

song_genie_data = []
rank = 1

song_genie = soup.select('table > tbody > tr')
len(song_genie)

for song in song_genie:
    title_genie = song.select('a.title')[0].text.strip()
    singer_genie = song.select('a.artist')[0].text
    
    song_genie.append(['Genie',rank,title_genie,singer_genie])
    rank = rank+1
    
url ='https://www.genie.co.kr/chart/top200?ditc=D&ymd=20250218&hh=12&rtm=Y&pg=2' 
songs = soup.select('table > tbody > tr')
for song in songs:
    title = song.select('a.title')[0].text.strip()
    singer = song.select('a.artist')[0].text.strip()
    song_genie.append(['Genie',rank,title,singer])
    rank = rank+1


import pandas as pd

columns = ['서비스','순위','타이틀','가수']
pd_data = pd.DataFrame(song_genie_data, columns= columns)
pd_data.head()
pd_data.tail()
pd_data.to_excel('./files/genie.xlsx',index=False)
































