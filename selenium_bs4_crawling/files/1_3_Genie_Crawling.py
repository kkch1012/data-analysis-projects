# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 09:21:06 2025

@author: Admin
"""

from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd

driver = webdriver.Chrome()

# 1 ~ 50위 가지 
url = 'https://www.genie.co.kr/chart/top200'
driver.get(url)
html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')

song_genie_data = []
rank = 1

# 첫 번째 페이지
song_genie = soup.select('table > tbody > tr')
for song in song_genie:
    title_genie = song.select('a.title')[0].text.strip()
    singer_genie = song.select('a.artist')[0].text.strip()
    song_genie_data.append(['Genie', rank, title_genie, singer_genie])
    rank = rank + 1

# 두 번째 페이지
url = 'https://www.genie.co.kr/chart/top200?ditc=D&ymd=20250218&hh=12&rtm=Y&pg=2'
driver.get(url)
html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')

songs = soup.select('table > tbody > tr')
for song in songs:
    title = song.select('a.title')[0].text.strip()
    singer = song.select('a.artist')[0].text.strip()
    song_genie_data.append(['Genie', rank, title, singer])
    rank = rank + 1

# 데이터를 엑셀로 저장
columns = ['서비스', '순위', '타이틀', '가수']
pd_data = pd.DataFrame(song_genie_data, columns=columns)
pd_data.head()
pd_data.tail()
pd_data.to_excel('./files/genie.xlsx', index=False)

driver.quit()
