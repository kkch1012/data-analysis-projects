# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 09:09:20 2025

@author: Admin

pip install selenium
"""

from selenium import webdriver

# 크롬 부라우저 실행
driver = webdriver.Chrome()

# URL 접속
url = 'https://www.naver.com/'
driver.get(url)

# 웹페이지 html 다운로드
html = driver.page_source













































































