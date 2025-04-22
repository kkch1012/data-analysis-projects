# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 16:15:29 2025

@author: Admin
"""

from selenium import webdriver
from bs4 import BeautifulSoup
import pandas as pd
import time

people_mega = pd.read_excel('./starbucks_location/files/people.xlsx')
people_mega.head()
people_mega
driver = webdriver.Chrome()
url = 'https://www.mega-mgccoffee.com/store/find/'
driver.get(url)

local_btn = 'body > div.wrap > div.cont_wrap.find_wrap > div > div.cont_box.find01 > div.map_search_wrap > div > div.cont_text_wrap.map_search_tab_wrap > div > ul > li:nth-child(2)'
driver.find_element('css selector',local_btn).click()


seoul_btn = '#store_area_search_list > li:nth-child(1)'
driver.find_element('css selector',seoul_btn).click()
time.sleep(1)
# BeautifulSoup으로 HTML 파서 만들기
html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')

store_list = []
for i in range(2,27):
    btn_change = "#store_area_search_list_result > li:nth-child({})".format(i)
    driver.find_element('css selector',btn_change).click()
    time.sleep(1)
    
# #map > div:nth-child(1) > div > div:nth-child(6) > div:nth-child(63) > div:nth-child(2) > div > div > div.cont_text.map_point_info_title_wrap > div.cont_text_title.cont_text_inner > b


megacoffee_soup_list = soup.select('li.quickResultLstCon')
megacoffee_soup_list[0]











































