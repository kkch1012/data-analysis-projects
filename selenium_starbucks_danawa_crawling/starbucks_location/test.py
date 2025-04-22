from selenium import webdriver
from bs4 import BeautifulSoup
import time
import pandas as pd
# Selenium 드라이버 설정
driver = webdriver.Chrome()
url = 'https://www.mega-mgccoffee.com/store/find/'
driver.get(url)

# 지역 선택 버튼 클릭
local_btn = 'body > div.wrap > div.cont_wrap.find_wrap > div > div.cont_box.find01 > div.map_search_wrap > div > div.cont_text_wrap.map_search_tab_wrap > div > ul > li:nth-child(2)'
driver.find_element('css selector', local_btn).click()

# 서울 선택 버튼 클릭
seoul_btn = '#store_area_search_list > li:nth-child(1)'
driver.find_element('css selector', seoul_btn).click()
time.sleep(1)

# BeautifulSoup으로 HTML 파서 만들기
html = driver.page_source
soup = BeautifulSoup(html, 'html.parser')

# store_list를 담을 리스트 생성
store_list = []

# li 태그를 클릭하며 내용 확인
for i in range(2, 27):
    # 각 버튼 클릭
    btn_change = "#store_area_search_list_result > li:nth-child({})".format(i)
    driver.find_element('css selector', btn_change).click()
    time.sleep(1)
    
    # 현재 페이지의 HTML을 다시 가져와서 BeautifulSoup 파싱
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    
    # 각 li 태그 선택
    store_items = soup.select('li.quickResultLstCon')

    # 각 store_item의 텍스트 출력
    for store in store_items:
        print(store.text.strip())  # 텍스트만 출력 (공백 제거)

    # store_list에 원하는 정보를 추가 (예: 텍스트 또는 링크 등)
    for store in store_items:
        store_list.append(store.text.strip())  # 텍스트를 store_list에 추가

# 최종적으로 store_list 내용 출력
print(store_list)
